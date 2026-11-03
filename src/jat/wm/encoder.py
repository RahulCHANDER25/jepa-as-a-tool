"""Frame -> V-JEPA 2 latent. The encoder stays frozen; only its output tokens are used.

A latent is [P, D]: P patch tokens (16 x 16 at 256 px, or 8 x 8 after 2 x 2 pooling) of
D = 1024 features (ViT-L).
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jat.wm.data import Rollouts
from jat.worlds import get_world
from jat.worlds.base import World

VJEPA2_ID = "facebook/vjepa2-vitl-fpc64-256"
FRAME_SIZE = 256
GRID = FRAME_SIZE // 16  # patch grid side before pooling
_MEAN = (0.485, 0.456, 0.406)  # ImageNet, from the model's video_preprocessor_config.json
_STD = (0.229, 0.224, 0.225)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder:
    """Renders states and encodes them with V-JEPA 2, the way V-JEPA 2-AC encodes one frame:
    the frame is repeated into a 2-frame clip (one tubelet), and each token is layer-normed
    (vjepa2 repo, app/vjepa_droid/train.py, forward_target)."""

    def __init__(
        self, model_id: str = VJEPA2_ID, pool: int = 2, device: torch.device | None = None
    ):
        from transformers import VJEPA2Model

        self.model_id, self.pool = model_id, pool
        self.device = device or default_device()
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        # Cast after loading: the dtype keyword changed name across transformers versions.
        model = VJEPA2Model.from_pretrained(model_id)
        self.model = model.to(device=self.device, dtype=dtype).eval()
        self.dtype = dtype
        self.n_tokens = (GRID // pool) ** 2
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        """uint8 [B, 256, 256, 3] -> float32 [B, P, D] on the encoder's device."""
        x = torch.from_numpy(frames).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)
        # Our own normalization, not the HF processor: it resizes to 292 and center-crops 256,
        # which would cut the table edges off.
        x = ((x - mean) / std).to(self.dtype)
        clip = x.unsqueeze(1).repeat(1, 2, 1, 1, 1)  # [B, T=2, C, H, W]
        tokens = self.model.get_vision_features(clip).float()  # [B, GRID * GRID, D]
        return pool_tokens(tokens, self.pool)

    def encode_states(self, world: World, states) -> torch.Tensor:
        frames = np.stack([world.render(s, FRAME_SIZE) for s in states])
        return self.encode_frames(frames)


def pool_tokens(tokens: torch.Tensor, pool: int) -> torch.Tensor:
    """Average `pool` x `pool` patch blocks, then layer-norm each token."""
    if pool > 1:
        b, _, d = tokens.shape
        grid = tokens.transpose(1, 2).reshape(b, d, GRID, GRID)
        tokens = F.avg_pool2d(grid, pool).flatten(2).transpose(1, 2)
    return F.layer_norm(tokens, (tokens.shape[-1],))


def encode_dataset(folder: Path, encoder: Encoder, batch_size: int = 64) -> None:
    """Writes latents.npy + latents.json next to rollouts.npz."""
    rollouts = Rollouts.load(folder)
    world = get_world(rollouts.world)
    states = rollouts.poses.reshape(-1, rollouts.poses.shape[-1])
    shape = (*rollouts.poses.shape[:2], encoder.n_tokens, encoder.dim)
    latents = np.lib.format.open_memmap(
        folder / "latents.npy", mode="w+", dtype=np.float16, shape=shape
    )
    flat = latents.reshape(-1, encoder.n_tokens, encoder.dim)
    for i in range(0, len(states), batch_size):
        batch = states[i : i + batch_size]
        flat[i : i + batch_size] = encoder.encode_states(world, batch).cpu().numpy()
        if (i // batch_size) % 50 == 0:
            print(f"encoded {i + len(batch)}/{len(states)} frames", flush=True)
    latents.flush()
    meta = {
        "world": world.name,
        "encoder": encoder.model_id,
        "pool": encoder.pool,
        "shape": list(shape),
    }
    (folder / "latents.json").write_text(json.dumps(meta, indent=1))


def load_latents(folder: Path) -> tuple[np.ndarray, dict]:
    """(memmap fp16 [E, T+1, P, D], latents.json)."""
    meta = json.loads((folder / "latents.json").read_text())
    return np.load(folder / "latents.npy", mmap_mode="r"), meta
