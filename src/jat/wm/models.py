"""The two trained parts of the world model, both small and both on top of frozen V-JEPA 2 latents.

- Translator: latent -> pose. The LLM only ever reads the translator's output.
- Predictor: (latent, push) -> next latent. Imagination happens here, in latent space.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from jat.worlds import get_world
from jat.worlds.base import TABLE, World
from jat.worlds.tblock import shape as geo

HALF = TABLE / 2
_LANDMARKS = torch.tensor(list(geo.LANDMARKS.values()), dtype=torch.float32)  # [4, 2] local px


# Pose encodings. "Output" = the translator's 4 numbers: x, y scaled to [-1, 1], cos, sin.


def pose_to_output(poses: torch.Tensor) -> torch.Tensor:
    """[B, 3] (x px, y px, theta) -> [B, 4]."""
    x, y, theta = poses.unbind(-1)
    return torch.stack([x / HALF - 1, y / HALF - 1, torch.cos(theta), torch.sin(theta)], -1)


def output_to_pose(out: torch.Tensor) -> torch.Tensor:
    """[B, 4] -> [B, 3] (x px, y px, theta)."""
    return torch.stack(
        [(out[:, 0] + 1) * HALF, (out[:, 1] + 1) * HALF, out[:, 3].atan2(out[:, 2])], -1
    )


def output_keypoints(out: torch.Tensor) -> torch.Tensor:
    """[B, 4] -> the 4 aim landmarks [B, 4, 2], in table px / HALF. Differentiable."""
    cs = F.normalize(out[:, 2:], dim=-1)
    c, s = cs[:, :1], cs[:, 1:]
    local = _LANDMARKS.to(out) / HALF  # [4, 2]
    lx, ly = local[:, 0], local[:, 1]
    world_x = out[:, :1] + c * lx - s * ly
    world_y = out[:, 1:2] + s * lx + c * ly
    return torch.stack([world_x, world_y], -1)


def keypoint_loss(out: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """Distance between predicted and true landmarks. Landmarks, not (x, y, theta): one loss
    in one unit, and no angle wrap-around."""
    target = output_keypoints(pose_to_output(poses))
    return (
        F.smooth_l1_loss(output_keypoints(out), target, beta=0.02)
        + 0.01 * (out[:, 2:].norm(dim=-1) - 1).pow(2).mean()
    )


class Translator(nn.Module):
    """latent [B, P, D] -> pose output [B, 4]. Each token is squeezed to a few features, then
    one MLP reads all tokens in place: token order carries the position on the table."""

    def __init__(self, n_tokens: int, dim: int, token_features: int = 32, hidden: int = 512):
        super().__init__()
        self.config = dict(n_tokens=n_tokens, dim=dim, token_features=token_features, hidden=hidden)
        self.squeeze = nn.Linear(dim, token_features)
        self.mlp = nn.Sequential(
            nn.Linear(n_tokens * token_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.squeeze(z).flatten(1))


class Predictor(nn.Module):
    """(latent [B, P, D], push [B, 4] in [-1, 1]) -> next latent [B, P, D].

    A transformer over the P tokens plus one push token. It predicts the change, and the output
    layer starts at zero, so an untrained predictor says "nothing moves".
    """

    def __init__(self, n_tokens: int, dim: int, width: int = 384, depth: int = 6, heads: int = 6):
        super().__init__()
        self.config = dict(n_tokens=n_tokens, dim=dim, width=width, depth=depth, heads=heads)
        self.embed = nn.Linear(dim, width)
        self.position = nn.Parameter(torch.randn(1, n_tokens, width) * 0.02)
        self.push = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor, push: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([self.push(push).unsqueeze(1), self.embed(z) + self.position], 1)
        h = self.norm(self.blocks(tokens))[:, 1:]
        return z + self.out(h)


class InverseDynamics(nn.Module):
    """(latent, next latent) -> push [B, 4]. Training aid only: it forces predicted latents to
    show which push made them, so the predictor cannot ignore the push."""

    def __init__(self, n_tokens: int, dim: int, token_features: int = 16, hidden: int = 512):
        super().__init__()
        self.squeeze = nn.Linear(2 * dim, token_features)
        self.mlp = nn.Sequential(
            nn.Linear(n_tokens * token_features, hidden), nn.GELU(), nn.Linear(hidden, 4)
        )

    def forward(self, z: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.squeeze(torch.cat([z, z_next - z], -1)).flatten(1))


def save_module(module: nn.Module, path: Path, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": module.config, "state": module.state_dict(), **extra}, path)


def load_module(cls: type[nn.Module], path: Path, device: torch.device | str = "cpu"):
    """(module in eval mode, the checkpoint's extra fields)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    module = cls(**checkpoint.pop("config"))
    module.load_state_dict(checkpoint.pop("state"))
    return module.to(device).eval(), checkpoint


class WorldModel:
    """Encoder + predictor + translator: what the `simulate` tool runs on.

    `encoder` is anything with `encode_states(world, states) -> [B, P, D]`; the real one is
    `jat.wm.encoder.Encoder`. `world` renders the real frames the encoder sees.
    """

    def __init__(self, world: World, encoder, predictor: Predictor, translator: Translator):
        self.world = world
        self.encoder, self.predictor, self.translator = encoder, predictor, translator
        self.device = next(predictor.parameters()).device

    @classmethod
    def load(cls, run: Path, device: torch.device | str = "cpu") -> "WorldModel":
        from jat.wm.encoder import Encoder

        translator, meta = load_module(Translator, run / "translator.pt", device)
        predictor, _ = load_module(Predictor, run / "predictor.pt", device)
        latents = meta["latents"]
        encoder = Encoder(latents["encoder"], latents["pool"], torch.device(device))
        # Runs trained before there were several worlds have no name: they are PushT.
        return cls(get_world(latents.get("world", "pusht")), encoder, predictor, translator)

    @torch.no_grad()
    def encode(self, states) -> torch.Tensor:
        return self.encoder.encode_states(self.world, states).to(self.device)

    @torch.no_grad()
    def step(self, z: torch.Tensor, pushes: np.ndarray) -> torch.Tensor:
        """pushes: [B, 4] in [-1, 1] (Push.numeric())."""
        return self.predictor(z, torch.as_tensor(pushes, dtype=torch.float32, device=self.device))

    @torch.no_grad()
    def read(self, z: torch.Tensor) -> np.ndarray:
        """[B, 3] poses (x px, y px, theta)."""
        return output_to_pose(self.translator(z)).cpu().numpy().astype(float)
