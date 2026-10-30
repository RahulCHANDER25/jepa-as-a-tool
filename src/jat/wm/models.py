"""The two trained parts of the world model, both small and both on top of frozen V-JEPA 2 latents.

- Translator: latent -> world state. The LLM only ever reads the translator's output. One trunk
  shared by every world, one small head per world.
- Predictor: (latent, push) -> next latent. Imagination happens here, in latent space. One per
  world: the physics differ.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from jat.wm.readouts import readout
from jat.worlds import get_world
from jat.worlds.base import World


class Translator(nn.Module):
    """latent [B, P, D] -> world state output [B, size of the world's readout].

    Trunk: each token is squeezed to a few features, then one MLP reads all tokens in place
    (token order carries the position in the frame). Heads: one linear layer per world.
    """

    def __init__(
        self,
        n_tokens: int,
        dim: int,
        heads: dict[str, int],
        token_features: int = 32,
        hidden: int = 512,
    ):
        super().__init__()
        self.config = dict(
            n_tokens=n_tokens, dim=dim, heads=heads, token_features=token_features, hidden=hidden
        )
        self.squeeze = nn.Linear(dim, token_features)
        self.mlp = nn.Sequential(
            nn.Linear(n_tokens * token_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict({w: nn.Linear(hidden, size) for w, size in heads.items()})

    def trunk(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.squeeze(z).flatten(1))

    def forward(self, z: torch.Tensor, world: str) -> torch.Tensor:
        return self.heads[world](self.trunk(z))

    def trunk_parameters(self) -> list[nn.Parameter]:
        return [*self.squeeze.parameters(), *self.mlp.parameters()]

    @staticmethod
    def upgrade(checkpoint: dict) -> dict:
        """A checkpoint from before the heads: one PushT readout as the MLP's last layer."""
        if "heads" not in checkpoint["config"]:
            checkpoint["config"]["heads"] = {"pusht": 4}
            state = checkpoint["state"]
            for key in ("weight", "bias"):
                state[f"heads.pusht.{key}"] = state.pop(f"mlp.4.{key}")
        return checkpoint


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
    if hasattr(cls, "upgrade"):
        checkpoint = cls.upgrade(checkpoint)
    module = cls(**checkpoint.pop("config"))
    module.load_state_dict(checkpoint.pop("state"))
    return module.to(device).eval(), checkpoint


class WorldModel:
    """Encoder + predictor + translator: what the `simulate` tool runs on.

    `encoder` is anything with `encode_states(world, states) -> [B, P, D]`; the real one is
    `jat.wm.encoder.Encoder`. `world` renders the real frames the encoder sees; `head` is the
    world the predictor was trained on, and the translator head it reads with. They differ only
    for an evaluation-only world (tabletop-new-look reads with the tabletop's model).
    """

    def __init__(
        self, world: World, encoder, predictor: Predictor, translator: Translator, head: str = ""
    ):
        self.world, self.head = world, head or world.model_world
        self.readout = readout(self.head)
        self.encoder, self.predictor, self.translator = encoder, predictor, translator
        self.device = next(predictor.parameters()).device

    @classmethod
    def load(
        cls, run: Path, device: torch.device | str = "cpu", world: str | None = None
    ) -> "WorldModel":
        """`world`: render with another world than the one trained on (evaluation only)."""
        from jat.wm.encoder import Encoder

        translator, meta = load_module(Translator, run / "translator.pt", device)
        predictor, extra = load_module(Predictor, run / "predictor.pt", device)
        encoder = Encoder(meta["latents"]["encoder"], meta["latents"]["pool"], torch.device(device))
        # Predictors trained before there were several worlds have no name: they are PushT.
        head = extra.get("world", "pusht")
        return cls(get_world(world or head), encoder, predictor, translator, head)

    @torch.no_grad()
    def encode(self, states) -> torch.Tensor:
        return self.encoder.encode_states(self.world, states).to(self.device)

    @torch.no_grad()
    def step(self, z: torch.Tensor, pushes: np.ndarray) -> torch.Tensor:
        """pushes: [B, 4] in [-1, 1] (Push.numeric())."""
        return self.predictor(z, torch.as_tensor(pushes, dtype=torch.float32, device=self.device))

    @torch.no_grad()
    def read(self, z: torch.Tensor) -> np.ndarray:
        """[B, S] world states."""
        out = self.translator(z, self.head)
        return self.readout.decode(out).cpu().numpy().astype(float)
