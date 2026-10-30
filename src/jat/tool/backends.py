"""What `simulate` runs on. Two backends share one interface, so the agent code never changes:
the oracle (real PushT physics) and the V-JEPA 2 world model.
"""

from typing import Any, Protocol

import numpy as np
import torch

from jat.env.pusht import Push, World
from jat.wm.models import WorldModel


class Backend(Protocol):
    name: str

    def observe(self, world: World) -> Any:
        """The backend's state for the real table right now."""

    def step(self, states: list, pushes: list[Push]) -> list:
        """One imagined push per state, batched."""

    def read(self, states: list) -> list[np.ndarray]:
        """The pose the LLM is told for each state."""


class OracleBackend:
    """Perfect imagination: a state is a pose, and stepping runs the real physics on a copy.
    `read` returns the true pose: the upper bound, with perfect perception too."""

    name = "oracle"

    def observe(self, world: World) -> np.ndarray:
        return world.pose()

    def step(self, states: list, pushes: list[Push]) -> list[np.ndarray]:
        return [np.asarray(World(s).push(p).pose) for s, p in zip(states, pushes, strict=True)]

    def read(self, states: list) -> list[np.ndarray]:
        return [np.asarray(s) for s in states]


class WorldModelBackend:
    """Imagination in V-JEPA 2 latent space. A state is a latent [P, D]. Every pose the LLM
    reads, real or imagined, is the translator's reading of a latent."""

    name = "world_model"

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def observe(self, world: World) -> torch.Tensor:
        return self.world_model.encode([world.pose()])[0]

    def step(self, states: list, pushes: list[Push]) -> list[torch.Tensor]:
        actions = np.stack([p.numeric() for p in pushes])
        return list(self.world_model.step(torch.stack(states), actions))

    def read(self, states: list) -> list[np.ndarray]:
        return list(self.world_model.read(torch.stack(states)))
