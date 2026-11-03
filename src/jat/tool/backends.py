"""What `simulate` runs on. Two backends share one interface, so the agent code never changes:
the oracle (the world's real physics) and the V-JEPA 2 world model.
"""

from typing import Any, Protocol

import numpy as np
import torch

from jat.wm.models import WorldModel
from jat.worlds.base import Push, World


class Backend(Protocol):
    name: str

    def observe(self, state) -> Any:
        """The backend's state for the real table in this true state."""

    def step(self, states: list, pushes: list[Push]) -> list:
        """One imagined push per state, batched."""

    def read(self, states: list) -> list[np.ndarray]:
        """The world state the LLM is told for each backend state."""


class OracleBackend:
    """Perfect imagination: a state is the true state, and stepping runs the real physics.
    `read` returns the true state: the upper bound, with perfect perception too."""

    name = "oracle"

    def __init__(self, world: World):
        self.world = world

    def observe(self, state) -> np.ndarray:
        return np.asarray(state)

    def step(self, states: list, pushes: list[Push]) -> list[np.ndarray]:
        return [
            np.asarray(self.world.push(s, p).state) for s, p in zip(states, pushes, strict=True)
        ]

    def read(self, states: list) -> list[np.ndarray]:
        return [np.asarray(s) for s in states]


class WorldModelBackend:
    """Imagination in V-JEPA 2 latent space. A state is a latent [P, D]. Every state the LLM
    reads, real or imagined, is the translator's reading of a latent."""

    name = "world_model"

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def observe(self, state) -> torch.Tensor:
        return self.world_model.encode([state])[0]

    def step(self, states: list, pushes: list[Push]) -> list[torch.Tensor]:
        actions = np.stack([p.numeric() for p in pushes])
        return list(self.world_model.step(torch.stack(states), actions))

    def read(self, states: list) -> list[np.ndarray]:
        return list(self.world_model.read(torch.stack(states)))
