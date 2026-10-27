"""What `simulate` runs on. Two backends share one interface, so the agent code never changes:
the oracle (real PushT physics) now, the V-JEPA 2 world model from M4.
"""

from typing import Any, Protocol

import numpy as np

from jat.env.pusht import Push, World


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

    `read` returns the true pose. From M4 on, the LLM-facing reading goes through
    render -> V-JEPA 2 -> translator instead, so oracle and world model differ only in the
    predictor.
    """

    name = "oracle"

    def observe(self, world: World) -> np.ndarray:
        return world.pose()

    def step(self, states: list, pushes: list[Push]) -> list[np.ndarray]:
        return [np.asarray(World(s).push(p).pose) for s, p in zip(states, pushes, strict=True)]

    def read(self, states: list) -> list[np.ndarray]:
        return [np.asarray(s) for s in states]
