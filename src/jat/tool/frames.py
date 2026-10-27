"""Frame handles: short ids the LLM uses to say where to imagine from.

Real frames are R0, R1, ... (one per real push). Imagined frames are I1, I2, ... (one per
imagined push). Each frame keeps the backend's state (a pose for the oracle, a latent for the
world model), so imagining from a frame never re-simulates its prefix.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from jat.env.pusht import Push


class UnknownFrame(KeyError):
    pass


@dataclass
class Frame:
    id: str
    kind: Literal["real", "imagined"]
    pose: np.ndarray  # what the LLM is told: the backend's reading of `state`
    state: Any
    parent: str | None = None
    push: Push | None = None
    true_pose: np.ndarray | None = None  # real frames only
    trajectory: list = field(default_factory=list)  # real frames only, for replays


class FrameStore:
    def __init__(self):
        self.frames: dict[str, Frame] = {}
        self._real = 0
        self._imagined = 0
        self.current: str | None = None  # latest real frame

    def add_real(self, state, pose, true_pose, parent=None, push=None, trajectory=()) -> str:
        frame_id = f"R{self._real}"
        self._real += 1
        self.frames[frame_id] = Frame(
            frame_id,
            "real",
            np.asarray(pose),
            state,
            parent,
            push,
            np.asarray(true_pose),
            list(trajectory),
        )
        self.current = frame_id
        return frame_id

    def add_imagined(self, parent: str, push: Push, state, pose) -> str:
        self.get(parent)
        self._imagined += 1
        frame_id = f"I{self._imagined}"
        self.frames[frame_id] = Frame(frame_id, "imagined", np.asarray(pose), state, parent, push)
        return frame_id

    def get(self, frame_id: str) -> Frame:
        try:
            return self.frames[frame_id]
        except KeyError:
            raise UnknownFrame(frame_id) from None
