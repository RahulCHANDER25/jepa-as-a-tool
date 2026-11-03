"""What every world shares: the table, the push action, tasks, and the World protocol.

A world's state is a flat float vector; in the T worlds it is the T's pose (x, y, theta).
Datasets and episode logs store states under the name `pose`, from when PushT was the only world.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

TABLE = 512  # table side in px; 1 px = 1 mm in the 3D worlds
PUSHER_RADIUS = 15.0
MARGIN = 20.0  # push points stay this far from the table edge
MIN_LENGTH, MAX_LENGTH = 10.0, 200.0


class Push(BaseModel):
    """One straight push of the pusher, in table pixels."""

    model_config = ConfigDict(frozen=True)

    # Lists, not tuples: tuple JSON schemas use `prefixItems`, which some tool-calling
    # endpoints reject.
    start: list[float] = Field(
        min_length=2, max_length=2, description="[x, y] where the pusher is placed"
    )
    end: list[float] = Field(
        min_length=2, max_length=2, description="[x, y] where the pusher stops"
    )

    def length(self) -> float:
        return math.dist(self.start, self.end)

    def numeric(self) -> np.ndarray:
        """[x1, y1, x2, y2] scaled to [-1, 1]: the world model's action input."""
        return np.array([*self.start, *self.end]) / (TABLE / 2) - 1

    def text(self) -> str:
        (x1, y1), (x2, y2) = self.start, self.end
        return f"({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})"


class PushResult(BaseModel):
    state: list[float]
    valid: bool
    moved: bool = False
    jammed: bool = False
    error: str | None = None
    # One row per 0.1 s step: pusher x, pusher y, then the state. For replays only.
    trajectory: list[tuple[float, ...]] = []


class Task(BaseModel):
    id: str
    family: str
    seed: int
    start: list[float]
    goal: list[float]
    max_pushes: int
    ref_plan: list[Push]
    world: str = "pusht"  # logs written before there were several worlds have no field

    @property
    def text(self) -> str:
        from jat.worlds import get_world

        return get_world(self.world).goal_text(self.goal)


@dataclass(frozen=True)
class ToolText:
    """The world's words in the tool's replies."""

    ends_up: str  # "Predicts {ends_up} after each push"
    misses: str  # an imagined push that moves nothing
    missed: str  # a real push that moved nothing
    at_goal: str  # "Call it when {at_goal}."


class Outcome(BaseModel):
    success: bool
    position_error: float  # px
    angle_error: float  # degrees
    coverage: float | None = None  # T worlds: share of the goal T covered


def check_bounds(push: Push) -> str | None:
    """The push rules every world shares: inside the table, length in range."""
    points = np.array([push.start, push.end])
    if not ((points >= MARGIN) & (points <= TABLE - MARGIN)).all():
        return f"start and end must lie inside [{MARGIN:.0f}, {TABLE - MARGIN:.0f}] on both axes"
    length = push.length()
    if not MIN_LENGTH <= length <= MAX_LENGTH:
        return f"push length {length:.0f} px is outside [{MIN_LENGTH:.0f}, {MAX_LENGTH:.0f}]"
    return None


class World(Protocol):
    """A table the agent pushes things on. Stateless: every call gets the state it acts on, and
    a push's result depends on (state, push) only."""

    name: str
    families: Mapping[str, object]  # task families, by name
    tool_text: ToolText
    readout: str  # what a translator head reads: "t_pose" or "points:<n>" (jat.wm.readouts)
    model_world: str  # whose world model this world uses: its own name, unless evaluation only

    def push(self, state, push: Push) -> PushResult:
        """Run one push and wait until everything is at rest."""

    def check_push(self, state, push: Push) -> str | None:
        """Why this push cannot run from this state, or None if it can."""

    def moved(self, before, after) -> bool: ...

    def render(self, state, size: int = 256) -> np.ndarray:
        """uint8 [size, size, 3]: the frame V-JEPA 2 sees. No pusher, no goal."""

    def random_start(self, rng: np.random.Generator) -> np.ndarray: ...

    def aimed_push(self, state, rng: np.random.Generator) -> Push:
        """A random push that hits the object. Raises RuntimeError if none fits."""

    def describe(self) -> str:
        """The world part of the agent's system prompt: table, object, physics, action."""

    def rules(self, task: Task) -> str:
        """The goal and success part of the agent's system prompt."""

    def state_text(self, state, goal=None) -> str:
        """A state as the agent reads it, with the error to `goal` if given."""

    def goal_text(self, goal) -> str: ...

    def outlines(self, state, goal: bool = False) -> list[tuple[tuple[int, int, int], np.ndarray]]:
        """(RGB color, top-view outline [N, 2] in table px), one per object, for sketches.
        `goal`: only the objects the goal is about."""

    def sample_task(self, family: str, seed: int, index: int) -> Task: ...

    def check(self, task: Task, state) -> Outcome: ...


def uniform_push(world: World, state, rng: np.random.Generator) -> Push:
    """Any valid push on the table; most of them miss the object."""
    while True:
        start = rng.uniform(MARGIN, TABLE - MARGIN, size=2)
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(MIN_LENGTH, MAX_LENGTH)
        end = start + length * np.array([math.cos(angle), math.sin(angle)])
        push = Push(start=tuple(start), end=tuple(end))
        if world.check_push(state, push) is None:
            return push
