"""Episode logs: one JSON file per episode. The report and the website read these."""

import numpy as np
from pydantic import BaseModel

from jat.tool.frames import FrameStore
from jat.worlds.base import Outcome, Push, Task, World

SCHEMA_VERSION = 1


class FrameLog(BaseModel):
    id: str
    kind: str  # "real" | "imagined"
    pose: list[float]  # the world state the LLM was told
    # Real frame: the table. Imagined frame: the real physics replayed along the same branch.
    true_pose: list[float]
    parent: str | None = None
    push: Push | None = None
    # Real frames only: pusher x, y, then the world state, per 0.1 s, for replays.
    trajectory: list[list[float]] = []


class ToolCallLog(BaseModel):
    name: str
    args: dict
    result: str


class TurnLog(BaseModel):
    content: str
    reasoning: str | None = None
    tool_calls: list[ToolCallLog] = []
    input_tokens: int | None = None
    output_tokens: int | None = None


class AgentInfo(BaseModel):
    provider: str
    model: str
    prompt_version: str
    temperature: float


class EpisodeLog(BaseModel):
    schema_version: int = SCHEMA_VERSION
    episode_id: str
    setting: str
    backend: str
    agent: AgentInfo | None = None  # None for the random baseline
    task: Task
    frames: list[FrameLog]
    turns: list[TurnLog] = []
    outcome: Outcome
    final_pose: list[float]
    stop_reason: str
    wall_time_s: float


def frame_logs(store: FrameStore, world: World) -> list[FrameLog]:
    """All frames in creation order, with the true pose filled in for imagined ones."""
    true: dict[str, np.ndarray] = {}
    logs = []
    for frame in store.frames.values():  # creation order: a parent always comes first
        if frame.kind == "real":
            true[frame.id] = frame.true_pose
        else:
            true[frame.id] = np.asarray(world.push(true[frame.parent], frame.push).state)
        logs.append(
            FrameLog(
                id=frame.id,
                kind=frame.kind,
                pose=np.asarray(frame.pose, dtype=float).tolist(),
                true_pose=true[frame.id].tolist(),
                parent=frame.parent,
                push=frame.push,
                trajectory=[list(row) for row in frame.trajectory],
            )
        )
    return logs
