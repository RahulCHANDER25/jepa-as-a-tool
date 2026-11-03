"""Run a grid of (task x setting x seed) episodes and save one JSON log per episode.

Episodes already on disk are skipped, so an interrupted run resumes where it stopped.
"""

import time
import traceback
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from jat.agent.episode_log import AgentInfo, EpisodeLog, frame_logs
from jat.agent.loop import run_episode
from jat.tool.backends import OracleBackend, WorldModelBackend
from jat.tool.session import Session
from jat.wm.models import WorldModel
from jat.worlds import get_world
from jat.worlds.base import Task

# A setting may carry a tag, "wm_tool@ft": the world model registered under that tag.
SETTINGS = ("no_tool", "wm_tool", "oracle_tool", "random")
ModelFactory = Callable[[], tuple[object, AgentInfo]]


def episode_id(task: Task, setting: str, seed: int) -> str:
    return f"{task.id}__{setting}__s{seed}"


def run_random(task: Task, seed: int) -> EpisodeLog:
    """Baseline without an LLM: random pushes aimed at the T."""
    started = time.time()
    rng = np.random.default_rng(
        [seed, zlib.crc32(task.id.encode())]
    )  # hash() is salted per process
    world = get_world(task.world)
    session = Session(task, OracleBackend(world), simulate_enabled=False)
    while not session.finished:
        session.act(world.aimed_push(session.state, rng))
    final_pose = session.state
    return EpisodeLog(
        episode_id=episode_id(task, "random", seed),
        setting="random",
        backend="oracle",
        task=task,
        frames=frame_logs(session.frames, world),
        outcome=world.check(task, final_pose),
        final_pose=final_pose.tolist(),
        stop_reason=session.stop_reason,
        wall_time_s=time.time() - started,
    )


def run_one(
    task: Task,
    setting: str,
    seed: int,
    model_factory: ModelFactory | None,
    world_models: dict[str, WorldModel] | None = None,
) -> EpisodeLog:
    if setting == "random":
        return run_random(task, seed)
    base, _, tag = setting.partition("@")
    if base not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}; expected one of {SETTINGS}")
    if base == "wm_tool":
        world_model = (world_models or {}).get(tag)
        if world_model is None:
            raise ValueError(f"{setting} needs a trained world model")
        backend = WorldModelBackend(world_model)
    else:
        # no_tool also uses the oracle backend: only to read the current frame, never to imagine.
        backend = OracleBackend(get_world(task.world))
    model, agent = model_factory()
    return run_episode(task, setting, backend, model, agent, episode_id(task, setting, seed))


def run_grid(
    tasks: list[Task],
    settings: list[str],
    seeds: list[int],
    model_factory: ModelFactory | None,
    out_dir: Path,
    workers: int = 8,
    world_models: dict[str, WorldModel] | None = None,
) -> tuple[list[EpisodeLog], int]:
    """Returns (all logs on disk for this grid, number of failed episodes). World models are
    shared by all worker threads: inference only, no state between calls."""
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(t, s, seed) for t in tasks for s in settings for seed in seeds]
    todo = [j for j in jobs if not (episodes_dir / f"{episode_id(*j)}.json").exists()]
    print(f"{len(jobs)} episodes, {len(jobs) - len(todo)} already done, {len(todo)} to run")

    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, *job, model_factory, world_models): job for job in todo}
        for i, future in enumerate(as_completed(futures), 1):
            job_id = episode_id(*futures[future])
            # One failed episode (API error, timeout) must not kill a long run. It has no log
            # file, so the next run retries it.
            try:
                log = future.result()
            except Exception:
                failures += 1
                print(f"[{i}/{len(todo)}] FAILED {job_id}\n{traceback.format_exc(limit=3)}")
                continue
            (episodes_dir / f"{log.episode_id}.json").write_text(log.model_dump_json(indent=1))
            mark = "ok " if log.outcome.success else "   "
            print(
                f"[{i}/{len(todo)}] {mark} {job_id}  {log.outcome.position_error:.0f} px "
                f"{log.outcome.angle_error:.0f}°  ({log.stop_reason}, {log.wall_time_s:.0f}s)"
            )

    logs = [
        EpisodeLog.model_validate_json((episodes_dir / f"{episode_id(*j)}.json").read_text())
        for j in jobs
        if (episodes_dir / f"{episode_id(*j)}.json").exists()
    ]
    return logs, failures


def load_logs(out_dir: Path) -> list[EpisodeLog]:
    return [
        EpisodeLog.model_validate_json(p.read_text())
        for p in sorted((out_dir / "episodes").glob("*.json"))
    ]
