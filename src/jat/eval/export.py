"""Website data: the success table, the world model's error table and the episode logs, as JSON
files the static site fetches. Episodes are copied whole; the site loads one only when shown."""

import json
import shutil
from pathlib import Path

from jat.agent.episode_log import EpisodeLog
from jat.eval.metrics import summarize
from jat.eval.run import load_logs

ALL_FAMILIES = "all"


def _pooled(logs: list[EpisodeLog]) -> list[EpisodeLog]:
    """The same logs, relabeled as one family, so `summarize` also gives the overall rows."""
    return [
        log.model_copy(update={"task": log.task.model_copy(update={"family": ALL_FAMILIES})})
        for log in logs
    ]


def _index_entry(log: EpisodeLog) -> dict:
    return {
        "id": log.episode_id,
        "task": log.task.id,
        "family": log.task.family,
        "setting": log.setting,
        "success": log.outcome.success,
        "position_error": round(log.outcome.position_error, 1),
        "angle_error": round(log.outcome.angle_error, 1),
        "real_pushes": sum(f.kind == "real" for f in log.frames) - 1,
        "imagined_pushes": sum(f.kind == "imagined" for f in log.frames),
    }


def export_web(run: Path, models: dict[str, Path], out: Path) -> int:
    """Writes results.json, horizon.json and episodes/ into `out`. Returns the episode count.

    `models` maps a display name (e.g. "v2") to a world-model run folder with wm_eval.json.
    """
    logs = load_logs(run)
    if not logs:
        raise FileNotFoundError(f"no episode logs in {run / 'episodes'}")
    rows = summarize(logs) + summarize(_pooled(logs))
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps({"run": run.name, "rows": [r.model_dump() for r in rows]}, indent=1)
    )

    horizon = [
        {"name": name, "rows": json.loads((folder / "wm_eval.json").read_text())["test"]}
        for name, folder in models.items()
    ]
    (out / "horizon.json").write_text(json.dumps({"models": horizon}, indent=1))

    episodes = out / "episodes"
    shutil.rmtree(episodes, ignore_errors=True)  # generated: drop episodes of an older run
    episodes.mkdir()
    for log in logs:
        (episodes / f"{log.episode_id}.json").write_text(log.model_dump_json())
    index = [_index_entry(log) for log in logs]
    (episodes / "index.json").write_text(json.dumps(index, indent=1))
    return len(logs)
