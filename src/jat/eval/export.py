"""Website data: per run, the success table, the paired tests and the episode logs; per world
model, its error table in each world. JSON files the static site fetches. Episodes are copied
whole; the site loads one only when shown."""

import json
from pathlib import Path

from jat.agent.episode_log import EpisodeLog
from jat.eval.metrics import paired, summarize
from jat.eval.run import load_logs

ALL_FAMILIES = "all"
BASELINE = "no_tool"


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


def _run_entry(name: str, logs: list[EpisodeLog]) -> dict:
    worlds = {log.task.world for log in logs}
    if len(worlds) != 1:
        raise ValueError(f"run {name}: expected one world, found {sorted(worlds)}")
    every = logs + _pooled(logs)
    settings = sorted({log.setting for log in logs} - {BASELINE})
    return {
        "name": name,
        "world": worlds.pop(),
        "agent": next((log.agent.model for log in logs if log.agent), None),
        "rows": [r.model_dump() for r in summarize(every)],
        "paired": [r.model_dump() for s in settings for r in paired(every, BASELINE, s)],
    }


def _test_rows(folder: Path) -> dict[str, list]:
    """A world model's test rows per world. Models trained before there were several worlds
    store one list, for PushT. Worlds evaluated after training (`jat wm-eval`, e.g. one the
    model never saw) sit next to it as wm_eval_<world>.json."""
    test = json.loads((folder / "wm_eval.json").read_text())["test"]
    rows = test if isinstance(test, dict) else {"pusht": test}
    for path in sorted(folder.glob("wm_eval_*.json")):
        rows[path.stem.removeprefix("wm_eval_")] = json.loads(path.read_text())["test"]
    return rows


def export_web(runs: dict[str, Path], models: dict[str, Path], out: Path) -> int:
    """Writes results.json, horizon.json and episodes/<run>/ into `out`. Returns the episode count.

    `runs` maps a run name (e.g. "pusht") to a folder written by `jat eval`; `models` maps a
    display name (e.g. "v2") to a world-model run folder with wm_eval.json.
    """
    logs = {name: load_logs(folder) for name, folder in runs.items()}
    for name, run_logs in logs.items():
        if not run_logs:
            raise FileNotFoundError(f"no episode logs in {runs[name] / 'episodes'}")
    out.mkdir(parents=True, exist_ok=True)
    entries = [_run_entry(name, run_logs) for name, run_logs in logs.items()]
    (out / "results.json").write_text(json.dumps({"runs": entries}, indent=1))

    horizon = [{"name": name, "worlds": _test_rows(folder)} for name, folder in models.items()]
    (out / "horizon.json").write_text(json.dumps({"models": horizon}, indent=1))

    episodes = out / "episodes"
    # Drop episodes of an older export, but keep the folders: a running dev server stops serving
    # a folder that is deleted and recreated under it.
    for old in episodes.rglob("*.json"):
        old.unlink()
    for name, run_logs in logs.items():
        folder = episodes / name
        folder.mkdir(parents=True, exist_ok=True)
        for log in run_logs:
            (folder / f"{log.episode_id}.json").write_text(log.model_dump_json())
        index = [_index_entry(log) for log in run_logs]
        (folder / "index.json").write_text(json.dumps(index, indent=1))
    return sum(len(run_logs) for run_logs in logs.values())
