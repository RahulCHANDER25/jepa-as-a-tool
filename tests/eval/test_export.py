import json

import pytest

from jat.eval.export import export_web
from jat.eval.run import run_grid
from jat.worlds import get_world


def _run(folder, world="pusht", n=2):
    sample = get_world(world).sample_task
    tasks = [sample(next(iter(get_world(world).families)), 0, i) for i in range(n)]
    run_grid(tasks, ["random"], [0], None, folder, workers=1)
    return folder


def test_export_writes_tables_and_every_episode_per_run(tmp_path):
    out = tmp_path / "web"
    runs = {
        "pusht": _run(tmp_path / "pusht"),
        "billiards": _run(tmp_path / "billiards", "billiards"),
    }
    old, new = tmp_path / "v1", tmp_path / "v2"
    old.mkdir()
    (old / "wm_eval.json").write_text(json.dumps({"test": [{"horizon": 0}]}))
    new.mkdir()
    (new / "wm_eval.json").write_text(json.dumps({"test": {"billiards": [{"horizon": 1}]}}))
    (new / "wm_eval_tabletop-new-look.json").write_text(json.dumps({"test": [{"horizon": 2}]}))

    assert export_web(runs, {"v1": old, "v2": new}, out) == 4
    entries = {r["name"]: r for r in json.loads((out / "results.json").read_text())["runs"]}
    assert entries["billiards"]["world"] == "billiards"
    assert {r["family"] for r in entries["pusht"]["rows"]} >= {"all"}
    for name in runs:
        index = json.loads((out / "episodes" / name / "index.json").read_text())
        assert all((out / "episodes" / name / f"{e['id']}.json").exists() for e in index)
    models = json.loads((out / "horizon.json").read_text())["models"]
    # a single-world table is PushT's; a world evaluated later has its own file
    assert [m["worlds"] for m in models] == [
        {"pusht": [{"horizon": 0}]},
        {"billiards": [{"horizon": 1}], "tabletop-new-look": [{"horizon": 2}]},
    ]


def test_export_replaces_episodes_of_an_older_export(tmp_path):
    out = tmp_path / "web"
    (out / "episodes" / "pusht").mkdir(parents=True)
    (out / "episodes" / "pusht" / "stale.json").write_text("{}")
    folder = (out / "episodes").stat().st_ino
    export_web({"pusht": _run(tmp_path / "run", n=1)}, {}, out)
    assert not (out / "episodes" / "pusht" / "stale.json").exists()
    assert (out / "episodes").stat().st_ino == folder  # same folder, not recreated


def test_export_refuses_a_run_without_episodes(tmp_path):
    (tmp_path / "empty" / "episodes").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        export_web({"pusht": tmp_path / "empty"}, {}, tmp_path / "web")
