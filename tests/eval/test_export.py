import json

from jat.env.tasks import sample_task
from jat.eval.export import export_web
from jat.eval.run import run_grid


def test_export_writes_tables_and_every_episode(tmp_path):
    run, model, out = tmp_path / "run", tmp_path / "wm", tmp_path / "web"
    tasks = [sample_task("rotate", 0, i) for i in range(2)]
    run_grid(tasks, ["random"], [0], None, run, workers=1)
    model.mkdir()
    (model / "wm_eval.json").write_text(json.dumps({"test": [{"horizon": 0}], "history": []}))

    assert export_web(run, {"v2": model}, out) == 2
    rows = json.loads((out / "results.json").read_text())["rows"]
    assert {r["family"] for r in rows} == {"rotate", "all"}
    index = json.loads((out / "episodes" / "index.json").read_text())
    assert all((out / "episodes" / f"{e['id']}.json").exists() for e in index)
    assert json.loads((out / "horizon.json").read_text())["models"][0]["name"] == "v2"


def test_export_replaces_episodes_of_an_older_run(tmp_path):
    run, out = tmp_path / "run", tmp_path / "web"
    run_grid([sample_task("rotate", 0, 0)], ["random"], [0], None, run, workers=1)
    (out / "episodes").mkdir(parents=True)
    (out / "episodes" / "stale.json").write_text("{}")
    folder = (out / "episodes").stat().st_ino
    export_web(run, {}, out)
    assert not (out / "episodes" / "stale.json").exists()
    assert (out / "episodes").stat().st_ino == folder  # same folder, not recreated
