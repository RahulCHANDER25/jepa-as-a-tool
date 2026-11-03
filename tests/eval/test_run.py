from jat.eval.run import run_grid, run_random
from jat.worlds import get_world

sample_task = get_world("pusht").sample_task


def test_random_baseline_is_reproducible():
    task = sample_task("rotate", 0, 0)
    assert run_random(task, seed=1).final_pose == run_random(task, seed=1).final_pose


def test_grid_writes_logs_and_skips_them_on_rerun(tmp_path, capsys):
    tasks = [sample_task("rotate", 0, i) for i in range(2)]
    logs, failures = run_grid(tasks, ["random"], [0], None, tmp_path, workers=2)
    assert failures == 0 and len(logs) == 2
    assert len(list((tmp_path / "episodes").glob("*.json"))) == 2

    run_grid(tasks, ["random"], [0], None, tmp_path, workers=2)
    assert "2 already done, 0 to run" in capsys.readouterr().out
