import math

from jat.env.tasks import sample_task
from jat.eval.metrics import mcnemar_p, paired, summarize, wilson_interval
from jat.eval.run import run_random


def test_wilson_interval_contains_the_rate():
    low, high = wilson_interval(15, 30)
    assert low < 0.5 < high


def test_wilson_interval_stays_inside_zero_one_at_the_extremes():
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    assert math.isclose(wilson_interval(0, 10)[1], 0.2775, abs_tol=1e-3)


def test_summarize_groups_by_family_and_setting():
    logs = [run_random(sample_task(f, 0, i), seed=0) for f in ("rotate", "pose") for i in range(2)]
    rows = summarize(logs)
    assert [(r.family, r.setting, r.n) for r in rows] == [
        ("pose", "random", 2),
        ("rotate", "random", 2),
    ]
    assert rows[1].real_pushes == 3.0


def test_mcnemar_p_is_exact():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 5) == 1.0
    assert math.isclose(mcnemar_p(0, 5), 2 / 32)
    assert math.isclose(mcnemar_p(17, 3), mcnemar_p(3, 17))


def test_paired_counts_only_tasks_both_settings_ran():
    def log(index: int, setting: str, success: bool):
        base = run_random(sample_task("rotate", 0, index), seed=0)
        return base.model_copy(
            update={
                "setting": setting,
                "episode_id": f"{base.task.id}__{setting}__s0",
                "outcome": base.outcome.model_copy(update={"success": success}),
            }
        )

    logs = [
        log(0, "no_tool", False), log(0, "wm_tool", True),
        log(1, "no_tool", True), log(1, "wm_tool", True),
        log(2, "no_tool", True), log(2, "wm_tool", False),
        log(3, "wm_tool", True),  # no baseline episode: left out
    ]  # fmt: skip
    [row] = paired(logs, "no_tool", "wm_tool")
    assert (row.family, row.n, row.only_setting, row.only_baseline) == ("rotate", 3, 1, 1)
    assert row.p_value == 1.0
