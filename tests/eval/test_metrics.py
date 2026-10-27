import math

from jat.env.tasks import sample_task
from jat.eval.metrics import summarize, wilson_interval
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
