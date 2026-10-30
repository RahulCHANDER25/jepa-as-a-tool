import numpy as np
import pytest

from jat.worlds import get_world
from jat.worlds.tblock import shape as geo
from jat.worlds.tblock.world import FAMILIES

PUSHT = get_world("pusht")


@pytest.mark.parametrize("family", list(FAMILIES))
def test_sampled_task_respects_family_and_ref_plan_reaches_goal(family):
    spec = FAMILIES[family]
    task = PUSHT.sample_task(family, seed=0, index=0)
    assert task.world == "pusht"
    shift, turn = geo.pose_error(task.goal, task.start)
    assert spec.min_shift <= shift <= spec.max_shift
    assert spec.min_turn <= turn <= spec.max_turn
    state = task.start
    for push in task.ref_plan:
        state = np.array(PUSHT.push(state, push).state)
    assert PUSHT.check(task, state).success


def test_task_sampling_is_reproducible():
    assert PUSHT.sample_task("rotate", 5, 2) == PUSHT.sample_task("rotate", 5, 2)


def test_check_fails_at_the_start_pose():
    task = PUSHT.sample_task("pose", seed=0, index=1)
    outcome = PUSHT.check(task, task.start)
    assert not outcome.success and outcome.coverage < 1.0
