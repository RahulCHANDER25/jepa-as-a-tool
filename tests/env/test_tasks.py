import pytest

from jat.env import geometry as geo
from jat.env.pusht import World
from jat.env.tasks import FAMILIES, check, sample_task


@pytest.mark.parametrize("family", list(FAMILIES))
def test_sampled_task_respects_family_and_ref_plan_reaches_goal(family):
    spec = FAMILIES[family]
    task = sample_task(family, seed=0, index=0)
    shift, turn = geo.pose_error(task.goal, task.start)
    assert spec.min_shift <= shift <= spec.max_shift
    assert spec.min_turn <= turn <= spec.max_turn
    world = World(task.start)
    for push in task.ref_plan:
        world.push(push)
    assert check(task, world.pose()).success


def test_task_sampling_is_reproducible():
    assert sample_task("rotate", seed=5, index=2) == sample_task("rotate", seed=5, index=2)


def test_check_fails_at_the_start_pose():
    task = sample_task("pose", seed=0, index=1)
    outcome = check(task, task.start)
    assert not outcome.success and outcome.coverage < 1.0
