import math

import numpy as np
import pytest

from jat.worlds import get_world
from jat.worlds.base import Push
from jat.worlds.pusht.world import END_TOLERANCE, JAM_DEPTH, STEP_PX, PushT, _penetration, _Table
from jat.worlds.tblock import shape as geo

CENTERED = [256.0, 200.0, 0.0]  # upright T in the middle: bar y 200..230, stem down to 320
PUSHT = get_world("pusht")


@pytest.mark.parametrize("theta", [0.0, 0.7, -2.5, math.pi])
def test_set_pose_round_trips(theta):
    # gym-pusht's own reset_to_state shifts the T by up to 45 px; ours must not.
    pose = [300.0, 250.0, theta]
    table = _Table()
    table.set_pose(pose)
    np.testing.assert_allclose(table.pose()[:2], pose[:2], atol=1e-9)


def test_push_is_deterministic_given_pose_and_push():
    push = Push(start=(150.0, 260.0), end=(260.0, 260.0))
    assert PUSHT.push(CENTERED, push).state == PUSHT.push(CENTERED, push).state


def test_push_depends_on_state_and_push_only():
    # A simulator that already ran other pushes and a brand-new one give the same result.
    rng = np.random.default_rng(3)
    state = np.array(PUSHT.random_start(rng))
    state = np.array(PUSHT.push(state, PUSHT.aimed_push(state, rng)).state)
    push = PUSHT.aimed_push(state, rng)
    np.testing.assert_allclose(
        PUSHT.push(state, push).state, PushT().push(state, push).state, atol=1e-9
    )


def test_push_starting_inside_t_is_invalid_and_pose_unchanged():
    result = PUSHT.push(CENTERED, Push(start=(256.0, 215.0), end=(256.0, 150.0)))
    assert not result.valid and "overlap" in result.error
    np.testing.assert_allclose(result.state, CENTERED)


def test_push_leaving_the_table_is_invalid():
    result = PUSHT.push(CENTERED, Push(start=(100.0, 100.0), end=(5.0, 100.0)))
    assert not result.valid and "inside" in result.error


def test_push_missing_t_leaves_pose_unchanged():
    result = PUSHT.push(CENTERED, Push(start=(50.0, 450.0), end=(200.0, 450.0)))
    assert result.valid and not result.moved
    np.testing.assert_allclose(result.state, CENTERED, atol=1e-9)


def test_push_through_center_translates_and_pusher_reaches_end():
    # Line y = 245 passes through the center of gravity (256, 245): no rotation expected.
    push = Push(start=(150.0, 245.0), end=(250.0, 245.0))
    result = PUSHT.push(CENTERED, push)
    assert result.moved and not result.jammed
    assert result.state[0] > CENTERED[0] + 20
    assert abs(math.degrees(result.state[2])) < 1.0
    assert math.dist(result.trajectory[-1][:2], push.end) < END_TOLERANCE


def test_push_stops_when_t_is_pinned_against_a_wall():
    # Stem points left and touches the left wall; push the bar straight toward the wall.
    pinned = [129.0, 256.0, math.pi / 2]
    result = PUSHT.push(pinned, Push(start=(200.0, 256.0), end=(60.0, 256.0)))
    assert result.jammed
    last = result.trajectory[-1]
    assert _penetration(last[2:], last[:2]) < JAM_DEPTH + STEP_PX
    assert geo.pose_error(result.state, pinned)[0] < 5.0


def test_random_aimed_pushes_almost_always_move_the_t():
    rng = np.random.default_rng(0)
    results = []
    for _ in range(30):
        pose = PUSHT.random_start(rng)
        results.append(PUSHT.push(pose, PUSHT.aimed_push(pose, rng)))
    assert all(r.valid for r in results)
    assert sum(r.moved for r in results) >= 27


def test_unknown_world_is_refused():
    with pytest.raises(ValueError, match="unknown world"):
        get_world("moon")
