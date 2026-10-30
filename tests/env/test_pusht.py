import math

import numpy as np
import pytest

from jat.env import geometry as geo
from jat.env.pusht import (
    END_TOLERANCE,
    JAM_DEPTH,
    STEP_PX,
    Push,
    World,
    _penetration,
    random_aimed_push,
    random_start_pose,
)

CENTERED = [256.0, 200.0, 0.0]  # upright T in the middle: bar y 200..230, stem down to 320


@pytest.mark.parametrize("theta", [0.0, 0.7, -2.5, math.pi])
def test_set_pose_round_trips(theta):
    # gym-pusht's own reset_to_state shifts the T by up to 45 px; World must not.
    pose = [300.0, 250.0, theta]
    np.testing.assert_allclose(World(pose).pose()[:2], pose[:2], atol=1e-9)


def test_push_is_deterministic_given_pose_and_push():
    push = Push(start=(150.0, 260.0), end=(260.0, 260.0))
    a, b = World(CENTERED).push(push), World(CENTERED).push(push)
    assert a.pose == b.pose


def test_branch_matches_original_after_same_pushes():
    rng = np.random.default_rng(3)
    world = World(random_start_pose(rng))
    world.push(random_aimed_push(world.pose(), rng))
    copy = world.branch()
    push = random_aimed_push(world.pose(), rng)
    np.testing.assert_allclose(world.push(push).pose, copy.push(push).pose, atol=1e-9)


def test_push_starting_inside_t_is_invalid_and_pose_unchanged():
    world = World(CENTERED)
    result = world.push(Push(start=(256.0, 215.0), end=(256.0, 150.0)))
    assert not result.valid and "overlap" in result.error
    np.testing.assert_allclose(world.pose(), CENTERED)


def test_push_leaving_the_table_is_invalid():
    result = World(CENTERED).push(Push(start=(100.0, 100.0), end=(5.0, 100.0)))
    assert not result.valid and "inside" in result.error


def test_push_missing_t_leaves_pose_unchanged():
    result = World(CENTERED).push(Push(start=(50.0, 450.0), end=(200.0, 450.0)))
    assert result.valid and not result.moved
    np.testing.assert_allclose(result.pose, CENTERED, atol=1e-9)


def test_push_through_center_translates_and_pusher_reaches_end():
    # Line y = 245 passes through the center of gravity (256, 245): no rotation expected.
    push = Push(start=(150.0, 245.0), end=(250.0, 245.0))
    result = World(CENTERED).push(push)
    assert result.moved and not result.jammed
    assert result.pose[0] > CENTERED[0] + 20
    assert abs(math.degrees(result.pose[2])) < 1.0
    assert math.dist(result.trajectory[-1][:2], push.end) < END_TOLERANCE


def test_push_stops_when_t_is_pinned_against_a_wall():
    # Stem points left and touches the left wall; push the bar straight toward the wall.
    pinned = [129.0, 256.0, math.pi / 2]
    result = World(pinned).push(Push(start=(200.0, 256.0), end=(60.0, 256.0)))
    assert result.jammed
    last = result.trajectory[-1]
    assert _penetration(last[2:], last[:2]) < JAM_DEPTH + STEP_PX
    assert geo.pose_error(result.pose, pinned)[0] < 5.0


def test_random_aimed_pushes_almost_always_move_the_t():
    rng = np.random.default_rng(0)
    results = []
    for _ in range(30):
        pose = random_start_pose(rng)
        results.append(World(pose).push(random_aimed_push(pose, rng)))
    assert all(r.valid for r in results)
    assert sum(r.moved for r in results) >= 27
