import math

import numpy as np
from gym_pusht.envs.pusht import PushTEnv

from jat.worlds.pusht.world import _Table
from jat.worlds.tblock import shape as geo


def test_keypoints_match_gym_pusht():
    table = _Table()
    table.set_pose([300.0, 250.0, 0.7])
    ours = geo.keypoints(table.pose())
    theirs = PushTEnv.get_keypoints(table._env._block_shapes)
    np.testing.assert_allclose(ours, theirs, atol=1e-6)


def test_pose_from_center_inverts_center():
    pose = geo.pose_from_center(200.0, 310.0, -2.1)
    np.testing.assert_allclose(geo.center(pose), [200.0, 310.0], atol=1e-9)


def test_coverage_is_one_on_itself_and_zero_when_apart():
    pose = [256.0, 200.0, 0.3]
    assert math.isclose(geo.coverage(pose, pose), 1.0)
    assert geo.coverage(pose, [100.0, 50.0, 0.3]) == 0.0


def test_angle_error_wraps_around():
    _, angle = geo.pose_error([256, 256, math.radians(359)], [256, 256, math.radians(1)])
    assert math.isclose(angle, 2.0, abs_tol=1e-6)


def test_stem_points_down_at_angle_zero_and_left_at_90_degrees():
    assert geo.stem_direction([256, 256, 0.0]) == "down"
    assert geo.stem_direction([256, 256, math.pi / 2]) == "left"
