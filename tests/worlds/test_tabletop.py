import math

import numpy as np

from jat.worlds import get_world
from jat.worlds.base import Push
from jat.worlds.tblock import shape as geo

TABLETOP = get_world("tabletop")
CENTERED = [256.0, 200.0, 0.0]  # upright T in the middle: bar y 200..230, stem down to 320


def test_push_through_center_slides_the_t_without_turning_it():
    # Line y = 245 passes through the center of gravity (256, 245).
    result = TABLETOP.push(CENTERED, Push(start=(150.0, 245.0), end=(250.0, 245.0)))
    assert result.moved and not result.jammed
    # the pusher stops at 250 + 15 = 265; the T's stem then sits at 241 + 24 px, plus a slide
    assert 20 < result.state[0] - CENTERED[0] < 35
    assert abs(math.degrees(result.state[2])) < 5


def test_push_off_center_turns_the_t():
    result = TABLETOP.push(CENTERED, Push(start=(150.0, 215.0), end=(250.0, 215.0)))
    assert math.degrees(result.state[2]) > 10


def test_push_is_deterministic_and_markov():
    rng = np.random.default_rng(4)
    state = TABLETOP.random_start(rng)
    state = np.array(TABLETOP.push(state, TABLETOP.aimed_push(state, rng)).state)
    push = TABLETOP.aimed_push(state, rng)
    assert TABLETOP.push(state, push).state == TABLETOP.push(state, push).state


def test_push_missing_the_t_changes_nothing():
    result = TABLETOP.push(CENTERED, Push(start=(50.0, 450.0), end=(200.0, 450.0)))
    assert result.valid and not result.moved
    np.testing.assert_allclose(result.state, CENTERED, atol=0.5)


def test_t_pinned_against_the_rim_stops_the_push():
    pinned = [129.0, 256.0, math.pi / 2]
    result = TABLETOP.push(pinned, Push(start=(200.0, 256.0), end=(60.0, 256.0)))
    assert result.jammed and geo.pose_error(result.state, pinned)[0] < 8


def test_tasks_are_solvable_in_the_tabletop_physics():
    task = TABLETOP.sample_task("rotate", 0, 0)
    assert task.world == "tabletop"
    state = task.start
    for push in task.ref_plan:
        state = TABLETOP.push(state, push).state
    assert TABLETOP.check(task, state).success


def test_new_look_has_the_same_physics_and_another_picture(can_render):
    new = get_world("tabletop-new-look")
    push = Push(start=(150.0, 215.0), end=(250.0, 215.0))
    assert new.push(CENTERED, push).state == TABLETOP.push(CENTERED, push).state
    a, b = TABLETOP.render(CENTERED), new.render(CENTERED)
    assert a.shape == b.shape == (256, 256, 3) and np.abs(a.astype(int) - b).mean() > 20


def test_the_t_is_visible_where_it_stands(can_render):
    empty = TABLETOP.render([-2000.0, -2000.0, 0.0])
    for x, y in [(80, 80), (430, 80), (80, 430), (430, 430), (256, 256)]:
        frame = TABLETOP.render(geo.pose_from_center(x, y, 0.5))
        changed = np.abs(frame.astype(int) - empty).sum(-1) > 60
        assert changed.sum() > 150  # the T covers a few hundred pixels at least


def test_frames_drawn_from_many_threads_are_the_same(can_render):
    from concurrent.futures import ThreadPoolExecutor

    world = get_world("tabletop")
    pose = [256.0, 240.0, 0.4]
    with ThreadPoolExecutor(6) as pool:  # the evaluation runner's workers render like this
        frames = list(pool.map(lambda _: world.render(pose), range(12)))
    assert all(np.array_equal(f, frames[0]) for f in frames)
