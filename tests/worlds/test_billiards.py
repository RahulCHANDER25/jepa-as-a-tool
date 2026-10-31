import numpy as np
import pytest

from jat.worlds import get_world
from jat.worlds.base import Push

BILLIARDS = get_world("billiards")
APART = [150.0, 256.0, 400.0, 100.0]  # white on the left, red far away


def test_a_longer_stroke_sends_the_ball_further():
    travel = []
    for length in (40, 80, 120):
        state = BILLIARDS.push(APART, Push(start=(100.0, 256.0), end=(100.0 + length, 256.0))).state
        travel.append(state[0] - APART[0])
    assert 0 < travel[0] < travel[1] < travel[2]


def test_a_hard_stroke_bounces_off_the_cushion():
    run = BILLIARDS._run(APART, Push(start=(100.0, 256.0), end=(300.0, 256.0)))
    assert "white" in run.cushion
    assert 25 <= run.result.state[0] <= 487  # still on the table


def test_the_white_ball_can_move_the_red_one():
    line = [150.0, 256.0, 300.0, 256.0]
    run = BILLIARDS._run(line, Push(start=(100.0, 256.0), end=(300.0, 256.0)))
    assert run.hit == {"white"} and run.result.state[2] > 300 + 5


def on_table(state) -> bool:
    return bool(np.all((np.asarray(state) > 0) & (np.asarray(state) < 512)))


def test_a_ball_pinned_against_a_cushion_stays_on_the_table():
    pinned = [46.0, 462.0, 38.0, 233.0]  # the stroke drives the white ball into the left cushion
    result = BILLIARDS.push(pinned, Push(start=(109.0, 446.0), end=(24.0, 452.0)))
    assert result.moved and on_table(result.state)


def test_random_strokes_keep_the_balls_on_the_table():
    rng = np.random.default_rng(0)
    for _ in range(8):
        state = BILLIARDS.random_start(rng)
        for _ in range(8):
            state = BILLIARDS.push(state, BILLIARDS.aimed_push(state, rng)).state
            assert on_table(state)


def test_stroke_is_deterministic_and_misses_change_nothing():
    push = Push(start=(100.0, 256.0), end=(250.0, 256.0))
    assert BILLIARDS.push(APART, push).state == BILLIARDS.push(APART, push).state
    miss = BILLIARDS.push(APART, Push(start=(50.0, 450.0), end=(200.0, 450.0)))
    assert miss.valid and not miss.moved


def test_cue_cannot_start_on_a_ball():
    result = BILLIARDS.push(APART, Push(start=(150.0, 256.0), end=(250.0, 256.0)))
    assert not result.valid and "overlap" in result.error


@pytest.mark.parametrize("family", list(BILLIARDS.families))
def test_tasks_are_solvable(family):
    task = BILLIARDS.sample_task(family, 0, 0)
    state = BILLIARDS.push(task.start, task.ref_plan[0]).state
    assert BILLIARDS.check(task, state).success and not BILLIARDS.check(task, task.start).success
    assert "red ball" in BILLIARDS.state_text(task.start, task.goal)


def test_both_balls_are_visible(can_render):
    frame = BILLIARDS.render(APART)
    empty = BILLIARDS.render([-2000.0, -2000.0, -2000.0, -1000.0])
    assert (np.abs(frame.astype(int) - empty).sum(-1) > 60).sum() > 100
