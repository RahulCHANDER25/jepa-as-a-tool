"""Billiards on the 3D table: a white and a red ball. The pusher becomes a cue: it moves along
the stroke in a fixed time, so a longer stroke hits harder, and withdraws once it has struck a
ball. Balls roll, hit each other and bounce off the cushions, then slow down and stop. The state
is the two balls' positions.

Tasks: put the red ball at a goal point. The goal is where a random strike sent it, so every
task is solvable; families differ in how that strike got it there (directly, off a cushion,
or through the white ball).
"""

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from jat.worlds.base import (
    MARGIN,
    MAX_LENGTH,
    MIN_LENGTH,
    PUSHER_RADIUS,
    TABLE,
    Outcome,
    Push,
    PushResult,
    Task,
    ToolText,
    check_bounds,
)
from jat.worlds.table3d import scene as s3d

RADIUS = 25.0  # px = mm
NAMES = ("white", "red")
TIMESTEP = 0.002  # contacts span 4 steps: at 2, a ball gains energy at each bounce
STROKE_TIME = 0.25  # s: every stroke takes this long, so its length sets its speed
# How long the cue stays on a ball, then it withdraws. A cue that keeps pushing a ball held by a
# cushion drives it into the cushion, which then fires it off the table.
STRIKE_TIME = 0.02  # s
SETTLE_MAX = 5.0  # s
REST_SPEED = 0.005  # m/s
LOG_EVERY = 0.1  # s
POSITION_TOLERANCE = 20.0  # px
START_WALL_MARGIN = 50.0
GOAL_WALL_MARGIN = 40.0
MIN_GAP = 30.0  # px between the two balls at the start
MAX_TRIES = 3000

CUSHION_HEIGHT = 0.035  # m, taller than a ball's radius

_BALL_CONTACT = 'condim="6" friction="0.2 0.005 0.003" solref="0.008 0.3"'
_CUSHION_CONTACT = 'solref="0.008 0.15"'  # a damping ratio well under 1: a bouncy cushion

_DESCRIBE = f"""\
You control a round cue tip (radius {PUSHER_RADIUS:.0f} px) on a 512 x 512 px billiards table.
Coordinates: x grows to the right, y grows downward, (0, 0) is the top-left corner.

Two balls of radius {RADIUS:.0f} px lie on the table: a white ball and a red ball.

Physics: a strike is a straight stroke of the cue. Every stroke takes {STROKE_TIME:.2f} s, so a longer
stroke hits harder. A ball keeps rolling after the cue leaves it, slows down and stops. Balls
bounce off each other and off the cushions on all four sides of the table.

Action: one straight stroke {{"start": [x, y], "end": [x, y]}}. The cue tip is lifted, placed at
start, then moved in a straight line to end. Rules: start and end inside [{MARGIN:.0f}, {TABLE - MARGIN:.0f}] on
both axes; length {MIN_LENGTH:.0f} to {MAX_LENGTH:.0f} px; the cue tip at start must not overlap a ball.
A stroke that never touches a ball does nothing. You may strike either ball."""

_RULES = """\
Goal: {goal}
Success: the red ball's center within {pos:.0f} px of the goal point. You have {pushes} strikes.
Call `act` to strike on the real table. Call `done` when the red ball is at the goal.
Call one tool per message."""


@dataclass(frozen=True)
class Family:
    max_pushes: int
    min_travel: float  # px the red ball moves in the reference strike
    ball: str  # the ball the reference strike hits
    cushion: bool  # the red ball must touch a cushion during the reference strike


FAMILIES = {
    "shot": Family(max_pushes=2, min_travel=100, ball="red", cushion=False),
    "bank": Family(max_pushes=2, min_travel=80, ball="red", cushion=True),
    "carom": Family(max_pushes=2, min_travel=60, ball="white", cushion=False),
}


def _ball(name: str, rgba: str) -> str:
    r = RADIUS * s3d.MM
    return f"""
    <body name="{name}" pos="0 0 {r}">
      <freejoint name="{name}"/>
      <geom name="{name}" type="sphere" size="{r}" mass="0.05" rgba="{rgba}" {_BALL_CONTACT}/>
    </body>"""


@dataclass
class _Run:
    result: PushResult
    hit: set  # balls the cue touched
    cushion: set  # balls that touched a cushion


class Billiards:
    name = "billiards"
    families = FAMILIES
    readout = f"points:{len(NAMES)}"
    model_world = "billiards"
    look = s3d.DEFAULT_LOOK
    tool_text = ToolText(
        ends_up="where the balls end up",
        misses="no ball moves: the stroke misses",
        missed="No ball moved: the stroke missed.",
        at_goal="the red ball is at the goal",
    )

    goal_objects = (1,)  # for clips: the goal is the red ball's

    @cached_property
    def scene(self) -> s3d.Scene:
        return self.clip_scene(0)

    def clip_scene(self, slots: int) -> s3d.Scene:
        """The scene, plus `slots` ghost pairs of balls for clips."""
        objects = _ball("white", "0.95 0.95 0.92 1") + _ball("red", self.look.second_rgba)
        geom = f'<geom type="sphere" size="{RADIUS * s3d.MM}" rgba="1 1 1 0.3"/>'
        xml = s3d.scene_xml(
            objects, self.look, TIMESTEP, rim_class=_CUSHION_CONTACT, rim_height=CUSHION_HEIGHT,
            ghosts=s3d.ghost_bodies(geom, len(NAMES), slots),
        )  # fmt: skip
        return s3d.Scene(xml)

    def poses(self, state) -> list[tuple[list[float], np.ndarray]]:
        """(position, quaternion) of each object's body."""
        r = RADIUS * s3d.MM
        return [([*s3d.to_world(x, y), r], np.array([1.0, 0, 0, 0])) for x, y in self._balls(state)]

    @cached_property
    def _geoms(self) -> dict[str, int]:
        m, mj = self.scene.model, self.scene.mujoco
        names = [*NAMES, "pusher", "rim_left", "rim_right", "rim_far", "rim_near"]
        return {n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, n) for n in names}

    @staticmethod
    def _balls(state) -> np.ndarray:
        return np.asarray(state, dtype=float).reshape(len(NAMES), 2)

    def _place(self, data, state) -> None:
        r = RADIUS * s3d.MM
        for i, (x, y) in enumerate(self._balls(state)):
            data.qpos[7 * i : 7 * i + 7] = [*s3d.to_world(x, y), r, 1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0

    @staticmethod
    def _state(data) -> np.ndarray:
        return np.array(
            [c for i in range(len(NAMES)) for c in s3d.from_world(*data.qpos[7 * i : 7 * i + 2])]
        )

    # World protocol

    def check_push(self, state, push: Push) -> str | None:
        if (error := check_bounds(push)) is not None:
            return error
        gaps = np.linalg.norm(self._balls(state) - push.start, axis=1)
        if (gaps < RADIUS + PUSHER_RADIUS + 1).any():
            return "the cue tip at start would overlap a ball; start outside the balls"
        return None

    def moved(self, before, after) -> bool:
        return bool((np.linalg.norm(self._balls(after) - self._balls(before), axis=1) > 1.0).any())

    def push(self, state, push: Push) -> PushResult:
        return self._run(state, push).result

    def _run(self, state, push: Push) -> _Run:
        before = np.asarray(state, dtype=float)
        error = self.check_push(before, push)
        if error is not None:
            return _Run(PushResult(state=before.tolist(), valid=False, error=error), set(), set())

        mj, model, data = self.scene.mujoco, self.scene.model, self.scene.data
        mj.mj_resetData(model, data)  # the result depends on (state, push) only
        self._place(data, before)
        start = np.array(s3d.to_world(*push.start))
        end = np.array(s3d.to_world(*push.end))
        stroke_steps = round(STROKE_TIME / TIMESTEP)
        log_every = round(LOG_EVERY / TIMESTEP)
        data.mocap_pos[0] = [*start, s3d.PUSHER_Z]
        mj.mj_forward(model, data)

        g = self._geoms
        ball_of = {g[n]: n for n in NAMES}
        rims = {g["rim_left"], g["rim_right"], g["rim_far"], g["rim_near"]}
        hit, cushion = set(), set()
        trajectory = []
        cue = push.start
        stroke_end = stroke_steps  # the cue withdraws once it has struck
        struck = 0  # steps the cue has spent on a ball
        strike_steps = round(STRIKE_TIME / TIMESTEP)
        for i in range(1, stroke_steps + round(SETTLE_MAX / TIMESTEP) + 1):
            if i <= stroke_end:
                target = start + (end - start) * i / stroke_steps
                data.mocap_pos[0] = [*target, s3d.PUSHER_Z]
                cue = s3d.from_world(*target)
            elif i == stroke_end + 1:
                data.mocap_pos[0] = s3d.PARKED
            mj.mj_step(model, data)
            on_ball = False
            for c in data.contact[: data.ncon]:
                pair = {c.geom1, c.geom2}
                for geom, name in ball_of.items():
                    if geom in pair:
                        if g["pusher"] in pair:
                            hit.add(name)
                            on_ball = True
                        if pair & rims:
                            cushion.add(name)
            struck += on_ball
            if struck >= strike_steps and i < stroke_end:
                stroke_end = i  # a strike, not a shove
            if i % log_every == 0:
                trajectory.append((*cue, *self._state(data)))
            speeds = [np.linalg.norm(data.qvel[6 * k : 6 * k + 3]) for k in range(len(NAMES))]
            if i > stroke_steps and max(speeds) < REST_SPEED:
                break
        after = self._state(data)
        result = PushResult(
            state=after.tolist(), valid=True, moved=self.moved(before, after), trajectory=trajectory
        )
        return _Run(result, hit, cushion)

    def render(self, state, size: int = 256) -> np.ndarray:
        return self.scene.render(lambda data: self._place(data, state), size)

    def random_start(self, rng: np.random.Generator) -> np.ndarray:
        while True:
            balls = rng.uniform(START_WALL_MARGIN, TABLE - START_WALL_MARGIN, size=(len(NAMES), 2))
            if np.linalg.norm(balls[0] - balls[1]) >= 2 * RADIUS + MIN_GAP:
                return balls.reshape(-1)

    def aimed_push(self, state, rng: np.random.Generator, ball: str | None = None) -> Push:
        """A stroke aimed at one ball (random unless given), a little off its center."""
        balls = self._balls(state)
        for _ in range(200):
            target = balls[NAMES.index(ball or NAMES[rng.integers(len(NAMES))])]
            angle = rng.uniform(0, 2 * math.pi)
            direction = np.array([math.cos(angle), math.sin(angle)])
            side = np.array([-direction[1], direction[0]])
            offset = rng.uniform(-0.5, 0.5) * RADIUS  # a cut: hits off the center line
            gap = rng.uniform(5, 30)
            start = target + side * offset - direction * (RADIUS + PUSHER_RADIUS + gap)
            length = rng.uniform(gap + 10, MAX_LENGTH)
            push = Push(start=tuple(start), end=tuple(start + direction * length))
            if self.check_push(state, push) is None:
                return push
        raise RuntimeError("no valid aimed stroke from this state")

    def describe(self) -> str:
        return _DESCRIBE

    def rules(self, task: Task) -> str:
        return _RULES.format(goal=task.text, pos=POSITION_TOLERANCE, pushes=task.max_pushes)

    def state_text(self, state, goal=None) -> str:
        (wx, wy), (rx, ry) = self._balls(state)
        lines = [f"White ball ({wx:.0f}, {wy:.0f}), red ball ({rx:.0f}, {ry:.0f})"]
        if goal is not None:
            lines.append(f"  goal error: red ball {self._error(state, goal):.0f} px from the goal")
        return "\n".join(lines)

    def outlines(self, state, goal: bool = False) -> list[tuple[tuple[int, int, int], np.ndarray]]:
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        circle = RADIUS * np.stack([np.cos(angles), np.sin(angles)], -1)
        balls = list(zip(((225, 225, 215), (200, 40, 40)), self._balls(state), strict=True))
        return [(color, center + circle) for color, center in balls[1 if goal else 0 :]]

    def goal_text(self, goal) -> str:
        rx, ry = self._balls(goal)[1]
        return f"Put the red ball at ({rx:.0f}, {ry:.0f}). The white ball may end anywhere."

    def _error(self, state, goal) -> float:
        return float(np.linalg.norm(self._balls(state)[1] - self._balls(goal)[1]))

    def sample_task(self, family: str, seed: int, index: int) -> Task:
        spec = self.families[family]
        rng = np.random.default_rng([seed, index])
        for _ in range(MAX_TRIES):
            start = self.random_start(rng)
            push = self.aimed_push(start, rng, ball=spec.ball)
            run = self._run(start, push)
            goal = np.array(run.result.state)
            red = self._balls(goal)[1]
            if (
                run.hit == {spec.ball}
                and self._error(goal, start) >= spec.min_travel
                and ("red" in run.cushion) == spec.cushion
                and ((red >= GOAL_WALL_MARGIN) & (red <= TABLE - GOAL_WALL_MARGIN)).all()
            ):
                return Task(
                    id=f"{family}-{seed}-{index:03d}",
                    family=family,
                    seed=seed,
                    start=start.tolist(),
                    goal=goal.tolist(),
                    max_pushes=spec.max_pushes,
                    ref_plan=[push],
                    world=self.name,
                )
        raise RuntimeError(
            f"no {family} task found in {MAX_TRIES} tries (seed {seed}, index {index})"
        )

    def check(self, task: Task, state) -> Outcome:
        error = self._error(state, task.goal)
        return Outcome(success=error < POSITION_TOLERANCE, position_error=error, angle_error=0.0)
