"""System prompts. Bump PROMPT_VERSION on any change: it is stored in every episode log."""

from jat.env.pusht import MARGIN, MAX_LENGTH, MIN_LENGTH, PUSHER_RADIUS
from jat.env.tasks import ANGLE_TOLERANCE, POSITION_TOLERANCE, Task
from jat.tool.session import Budget

PROMPT_VERSION = "v1"

_WORLD = f"""\
You control a round pusher (radius {PUSHER_RADIUS:.0f} px) on a 512 x 512 px table seen from above.
Coordinates: x grows to the right, y grows downward, (0, 0) is the top-left corner.

The object is a T-shaped block: a bar 120 x 30 px with a stem 30 x 90 px under its middle.
At angle 0° the bar is on top and the stem points down. A positive angle turns the T clockwise
on screen. The T's position is its center of gravity, on the stem 45 px below the bar's top edge.

Physics: no momentum. The T moves only while the pusher touches it and stops at once.
A push whose line passes off the center of gravity rotates the T as well as moving it.
The table has walls on all four sides.

Action: one straight push {{"start": [x, y], "end": [x, y]}}. The pusher is lifted, placed at start,
then moved in a straight line to end. Rules: start and end inside [{MARGIN:.0f}, {512 - MARGIN:.0f}] on
both axes; length {MIN_LENGTH:.0f} to {MAX_LENGTH:.0f} px; the pusher at start must not overlap the T.
A push that never touches the T does nothing.
Every observation lists aim points on the T (tips, bar middle, stem end) to help you aim."""

_WITH_SIMULATE = """\
Before acting you can imagine. `simulate(from_frame, plans)` predicts the result of up to
{plans} plans of up to {pushes} pushes each, from any frame: a real one (R0, R1, ...) or an
imagined one (I1, I2, ...). Each predicted step gets a new frame id you can continue from.
You have {imagined} imagined pushes between two real pushes.
Good practice: compare several different candidates, keep the best, refine it or extend it from
its imagined frame, and only then act on the real table."""

_WITHOUT_SIMULATE = """\
You cannot test pushes in advance. Reason carefully about the geometry and the physics,
then act."""

_RULES = """\
Goal: {goal}
Success: center within {pos:.0f} px and angle within {ang:.0f}° of the goal. You have {pushes} real pushes.
Call `act` to push on the real table. Call `done` when the T is at the goal.
Call one tool per message."""


def system_prompt(task: Task, simulate_enabled: bool, budget: Budget) -> str:
    if simulate_enabled:
        tool_part = _WITH_SIMULATE.format(
            plans=budget.plans_per_call,
            pushes=budget.pushes_per_plan,
            imagined=budget.imagined_per_push,
        )
    else:
        tool_part = _WITHOUT_SIMULATE
    rules = _RULES.format(
        goal=task.text, pos=POSITION_TOLERANCE, ang=ANGLE_TOLERANCE, pushes=task.max_pushes
    )
    return "\n\n".join([_WORLD, tool_part, rules])
