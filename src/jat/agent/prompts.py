"""System prompts: the world's description, the tool part, the world's rules. Bump PROMPT_VERSION on
any change to the text of an existing world: it is stored in every episode log."""

from jat.tool.session import Budget
from jat.worlds import get_world
from jat.worlds.base import Task

PROMPT_VERSION = "v1"

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


def system_prompt(task: Task, simulate_enabled: bool, budget: Budget) -> str:
    if simulate_enabled:
        tool_part = _WITH_SIMULATE.format(
            plans=budget.plans_per_call,
            pushes=budget.pushes_per_plan,
            imagined=budget.imagined_per_push,
        )
    else:
        tool_part = _WITHOUT_SIMULATE
    world = get_world(task.world)
    return "\n\n".join([world.describe(), tool_part, world.rules(task)])
