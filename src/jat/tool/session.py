"""One episode as the LLM sees it: the real table, the frame tree, the budgets, and the tools
`simulate`, `act` and `done` (LangChain StructuredTools)."""

from dataclasses import dataclass
from textwrap import indent

import numpy as np
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from jat.tool.backends import Backend
from jat.tool.frames import FrameStore, UnknownFrame
from jat.worlds import get_world
from jat.worlds.base import Push, Task


@dataclass(frozen=True)
class Budget:
    imagined_per_push: int = 40  # resets after each real push
    plans_per_call: int = 8
    pushes_per_plan: int = 4


class SimulateArgs(BaseModel):
    from_frame: str = Field(
        description="Frame to imagine from: a real frame (R0, R1, ...) or an imagined one (I1, I2, ...)."
    )
    plans: list[list[Push]] = Field(
        description="Candidate plans to compare. Each plan is a list of pushes run in order."
    )


class ActArgs(BaseModel):
    push: Push = Field(description="The push to run on the real table.")


def _validation_message(error: ValidationError) -> str:
    details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return f"Invalid arguments ({details}). Nothing was run."


class Session:
    def __init__(
        self, task: Task, backend: Backend, simulate_enabled: bool, budget: Budget = Budget()
    ):
        self.task = task
        self.goal = np.array(task.goal)
        self.backend = backend
        self.simulate_enabled = simulate_enabled
        self.budget = budget
        self.world = get_world(task.world)
        self.state = np.array(task.start)  # the real table
        self.frames = FrameStore()
        self.pushes_used = 0
        self.imagined_left = budget.imagined_per_push
        self.stop_reason: str | None = None
        observed = backend.observe(self.state)
        self.frames.add_real(observed, backend.read([observed])[0], true_pose=self.state)

    @property
    def finished(self) -> bool:
        return self.stop_reason is not None

    @property
    def pushes_left(self) -> int:
        return self.task.max_pushes - self.pushes_used

    def stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason

    def observation(self) -> str:
        frame = self.frames.get(self.frames.current)
        return (
            f"Current real frame: {frame.id}\n{self.world.state_text(frame.pose, self.goal)}\n"
            f"Real pushes left: {self.pushes_left}."
        )

    # LLM-facing tools. Each returns text and never raises on bad input.

    def simulate(self, from_frame: str, plans: list[list[Push]]) -> str:
        if self.finished:
            return "The episode is over."
        try:
            root = self.frames.get(from_frame)
        except UnknownFrame:
            return f"Unknown frame '{from_frame}'. The current real frame is {self.frames.current}."
        b = self.budget
        if not 1 <= len(plans) <= b.plans_per_call:
            return f"Give between 1 and {b.plans_per_call} plans."
        if any(not 1 <= len(plan) <= b.pushes_per_plan for plan in plans):
            return f"Each plan needs between 1 and {b.pushes_per_plan} pushes."
        cost = sum(len(plan) for plan in plans)
        if cost > self.imagined_left:
            return (
                f"This call needs {cost} imagined pushes; {self.imagined_left} are left before your "
                "next real push. Use fewer or shorter plans."
            )

        heads = [root.id] * len(plans)  # last frame reached by each plan
        alive = [True] * len(plans)
        lines = [[f"plan {i + 1}:"] for i in range(len(plans))]
        for k in range(max(len(plan) for plan in plans)):
            runnable = []
            for i, plan in enumerate(plans):
                if not alive[i] or k >= len(plan):
                    continue
                parent = self.frames.get(heads[i])
                error = self.world.check_push(parent.pose, plan[k])
                if error is None:
                    runnable.append(i)
                else:
                    lines[i].append(
                        f"  push {plan[k].text()} from {parent.id} rejected: {error}. Plan stopped."
                    )
                    alive[i] = False
            if not runnable:
                continue
            # one batched backend call per step across all plans
            parents = [self.frames.get(heads[i]) for i in runnable]
            pushes = [plans[i][k] for i in runnable]
            states = self.backend.step([p.state for p in parents], pushes)
            for i, parent, push, state, pose in zip(
                runnable, parents, pushes, states, self.backend.read(states), strict=True
            ):
                frame_id = self.frames.add_imagined(parent.id, push, state, pose)
                heads[i] = frame_id
                self.imagined_left -= 1
                note = (
                    ""
                    if self.world.moved(parent.pose, pose)
                    else "  (the T does not move: the push misses it)"
                )
                lines[i].append(f"  {frame_id} = {parent.id} + push {push.text()}{note}")
                lines[i].append(indent(self.world.state_text(pose, self.goal), "    "))

        footer = f"{self.imagined_left} imagined pushes left before your next real push."
        return "\n".join("\n".join(block) for block in lines) + "\n" + footer

    def act(self, push: Push) -> str:
        if self.finished:
            return "The episode is over."
        result = self.world.push(self.state, push)
        if not result.valid:
            return f"Push rejected: {result.error}. Nothing happened and no push was used."
        self.state = np.array(result.state)
        observed = self.backend.observe(self.state)
        pose = self.backend.read([observed])[0]
        frame_id = self.frames.add_real(
            observed,
            pose,
            true_pose=self.state,
            parent=self.frames.current,
            push=push,
            trajectory=result.trajectory,
        )
        self.pushes_used += 1
        self.imagined_left = self.budget.imagined_per_push
        lines = [f"Pushed {push.text()} on the real table.", f"Current real frame: {frame_id}"]
        lines.append(self.world.state_text(pose, self.goal))
        if not result.moved:
            lines.append("The T did not move: the push missed it.")
        if result.jammed:
            lines.append("The T got pinned against a wall, so the push stopped early.")
        if self.pushes_left == 0:
            self.stop("pushes_exhausted")
            lines.append("No real pushes left. The episode is over.")
        else:
            lines.append(f"Real pushes left: {self.pushes_left}.")
        return "\n".join(lines)

    def done(self) -> str:
        self.stop("done")
        return f"Episode ended at frame {self.frames.current}."

    def tools(self) -> list[StructuredTool]:
        b = self.budget
        tools = []
        if self.simulate_enabled:
            tools.append(
                StructuredTool.from_function(
                    func=self.simulate,
                    name="simulate",
                    description=(
                        "Imagine pushes without touching the real table. Predicts where the T ends up "
                        "after each push of each plan, starting from `from_frame`. Each predicted step "
                        "gets a new frame id (I1, I2, ...) that a later call can start from. "
                        f"Up to {b.plans_per_call} plans of up to {b.pushes_per_plan} pushes per call; "
                        f"{b.imagined_per_push} imagined pushes between two real pushes."
                    ),
                    args_schema=SimulateArgs,
                    handle_validation_error=_validation_message,
                )
            )
        tools.append(
            StructuredTool.from_function(
                func=self.act,
                name="act",
                description="Run one push on the real table. Uses one of your real pushes.",
                args_schema=ActArgs,
                handle_validation_error=_validation_message,
            )
        )
        tools.append(
            StructuredTool.from_function(
                func=self.done,
                name="done",
                description="End the episode now. Call it when the T is at the goal.",
            )
        )
        return tools
