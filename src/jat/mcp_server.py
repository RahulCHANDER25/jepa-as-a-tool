"""An MCP server: any MCP client (Claude Desktop, Cursor, an agent framework) gets the tools our
agent uses. Start a task, imagine pushes in the V-JEPA 2 world model, push on the real table.
Every reply is the same text our agent reads, plus a top view of the table and the imagined
states.

    jat mcp --world billiards --wm runs/billiards    # imagination: the world model
    jat mcp --world pusht                            # imagination: the real physics (oracle)

One episode at a time per server: a client starts a task, then plays it.
"""

from pathlib import Path

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from jat.agent.prompts import system_prompt
from jat.tool.backends import OracleBackend, WorldModelBackend
from jat.tool.session import Budget, Session
from jat.tool.sketch import png, sketch
from jat.worlds import get_world
from jat.worlds.base import Push


class Game:
    """The state behind the MCP tools: one world, one backend, the current episode."""

    def __init__(self, world: str, wm: Path | None = None, budget: Budget = Budget()):
        self.world = get_world(world)
        if wm is not None:
            from jat.wm.models import WorldModel

            self.backend = WorldModelBackend(WorldModel.load(wm, world=world))
        else:
            self.backend = OracleBackend(self.world)
        self.budget = budget
        self.session: Session | None = None

    def _image(self, imagined: list[str] = (), picked: str | None = None) -> Image:
        s = self.session
        ghosts = [(i, s.frames.get(i).pose) for i in imagined]
        frame = sketch(self.world, s.state, s.task.goal, ghosts, picked)
        return Image(data=png(frame), format="png")

    def _need_task(self) -> Session:
        if self.session is None:
            raise ToolError("No task yet: call start_task first.")
        return self.session

    def start_task(self, family: str = "", seed: int = 0, index: int = 0) -> list:
        family = family or next(iter(self.world.families))
        if family not in self.world.families:
            raise ToolError(f"Unknown family {family!r}: {', '.join(self.world.families)}")
        task = self.world.sample_task(family, seed, index)
        self.session = Session(task, self.backend, simulate_enabled=True, budget=self.budget)
        rules = system_prompt(task, True, self.budget)
        return [f"{rules}\n\n{self.session.observation()}", self._image()]

    def simulate(self, from_frame: str, plans: list[list[Push]]) -> list:
        session = self._need_task()
        before = set(session.frames.frames)
        text = session.simulate(from_frame, plans)
        new = [i for i in session.frames.frames if i not in before]
        return [text, self._image(new)]

    def act(self, push: Push) -> list:
        session = self._need_task()
        return [session.act(push), self._image()]

    def done(self) -> str:
        session = self._need_task()
        session.done()
        outcome = self.world.check(session.task, session.state)
        verdict = "Success" if outcome.success else "Not at the goal"
        return (
            f"{verdict}: {outcome.position_error:.0f} px from the goal, {outcome.angle_error:.0f}°."
        )


def build_server(game: Game) -> MCPServer:
    families = ", ".join(game.world.families)
    server = MCPServer(
        name="jepa-as-a-tool",
        instructions=(
            f"A {game.world.name} table. start_task gives the rules and the goal. Before each real "
            "push (act), imagine candidate plans with simulate: it predicts the result of each "
            f"push without touching the real table. Task families: {families}."
        ),
    )
    server.tool(description=f"Start a new task. family: one of {families}.")(game.start_task)
    server.tool(
        description=(
            "Imagine plans without touching the real table. from_frame: a real frame (R0, R1, "
            "...) or an imagined one (I1, I2, ...). Each plan is a list of pushes "
            '{"start": [x, y], "end": [x, y]}; each predicted step gets a new frame id.'
        )
    )(game.simulate)
    server.tool(description="Push once on the real table. Uses one real push.")(game.act)
    server.tool(description="End the task and get the result.")(game.done)
    return server
