from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from jat.agent.episode_log import AgentInfo, EpisodeLog
from jat.agent.loop import MAX_NUDGES, run_episode
from jat.tool.backends import OracleBackend
from jat.worlds import get_world

sample_task = get_world("pusht").sample_task

TASK = sample_task("rotate", seed=0, index=0)
AGENT = AgentInfo(provider="scripted", model="scripted", prompt_version="test", temperature=0.0)


class ScriptedModel(BaseChatModel):
    """Replays fixed replies, so the loop runs without an API."""

    replies: list[AIMessage]
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        reply = self.replies[len(self.seen) - 1]
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _call(name: str, args: dict, i: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


def _push(p) -> dict:
    return {"start": list(p.start), "end": list(p.end)}


def test_scripted_episode_imagines_acts_and_reaches_the_goal():
    a, b = (_push(p) for p in TASK.ref_plan)
    model = ScriptedModel(
        replies=[
            _call("simulate", {"from_frame": "R0", "plans": [[a, b]]}, 0),
            _call("act", {"push": a}, 1),
            _call("simulate", {"from_frame": "I1", "plans": [[b]]}, 2),
            _call("act", {"push": b}, 3),
            _call("done", {}, 4),
        ]
    )
    log = run_episode(
        TASK, "oracle_tool", OracleBackend(get_world("pusht")), model, AGENT, "test-episode"
    )

    assert log.outcome.success and log.stop_reason == "done"
    assert [f.id for f in log.frames] == ["R0", "I1", "I2", "R1", "I3", "R2"]
    imagined = next(f for f in log.frames if f.id == "I2")
    assert imagined.pose == imagined.true_pose  # oracle imagination is exact
    assert "I1 = R0 + push" in log.turns[0].tool_calls[0].result
    assert EpisodeLog.model_validate_json(log.model_dump_json()) == log


def test_bad_tool_arguments_come_back_as_text():
    model = ScriptedModel(
        replies=[
            _call("act", {"push": {"start": [1.0], "end": [2.0, 3.0]}}, 0),
            _call("done", {}, 1),
        ]
    )
    log = run_episode(
        TASK, "no_tool", OracleBackend(get_world("pusht")), model, AGENT, "test-bad-args"
    )
    assert "Invalid arguments" in log.turns[0].tool_calls[0].result
    assert log.stop_reason == "done"


def test_no_tool_setting_hides_simulate():
    model = ScriptedModel(
        replies=[
            _call("simulate", {"from_frame": "R0", "plans": [[_push(TASK.ref_plan[0])]]}, 0),
            _call("done", {}, 1),
        ]
    )
    log = run_episode(
        TASK, "no_tool", OracleBackend(get_world("pusht")), model, AGENT, "test-no-tool"
    )
    assert log.turns[0].tool_calls[0].result.startswith("Unknown tool 'simulate'")
    assert not any(f.kind == "imagined" for f in log.frames)


def test_episode_stops_after_repeated_replies_without_tool_calls():
    model = ScriptedModel(replies=[AIMessage(content="thinking...")] * (MAX_NUDGES + 1))
    log = run_episode(
        TASK, "oracle_tool", OracleBackend(get_world("pusht")), model, AGENT, "test-silent"
    )
    assert log.stop_reason == "no_tool_call" and len(log.turns) == MAX_NUDGES + 1
