"""The agent loop: the LLM calls `simulate` / `act` / `done` until the episode ends."""

import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from jat.agent.episode_log import AgentInfo, EpisodeLog, ToolCallLog, TurnLog, frame_logs
from jat.agent.prompts import system_prompt
from jat.tool.backends import Backend
from jat.tool.session import Budget, Session
from jat.worlds.base import Task

MAX_TURNS = 40
MAX_NUDGES = 3  # replies in a row without a tool call before giving up


def _reasoning(reply: AIMessage) -> str | None:
    """Reasoning text, wherever this provider puts it."""
    # UNVERIFIED for Muse Spark Open on Baseten: check the field in the M0 Baseten check.
    if text := reply.additional_kwargs.get("reasoning_content"):
        return text
    blocks = [b.get("reasoning", "") for b in reply.content_blocks if b.get("type") == "reasoning"]
    return "\n".join(blocks) or None


def run_episode(
    task: Task,
    setting: str,
    backend: Backend,
    model: BaseChatModel,
    agent: AgentInfo,
    episode_id: str,
    budget: Budget = Budget(),
) -> EpisodeLog:
    started = time.time()
    session = Session(task, backend, simulate_enabled=setting != "no_tool", budget=budget)
    tools = session.tools()
    by_name = {tool.name: tool for tool in tools}
    llm = model.bind_tools(tools)
    messages = [
        SystemMessage(system_prompt(task, session.simulate_enabled, budget)),
        HumanMessage(session.observation()),
    ]
    turns, nudges = [], 0

    while not session.finished and len(turns) < MAX_TURNS:
        reply = llm.invoke(messages)
        messages.append(reply)
        calls = []
        for call in reply.tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                result = f"Unknown tool '{call['name']}'. Use one of: {', '.join(by_name)}."
            else:
                result = str(tool.invoke(call["args"]))
            calls.append(ToolCallLog(name=call["name"], args=call["args"], result=result))
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
        usage = reply.usage_metadata or {}
        turns.append(
            TurnLog(
                content=reply.text,
                reasoning=_reasoning(reply),
                tool_calls=calls,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
            )
        )
        if calls:
            nudges = 0
            continue
        nudges += 1
        if nudges > MAX_NUDGES:
            session.stop("no_tool_call")
            break
        messages.append(HumanMessage(f"Reply with a tool call: {', '.join(by_name)}."))

    session.stop("max_turns")  # no-op when the episode already ended for another reason
    final_pose = session.state
    return EpisodeLog(
        episode_id=episode_id,
        setting=setting,
        backend=backend.name,
        agent=agent,
        task=task,
        frames=frame_logs(session.frames, session.world),
        turns=turns,
        outcome=session.world.check(task, final_pose),
        final_pose=final_pose.tolist(),
        stop_reason=session.stop_reason,
        wall_time_s=time.time() - started,
    )
