from jat.agent.episode_log import AgentInfo
from jat.agent.llm import chat_model
from jat.agent.loop import run_episode
from jat.clip import render_clip
from jat.tool.backends import OracleBackend
from jat.worlds import get_world


def test_a_3d_episode_becomes_a_video(tmp_path, can_render):
    world = get_world("tabletop")
    task = world.sample_task("rotate", 0, 0)
    agent = AgentInfo(provider="dry", model="dry", prompt_version="test", temperature=0.0)
    log = run_episode(task, "oracle_tool", OracleBackend(world), chat_model("dry", 0), agent, "x")
    out = render_clip(log, tmp_path / "clip.mp4", size=128)
    assert out.exists() and out.stat().st_size > 1000
