from jat.agent.episode_log import AgentInfo
from jat.agent.llm import chat_model
from jat.agent.loop import run_episode
from jat.clip import Film, _last_thought_before_each_push, render_clip
from jat.tool.backends import OracleBackend
from jat.worlds import get_world


def test_a_3d_episode_becomes_a_video(tmp_path, can_render):
    world = get_world("tabletop")
    task = world.sample_task("rotate", 0, 0)
    agent = AgentInfo(provider="dry", model="dry", prompt_version="test", temperature=0.0)
    log = run_episode(task, "oracle_tool", OracleBackend(world), chat_model("dry", 0), agent, "x")
    out = render_clip(log, tmp_path / "clip.mp4", size=128)
    assert out.exists() and out.stat().st_size > 1000
    assert out.with_suffix(".vtt").read_text().startswith("WEBVTT")


def test_each_push_is_captioned_with_the_last_words_before_it():
    world = get_world("tabletop")
    task = world.sample_task("rotate", 0, 0)
    agent = AgentInfo(provider="dry", model="dry", prompt_version="test", temperature=0.0)
    log = run_episode(task, "oracle_tool", OracleBackend(world), chat_model("dry", 0), agent, "x")
    said = _last_thought_before_each_push(log)
    real = [f.id for f in log.frames if f.kind == "real" and f.parent is not None]
    assert real and list(said) == real


def test_the_words_are_subtitles_timed_with_the_frames(can_render):
    world = get_world("tabletop")
    state = world.sample_task("rotate", 0, 0).start
    film = Film(world, 128)
    film.text = "push the stem to the left"
    film.shot(state, seconds=1.0)
    film.shot(state, seconds=0.5)
    film.text = ""
    film.shot(state, seconds=1.0)
    assert len(film.frames) == 75
    assert (
        film.subtitles() == "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\npush the stem to the left\n"
    )


def test_a_2d_episode_becomes_a_video_too(tmp_path):
    world = get_world("pusht")
    task = world.sample_task("rotate", 0, 0)
    agent = AgentInfo(provider="dry", model="dry", prompt_version="test", temperature=0.0)
    log = run_episode(task, "oracle_tool", OracleBackend(world), chat_model("dry", 0), agent, "x")
    out = render_clip(log, tmp_path / "clip.mp4", size=128)
    assert out.exists() and out.stat().st_size > 1000
