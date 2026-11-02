import numpy as np

from jat.agent.episode_log import AgentInfo
from jat.agent.llm import chat_model
from jat.agent.loop import run_episode
from jat.clip import CAPTION_HEIGHT, Film, _last_thought_before_each_push, render_clip
from jat.tool.backends import OracleBackend
from jat.worlds import get_world


def test_a_3d_episode_becomes_a_video(tmp_path, can_render):
    world = get_world("tabletop")
    task = world.sample_task("rotate", 0, 0)
    agent = AgentInfo(provider="dry", model="dry", prompt_version="test", temperature=0.0)
    log = run_episode(task, "oracle_tool", OracleBackend(world), chat_model("dry", 0), agent, "x")
    out = render_clip(log, tmp_path / "clip.mp4", size=128)
    assert out.exists() and out.stat().st_size > 1000


def test_each_push_is_captioned_with_the_last_words_before_it():
    world = get_world("tabletop")
    task = world.sample_task("rotate", 0, 0)
    agent = AgentInfo(provider="dry", model="dry", prompt_version="test", temperature=0.0)
    log = run_episode(task, "oracle_tool", OracleBackend(world), chat_model("dry", 0), agent, "x")
    said = _last_thought_before_each_push(log)
    real = [f.id for f in log.frames if f.kind == "real" and f.parent is not None]
    assert real and list(said) == real


def test_the_caption_sits_above_the_table_and_never_covers_it(can_render):
    world = get_world("tabletop")
    state = world.sample_task("rotate", 0, 0).start
    film = Film(world, 128)
    film.text = "a long thought about where to push the T next " * 10
    film.shot(state)
    film.text = ""
    film.shot(state)
    words, silent = film.frames
    assert words.shape == silent.shape == (128 + CAPTION_HEIGHT, 128, 3)
    assert np.array_equal(words[CAPTION_HEIGHT:], silent[CAPTION_HEIGHT:])
    assert not np.array_equal(words[:CAPTION_HEIGHT], silent[:CAPTION_HEIGHT])


def test_a_2d_episode_becomes_a_video_too(tmp_path):
    world = get_world("pusht")
    task = world.sample_task("rotate", 0, 0)
    agent = AgentInfo(provider="dry", model="dry", prompt_version="test", temperature=0.0)
    log = run_episode(task, "oracle_tool", OracleBackend(world), chat_model("dry", 0), agent, "x")
    out = render_clip(log, tmp_path / "clip.mp4", size=128)
    assert out.exists() and out.stat().st_size > 1000
