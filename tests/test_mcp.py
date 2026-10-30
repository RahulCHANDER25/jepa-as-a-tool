import anyio
from mcp import Client

from jat.mcp_server import Game, build_server


async def _play():
    game = Game("billiards")
    async with Client(build_server(game)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        start = await client.call_tool("start_task", {"family": "shot"})
        push = game.session.task.ref_plan[0].model_dump()
        imagined = await client.call_tool("simulate", {"from_frame": "R0", "plans": [[push]]})
        acted = await client.call_tool("act", {"push": push})
        done = await client.call_tool("done", {})
    return names, start, imagined, acted, done


def test_an_mcp_client_plays_a_task_with_text_and_pictures():
    names, start, imagined, acted, done = anyio.run(_play)
    assert names == {"start_task", "simulate", "act", "done"}
    assert "red ball" in start.content[0].text and start.content[1].type == "image"
    assert "I1 = R0 + push" in imagined.content[0].text and imagined.content[1].type == "image"
    assert "Current real frame: R1" in acted.content[0].text
    assert done.content[0].text.startswith("Success")  # the task's own reference strike


def test_tools_before_a_task_answer_with_an_error():
    async def call():
        async with Client(build_server(Game("pusht"))) as client:
            return await client.call_tool("done", {})

    result = anyio.run(call)
    assert result.is_error and "start_task" in result.content[0].text
