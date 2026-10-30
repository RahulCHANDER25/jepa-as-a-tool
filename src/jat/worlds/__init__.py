"""The worlds an agent can act in. A world owns its physics, its frames, its rules and the text
the agent reads (`jat.worlds.base.World`); the world model, the tool, the agent and the
evaluation take a world by name.

Adding a world: one folder here with a class implementing World, and one line in `_WORLDS`.
"""

import importlib
from functools import cache

from jat.worlds.base import World

# name -> "module:class". Imported on first use: each world pulls in its own simulator.
_WORLDS = {
    "pusht": "jat.worlds.pusht.world:PushT",
    "tabletop": "jat.worlds.tabletop.world:Tabletop",
    "tabletop-new-look": "jat.worlds.tabletop.world:TabletopNewLook",
    "billiards": "jat.worlds.billiards.world:Billiards",
}
WORLDS = tuple(_WORLDS)


@cache
def get_world(name: str) -> World:
    try:
        path = _WORLDS[name]
    except KeyError:
        raise ValueError(f"unknown world {name!r}; expected one of {', '.join(WORLDS)}") from None
    module, cls = path.split(":")
    return getattr(importlib.import_module(module), cls)()
