# Add your own world

A world is a table on which an agent pushes things. Give the toolkit one, and you get the whole
chain: random-push data, V-JEPA 2 latents, a translator head, a predictor, the `simulate` tool,
agent evaluations, an MCP server, a browser demo and video clips. Nothing outside your folder
changes.

The worked example is billiards: [`src/jat/worlds/billiards/world.py`](../src/jat/worlds/billiards/world.py),
about 300 lines, written exactly this way.

## 1. One folder, one class

```
src/jat/worlds/
  myworld/
    __init__.py      one-line docstring
    world.py         class MyWorld
```

and one line in [`src/jat/worlds/__init__.py`](../src/jat/worlds/__init__.py):

```python
_WORLDS = {
    ...
    "myworld": "jat.worlds.myworld.world:MyWorld",
}
```

## 2. What the class provides

The contract is the `World` protocol in [`src/jat/worlds/base.py`](../src/jat/worlds/base.py).
A state is a flat list of floats: whatever fully describes the table at rest.

| Member | What it does | Billiards |
|---|---|---|
| `name` | the registry name | `"billiards"` |
| `push(state, push) -> PushResult` | run one push, wait until everything is at rest. **Stateless**: the result depends on (state, push) only | MuJoCo, the cue moves along the stroke in 0.25 s |
| `check_push(state, push)` | why a push cannot run, or `None`. Start with `base.check_bounds` | the cue tip must not start on a ball |
| `moved(before, after)` | did anything move | any ball moved more than 1 px |
| `render(state, size)` | the frame V-JEPA 2 sees, `uint8 [size, size, 3]`, no pusher | the 3D table from `table3d` |
| `random_start(rng)`, `aimed_push(state, rng)` | random data: starts, and pushes that hit something | two balls apart; a stroke at one ball |
| `readout` | what the translator reads: `"t_pose"` or `"points:<n>"` | `"points:2"` |
| `describe()`, `rules(task)` | the world and the goal parts of the agent's system prompt | table, balls, how strokes work |
| `state_text(state, goal)`, `goal_text(goal)` | a state and a goal as the agent reads them | `White ball (250, 416), red ball (435, 197)` |
| `tool_text` | the world's words in the tool's replies | "no ball moves: the stroke misses" |
| `families`, `sample_task(family, seed, index)`, `check(task, state)` | evaluation tasks: make goals by running random pushes in your physics, so every task is solvable | shot, bank, carom |
| `outlines(state, goal)` | top-view shapes for sketches (MCP pictures) | two circles |
| `model_world` | whose world model to use: your own name | `"billiards"` |

Worlds with a T can subclass `tblock.world.TWorld` and only bring `push` and `render` (that is
how [`tabletop`](../src/jat/worlds/tabletop/world.py) works). 3D worlds can reuse the MuJoCo table,
camera and pusher in [`table3d`](../src/jat/worlds/table3d/scene.py); adding `clip_scene` and
`poses` then gives you `jat clip` videos too.

## 3. Check that it behaves

Before training anything, write tests like [`tests/worlds/test_billiards.py`](../tests/worlds/test_billiards.py):

- a push moves what it should, and a push that misses changes nothing;
- the same state and push always give the same result;
- `sample_task` tasks are solved by their own `ref_plan`;
- the objects are visible in `render` wherever they can be.

## 4. Train its world model

```bash
jat gen data/myworld --world myworld --episodes 5000 --steps 8      # CPU
jat encode data/myworld                                             # GPU, V-JEPA 2
jat train translator data/myworld --out runs/myworld                # a head of its own
jat train predictor data/myworld --out runs/myworld --rollout 4     # its physics, in latent space
jat wm-eval data/myworld --wm runs/myworld                          # imagined vs true, no LLM
```

Few labels? Reuse a trained trunk and train only your world's head:

```bash
jat train translator data/myworld --out runs/myworld --trunk-from runs/shared/translator.pt
```

## 5. Use it

```bash
jat eval --world myworld --families <f> --settings no_tool,wm_tool,oracle_tool --wm runs/myworld --out results/myworld
jat mcp --world myworld --wm runs/myworld          # for Claude Desktop, Cursor, any MCP client
jat export-dream --wm runs/myworld --out web/public/dream/myworld
```
