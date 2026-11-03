# jepa-as-a-tool

**An LLM agent that imagines its moves in a V-JEPA 2 world model before touching the real table.**

Entry to the Meta Global AI Developer Hackathon (Oct 27 – Nov 4, 2026).
Website with live episodes: https://rahulchander25.github.io/jepa-as-a-tool/

Muse Spark pushes objects to a goal, one straight push at a time, in three worlds: PushT (2D), a 3D
tabletop and 3D billiards. Before each real push it can call one tool, `simulate(from_frame,
plans)`, which imagines candidate push sequences in a learned world model and says, in text, where
things would end up. **One** world model serves the three worlds; it never saw the physics engines,
it learned from images only.

## How it works

```
frame ──▶ V-JEPA 2 (frozen) ──▶ latent ──▶ predictor (+ push) ──▶ latent ──▶ translator ──▶ "T at (212, 180), 35°"
                                              ▲ chains of pushes stay in latent space
```

| Part | What it is | Trained here? |
|---|---|---|
| **Encoder** | Meta's V-JEPA 2 (ViT-L, `facebook/vjepa2-vitl-fpc64-256`): a 256 px frame becomes 64 tokens × 1,024 | no, frozen |
| **Predictor** | a 6-layer transformer: latent + push → next latent; one for the three worlds, not told which one | yes |
| **Translator** | latent → positions: a shared trunk and one head per world; the only thing the LLM reads | yes |
| **Tool** | `simulate`: several plans per call, an id for every imagined state (`I1`, `I2`, …) so the agent can continue a branch | — |
| **Agent** | Muse Spark 1.3 on the Meta Model API, with LangChain tools and our own loop | no |

The same tool also runs on an **oracle** backend (the real physics), which gives the best case for
an agent with perfect imagination. The trained model is on Hugging Face:
[`racoon25/jepa-as-a-tool`](https://huggingface.co/racoon25/jepa-as-a-tool) (`three-worlds/`).

## Results

The main run ([`notebooks/train_and_evaluate.ipynb`](notebooks/train_and_evaluate.ipynb)): one
world model trained from scratch on the three worlds, then the agent in those worlds and in a fourth
it never saw. 100 tasks per family, the same tasks in every setting, Muse Spark 1.3 (contributor
mode), 2,700 episodes. Success rate with its 95% interval; *p*: exact McNemar test, world model vs
alone, on the same tasks.

| World | Family | Alone | + World model | + Real physics | *p* |
|---|---|---|---|---|---|
| PushT | pose | 83% [74, 89] | 88% [80, 93] | 99% [95, 100] | 0.41 |
| PushT | **rotate** | 65% [55, 74] | **79% [70, 86]** | 96% [90, 98] | **0.04** |
| Tabletop (3D) | **pose** | 67% [57, 75] | **89% [81, 94]** | 97% [92, 99] | **< 0.001** |
| Tabletop (3D) | **rotate** | 57% [47, 66] | **72% [63, 80]** | 97% [92, 99] | **0.02** |
| Billiards (3D) | shot | 83% [74, 89] | 86% [78, 91] | 100% [96, 100] | 0.69 |
| Billiards (3D) | bank | 89% [81, 94] | 81% [72, 87] | 100% [96, 100] | 0.15 |
| Billiards (3D) | carom | 88% [80, 93] | 96% [90, 98] | 100% [96, 100] | 0.06 |
| **New room** (never seen) | pose | 70% [60, 78] | **11% [6, 19]** | 98% [93, 99] | **< 0.001** |
| **New room** (never seen) | rotate | 50% [40, 60] | **10% [6, 17]** | 97% [92, 99] | **< 0.001** |

- **The world model helps where it is accurate:** in the 3D tabletop (+22 and +15 points) and on
  PushT turns (+14).
- **Billiards:** the agent alone already wins 83–89%, and fast balls are harder to predict (11 px
  after one push): no significant gain.
- **A world model that misreads the room is worse than none.** The tabletop's physics, another
  table, light and camera: the model is off by 32 px after one push, more than predicting that
  nothing moves (21 px), and the agent, which trusts it, drops from 70% to 11%.

World model accuracy on held-out random pushes (median position error, px, on a 512 px table):

| World | 1 push | 2 pushes | 3 pushes | 4 pushes | Nothing moves, 4 pushes |
|---|---|---|---|---|---|
| PushT | 3.5 | 4.7 | 5.6 | 5.6 | 80 |
| Tabletop | 3.8 | 5.6 | 7.8 | 9.2 | 74 |
| Billiards | 11.3 | 20.3 | 27.3 | 32.7 | 177 |
| New room | 32.5 | 36.3 | 33.0 | 35.8 | 71 |

The translator alone reads one frame to 2.6 px (PushT), 2.0 px (tabletop) and 5.1 px (billiards).
The runs before the main one, and what each taught, are in
[`notebooks/experiments`](notebooks/experiments).

## Reproduce

```bash
uv sync && uv run pytest                    # install and test (CPU is enough)
cp .env.example .env                        # then set META_API_KEY
```

| Step | Command | Where, how long |
|---|---|---|
| Everything: data, V-JEPA 2 latents, translator, predictor, the 2,700 episodes, reports | [`notebooks/train_and_evaluate.ipynb`](notebooks/train_and_evaluate.ipynb), or `uv run jat pipeline --work main-run --size full --gpus 2 --provider meta` | Kaggle, 2 × T4, ~10 h, one session |
| The same stages at toy sizes, scripted agent: no GPU, no API | `uv run jat pipeline --work /tmp/dry --size small --provider dry` | CPU, minutes |
| Tables and paired tests of one world | `uv run jat report main-run/results/tabletop` | seconds |
| Website data | `just export-web --run pusht=main-run/results/pusht --run tabletop=main-run/results/tabletop --run billiards=main-run/results/billiards --run tabletop-new-look=main-run/results/tabletop-new-look --wm main=main-run/runs/wm`, then `just web` | seconds |

Interrupted runs resume: finished steps and episodes are skipped.

## Worlds

| World | Physics | Trained? |
|---|---|---|
| `pusht` | gym-pusht, 2D, no momentum | yes |
| `tabletop` | MuJoCo, 3D camera, the T has mass and slides | yes |
| `billiards` | MuJoCo, a white and a red ball, a cue; balls roll and bounce off the cushions | yes |
| `tabletop-new-look` | the tabletop's physics, another table, light, color and camera | no: evaluated with the tabletop's model |

One translator reads them all: a shared trunk on V-JEPA 2's latents, and one small head per world.
**Adding a world** is one folder and one line: see [docs/add-a-world.md](docs/add-a-world.md).

## Use it from any agent (MCP)

```bash
uv run jat mcp --world tabletop --wm hf:racoon25/jepa-as-a-tool/three-worlds
```

In Claude Desktop's `claude_desktop_config.json` (or any MCP client):

```json
{
  "mcpServers": {
    "jepa-tabletop": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/jepa-as-a-tool", "jat", "mcp", "--world", "tabletop", "--wm", "hf:racoon25/jepa-as-a-tool/three-worlds"]
    }
  }
}
```

Four tools: `start_task`, `simulate`, `act`, `done`. Each reply is the text our agent reads plus a
top view of the table with the imagined states.

## Repository

| Folder | Content |
|---|---|
| `src/jat/worlds` | one folder per world (physics, frames, rules, tasks); `tblock`: what the T worlds share; `table3d`: the MuJoCo table |
| `src/jat/wm` | data generator, V-JEPA 2 encoding, predictor, translator (trunk + heads), readouts, training, transfer test, ONNX export |
| `src/jat/tool` | the `simulate` tool, frame handles, oracle and world-model backends, top-view sketches |
| `src/jat/agent` | prompts, LLM client, agent loop, episode logs |
| `src/jat/eval` | evaluation grid, metrics, website export, learning from use |
| `src/jat/pipeline.py`, `mcp_server.py`, `clip.py`, `hub.py` | the main run, the MCP server, 3D videos, Hugging Face upload and download |
| `notebooks` | the main run (Kaggle); `experiments/`: the runs before it, and the open-model attempt |
| `web` | the website (Astro): episode player, dream mode |

## Built during the hackathon

Everything in this repository was written between Oct 27 and Nov 4, 2026: the push action on top
of `gym-pusht`, the task families, the `simulate` tool and agent loop, the data generator, the
V-JEPA 2 encoding, the predictor and translator, the evaluation and the website; then the three
MuJoCo worlds, the shared translator and transfer test, learning from use, the overnight pipeline,
the MCP server, dream mode and the 3D clips.

**Meta models used:** V-JEPA 2 (ViT-L) as the frozen encoder; Muse Spark 1.3 as the agent, through
the Meta Model API.

## License

MIT
