# jepa-as-a-tool

**An LLM agent that imagines its moves in a V-JEPA 2 world model before touching the real table.**

Entry to the Meta Global AI Developer Hackathon (Oct 27 – Nov 4, 2026).
Website with live episodes: https://rahulchander25.github.io/jepa-as-a-tool/

Muse Spark plays PushT: push a T-shaped block to a goal pose, one straight push at a time. Before each
real push it can call one tool, `simulate(from_frame, plans)`, which imagines candidate push
sequences in a learned world model and says, in text, where the T would end up. The world model
never saw the physics engine; it learned from images only.

## How it works

```
frame ──▶ V-JEPA 2 (frozen) ──▶ latent ──▶ predictor (+ push) ──▶ latent ──▶ translator ──▶ "T at (212, 180), 35°"
                                              ▲ chains of pushes stay in latent space
```

| Part | What it is | Trained here? |
|---|---|---|
| **Encoder** | Meta's V-JEPA 2 (ViT-L, `facebook/vjepa2-vitl-fpc64-256`): a 256 px frame becomes 64 tokens × 1,024 | no, frozen |
| **Predictor** | a 6-layer transformer: latent + push → next latent | yes |
| **Translator** | latent → pose of the T; the only thing the LLM reads | yes |
| **Tool** | `simulate`: several plans per call, an id for every imagined state (`I1`, `I2`, …) so the agent can continue a branch | — |
| **Agent** | Muse Spark 1.3 on the Meta Model API, with LangChain tools and our own loop | no |

The same tool also runs on an **oracle** backend (the real PushT physics), which gives the best case
for an agent with perfect imagination.

## Results

150 tasks (3 families × 50), the same tasks in every setting, Muse Spark 1.3 (contributor mode,
temperature 0.7), world model v2. Success rate with its 95% interval:

| Family | Alone | + World model | + Real physics |
|---|---|---|---|
| Move (`translate`) | 92% [81, 97] | 94% [84, 98] | 100% [93, 100] |
| **Turn (`rotate`)** | **50% [37, 63]** | **76% [63, 86]** | 96% [87, 99] |
| Move and turn (`pose`) | 84% [71, 92] | 74% [60, 84] | 98% [90, 100] |

- **Turn:** on the same tasks the world model wins 17 that the agent alone loses and loses 4
  (exact McNemar test, p = 0.007). It recovers more than half (57%) of the oracle's gain.
- **Move:** easy without help, as expected.
- **Move and turn:** the world model hurts (4 won, 9 lost, p = 0.27). With the oracle the agent
  reaches 98%, so the weak point is the predictions: these tasks need chains of three pushes and the
  errors add up. After one imagined push 83% of the predictions are within 20 px and 15°; after
  three, 49%.
- The agent calls the tool in every episode, imagining 50–77 pushes per task, at about 3.5× the
  tokens of the agent alone.

World model accuracy on held-out random pushes (median error): 2.8 px and 1.5° after one push,
4.2 px and 2.5° after three. On the pushes the agent actually picks, the error is about twice
as large; near misses, where the push slides past the T, are the worst case.

## Reproduce

```bash
uv sync && uv run pytest                    # install and test (CPU is enough)
cp .env.example .env                        # then set META_API_KEY
```

| Step | Command | Where, how long |
|---|---|---|
| Train the world model | [`notebooks/train_wm_v2.ipynb`](notebooks/train_wm_v2.ipynb) (`jat gen`, `jat encode`, `jat train translator`, `jat train predictor`) | GPU ≥ 15 GB, ~4–5 h |
| World-model accuracy | `uv run jat wm-eval data/random-12k --wm runs/wm-v2` | GPU, minutes |
| Agent evaluation | [`notebooks/eval_wm_v2.ipynb`](notebooks/eval_wm_v2.ipynb), or `META_MODEL=muse-spark-1.3-contributor uv run jat eval --out results/eval-wm-v2 --families translate,rotate,pose --n 50 --settings no_tool,wm_tool,oracle_tool --wm runs/wm-v2` | CPU, ~2–3 h, ~$5–9 |
| Tables and paired tests | `uv run jat report results/eval-wm-v2` | seconds |
| Website data | `just export-web results/eval-wm-v2`, then `just web` | seconds |

Interrupted evaluations resume: finished episodes are skipped.

**All four worlds in one run:** [`notebooks/overnight.ipynb`](notebooks/overnight.ipynb) (Kaggle,
2 × T4, ~8–9 h) trains the tabletop and billiards world models, the shared translator and the
transfer test, fine-tunes PushT on the agent's own imagined pushes, and runs every evaluation.
`uv run jat pipeline --work /tmp/dry --size small --provider dry --wm-v2 runs/wm-v2 --eval-v2
results/eval-wm-v2` runs the same stages at toy sizes on a CPU, with a scripted agent: no GPU, no API.

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
uv run jat mcp --world billiards --wm runs/billiards     # or --wm hf:<owner>/<repo>/billiards
```

In Claude Desktop's `claude_desktop_config.json` (or any MCP client):

```json
{
  "mcpServers": {
    "jepa-billiards": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/jepa-as-a-tool", "jat", "mcp", "--world", "billiards", "--wm", "runs/billiards"]
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
| `src/jat/pipeline.py`, `mcp_server.py`, `clip.py`, `hub.py` | the overnight run, the MCP server, 3D videos, Hugging Face upload and download |
| `notebooks` | GPU training and the evaluations, for Kaggle or Colab |
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
