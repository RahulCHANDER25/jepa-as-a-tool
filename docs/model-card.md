---
license: mit
tags:
  - world-model
  - v-jepa-2
  - agents
  - tool-use
---

# jepa-as-a-tool: one world model for three worlds

The world model behind [jepa-as-a-tool](https://github.com/RahulCHANDER25/jepa-as-a-tool): an LLM
agent (Muse Spark) calls a `simulate` tool that imagines its pushes in this model before acting.

```
frame ──▶ V-JEPA 2 (frozen) ──▶ latent ──▶ predictor (+ push) ──▶ latent ──▶ translator ──▶ positions
```

`three-worlds/` holds the two trained parts. The encoder is Meta's frozen V-JEPA 2
(`facebook/vjepa2-vitl-fpc64-256`), downloaded separately.

| File | What it is |
|---|---|
| `predictor.pt` | 6-layer transformer: V-JEPA 2 latent + push → next latent. One model for PushT (2D), a 3D tabletop and 3D billiards; it is not told which world it is in |
| `translator.pt` | latent → positions: a shared trunk and one head per world |
| `*.json` | its errors on held-out random pushes |

Trained from scratch in one Kaggle session (2 × T4) by
[`notebooks/train_and_evaluate.ipynb`](https://github.com/RahulCHANDER25/jepa-as-a-tool/blob/dev/notebooks/train_and_evaluate.ipynb).

## Accuracy

Median position error (px, 512 px table) against the real physics, on held-out random pushes:

| World | 1 push | 2 pushes | 3 pushes | 4 pushes |
|---|---|---|---|---|
| PushT | 3.5 | 4.7 | 5.6 | 5.6 |
| Tabletop | 3.8 | 5.6 | 7.8 | 9.2 |
| Billiards | 11.3 | 20.3 | 27.3 | 32.7 |
| Tabletop, a room never seen | 32.5 | 36.3 | 33.0 | 35.8 |

## Use

```bash
git clone -b dev https://github.com/RahulCHANDER25/jepa-as-a-tool && cd jepa-as-a-tool && uv sync
uv run jat mcp --world tabletop --wm hf:racoon25/jepa-as-a-tool/three-worlds
```
