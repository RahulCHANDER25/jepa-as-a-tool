# Experiments

The runs that led to the main notebook, [`../train_and_evaluate.ipynb`](../train_and_evaluate.ipynb),
in the order they ran. Each one checks out commit `d825bff`, the last code that ran it: the main
pipeline has since replaced theirs.

| Notebook | What it did | What came out of it |
|---|---|---|
| `train_wm_v1.ipynb` | First world model on PushT: V-JEPA 2 latents, translator, predictor | The first version, replaced by v2 |
| `train_wm_v2.ipynb` | PushT world model v2 | 2.8 px and 1.5° median error one push ahead, on held-out random pushes |
| `eval_wm_v2.ipynb` | Muse Spark on PushT, alone / with v2 / with the real physics | Rotate tasks: 50% alone, 76% with the world model, 96% with the real physics |
| `train_four_worlds.ipynb` | Tabletop and billiards (MuJoCo), one translator for several worlds, the transfer test, learning from use | The shared translator reads the tabletop almost as well as its own (2.3 vs 1.9 px); two bugs (below) |
| `rebuild_billiards.ipynb` | Billiards on fixed physics, the 3D evaluations again, learning from use checked fairly | The agent's own pushes halve the PushT error one push ahead (3.2 → 1.6 px); a world model that misreads a new room makes the agent fail (68% → 6%) |

Bugs found along the way, each fixed with a test:
- **Billiards balls flew off the table**: a cue that kept pushing a ball held by a cushion. The cue
  now withdraws after the strike.
- **3D evaluations with the world model crashed**: MuJoCo frames drawn from several threads at once.
  Every frame is now drawn on one thread.
- **Carom tasks could not be found**: after the cue fix, the white ball rarely reaches a red ball
  that starts anywhere. It now starts close to it.
