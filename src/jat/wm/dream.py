"""Dream mode: the world model in a web browser. V-JEPA 2 is too big to run there, so the
latents of a few start states are computed here; the predictor and the translator (with the
world's head and readout built in) go to ONNX for onnxruntime-web.

A folder per world:
  predictor.onnx   (latent [1, P, D], push [1, 4] in [-1, 1]) -> next latent; int8 weights
  reader.onnx      latent [1, P, D] -> world state [1, S] (T: x, y, theta; balls: x, y each)
  starts.bin       float16 [N, P, D]: V-JEPA 2 latents of the start states
  dream.json       world, readout, shapes, the start states, and the check against PyTorch
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from jat.wm.models import WorldModel
from jat.worlds.base import TABLE


class _Reader(nn.Module):
    """Translator head + readout decoding: latent -> world state."""

    def __init__(self, world_model: WorldModel):
        super().__init__()
        self.translator, self.head, self.readout = (
            world_model.translator,
            world_model.head,
            world_model.readout,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.readout.decode(self.translator(z, self.head))


def _export(
    module: nn.Module, inputs: tuple, names: list[str], output: str, path: Path, int8: bool
) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    full = path.with_suffix(".fp32.onnx") if int8 else path
    # The TorchScript exporter: its graph quantizes cleanly (the dynamo one fails shape checks)
    torch.onnx.export(
        module, inputs, full, input_names=names, output_names=[output], dynamo=False,
        opset_version=17,
    )  # fmt: skip
    if int8:
        quantize_dynamic(full, path, weight_type=QuantType.QInt8, per_channel=True)
        full.unlink()


def export_dream(world_model: WorldModel, starts: list, out: Path, pushes: int = 3) -> dict:
    """Exports, then replays `pushes` random aimed pushes from every start in PyTorch and in
    onnxruntime: the readings must agree, or the browser would dream something else."""
    import onnxruntime as ort

    out.mkdir(parents=True, exist_ok=True)
    world = world_model.world
    z0 = world_model.encode(starts).float()  # [N, P, D]
    n_tokens, dim = z0.shape[1:]
    sample_push = torch.zeros(1, 4)
    # int8 predictor (46 -> 12 MB); the reader stays fp32 (5 MB): small, and its error shows directly
    _export(world_model.predictor.eval(), (z0[:1], sample_push), ["z", "push"], "z_next",
            out / "predictor.onnx", int8=True)  # fmt: skip
    _export(_Reader(world_model).eval(), (z0[:1],), ["z"], "state", out / "reader.onnx", False)
    z0.half().numpy().tofile(out / "starts.bin")

    predictor = ort.InferenceSession(out / "predictor.onnx")
    reader = ort.InferenceSession(out / "reader.onnx")
    rng = np.random.default_rng(0)
    gaps = []
    for i, start in enumerate(starts):
        state, z_t, z_o = np.asarray(start), z0[i : i + 1], z0[i : i + 1].numpy()
        for _ in range(pushes):
            push = world.aimed_push(state, rng)
            action = torch.tensor(push.numeric(), dtype=torch.float32)[None]
            z_t = world_model.step(z_t, action.numpy())
            z_o = predictor.run(None, {"z": z_o, "push": action.numpy()})[0]
            torch_state = world_model.read(z_t)[0]
            onnx_state = reader.run(None, {"z": z_o})[0][0]
            gaps.append(np.abs(torch_state - onnx_state)[:2].max())  # px of the first point
            state = torch_state
    manifest = {
        "world": world.name,
        "readout": world.readout,
        "table": TABLE,
        "tokens": int(n_tokens),
        "dim": int(dim),
        "starts": [np.asarray(s).tolist() for s in starts],
        "check": {
            "pushes": len(gaps),
            "median_px_gap": float(np.median(gaps)),
            "max_px_gap": float(max(gaps)),
        },
    }
    (out / "dream.json").write_text(json.dumps(manifest, indent=1))
    return manifest
