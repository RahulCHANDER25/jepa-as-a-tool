"""A dataset folder (rollouts + latents) as torch tensors, ready for training and evaluation."""

import math
import warnings
from pathlib import Path

import numpy as np
import torch

from jat.wm.data import TRAIN, Rollouts
from jat.wm.encoder import load_latents
from jat.wm.readouts import readout
from jat.worlds.base import TABLE


def available_ram() -> float:
    """Free RAM plus reclaimable cache, in bytes (Linux MemAvailable); inf if unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return math.inf


class LatentData:
    def __init__(self, folder: Path, device: torch.device, max_train: int | None = None):
        """`max_train`: keep only the first N training episodes (label-budget experiments)."""
        self.folder = folder
        self.rollouts = Rollouts.load(folder)
        self.world = self.rollouts.world
        self.readout = readout(self.world)
        self.max_train = max_train
        latents, self.meta = load_latents(folder)
        self.device = device
        # In RAM when it fits; otherwise each batch reads the file (slower, still works).
        self.in_ram = latents.nbytes < 0.7 * available_ram()
        if self.in_ram:
            latents = np.array(latents)
        with warnings.catch_warnings():
            # A memmap is read-only; torch warns about writes we never do.
            warnings.simplefilter("ignore", UserWarning)
            z = torch.from_numpy(latents)  # fp16
        # On the GPU when it fits with room to spare: no host-to-device copy per batch.
        if device.type == "cuda" and z.nbytes < 0.5 * torch.cuda.mem_get_info(device)[0]:
            z = z.to(device)
        self.z = z  # [E, T+1, P, D]
        self.poses = torch.tensor(self.rollouts.poses, dtype=torch.float32, device=device)
        # Same scaling as Push.numeric()
        pushes = self.rollouts.pushes / (TABLE / 2) - 1
        self.pushes = torch.tensor(pushes, dtype=torch.float32, device=device)  # [E, T, 4]
        self.moved = torch.tensor(self.rollouts.moved, device=device)  # [E, T]

    @property
    def n_tokens(self) -> int:
        return self.z.shape[2]

    @property
    def dim(self) -> int:
        return self.z.shape[3]

    @property
    def n_steps(self) -> int:
        return self.rollouts.n_steps

    def episodes(self, split: int) -> torch.Tensor:
        index = np.flatnonzero(self.rollouts.split == split)
        if split == TRAIN and self.max_train is not None:
            index = index[: self.max_train]
        return torch.from_numpy(index).to(self.device)

    def latent(self, episodes: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        """float32 [B, P, D] on the device."""
        z = self.z[episodes.to(self.z.device), steps.to(self.z.device)]
        return z.to(self.device, non_blocking=True).float()
