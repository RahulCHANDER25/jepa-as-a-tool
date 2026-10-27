"""A dataset folder (rollouts + latents) as torch tensors, ready for training and evaluation."""

import math
import warnings
from pathlib import Path

import numpy as np
import torch

from jat.env import geometry as geo
from jat.wm.data import Rollouts
from jat.wm.encoder import load_latents


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
    def __init__(self, folder: Path, device: torch.device):
        self.rollouts = Rollouts.load(folder)
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
        pushes = self.rollouts.pushes / (geo.TABLE / 2) - 1
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
        return torch.from_numpy(np.flatnonzero(self.rollouts.split == split)).to(self.device)

    def latent(self, episodes: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        """float32 [B, P, D] on the device."""
        z = self.z[episodes.to(self.z.device), steps.to(self.z.device)]
        return z.to(self.device, non_blocking=True).float()
