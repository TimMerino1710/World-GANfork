from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TensorBlock2VecStats:
    vocab_size: int
    counts: torch.Tensor  # (V,)
    discards: torch.Tensor  # (V,)


class TensorBlock2VecDataset(Dataset):
    """
    Word2Vec/Skip-gram style dataset over a 3D categorical volume.

    You provide a volume of integer token indices shaped (X, Y, Z).
    Each sample returns:
      - target: int token index
      - context: int64 array of neighboring token indices (cube neighborhood excluding center)
    """

    def __init__(self, volume_idx: torch.Tensor, *, neighbor_radius: int = 1, subsample_t: float = 1e-3):
        super().__init__()
        if volume_idx.ndim != 3:
            raise ValueError(f"volume_idx must be 3D (X,Y,Z), got {tuple(volume_idx.shape)}")
        if neighbor_radius < 1:
            raise ValueError("neighbor_radius must be >= 1")

        self.volume = volume_idx.to(torch.long).contiguous()
        self.X, self.Y, self.Z = map(int, self.volume.shape)
        self.r = int(neighbor_radius)

        # Valid coordinate range (exclude borders so neighborhood fits)
        self.x0, self.x1 = self.r, self.X - self.r
        self.y0, self.y1 = self.r, self.Y - self.r
        self.z0, self.z1 = self.r, self.Z - self.r
        if self.x1 <= self.x0 or self.y1 <= self.y0 or self.z1 <= self.z0:
            raise ValueError(
                f"Volume too small for neighbor_radius={self.r}: shape={tuple(self.volume.shape)}"
            )

        # Vocabulary size inferred from max index
        self.vocab_size = int(self.volume.max().item()) + 1

        # Precompute neighbor offsets (exclude center)
        offsets = []
        for dx, dy, dz in product(range(-self.r, self.r + 1), repeat=3):
            if dx == 0 and dy == 0 and dz == 0:
                continue
            offsets.append((dx, dy, dz))
        self._offsets = offsets

        # Frequency-based subsampling (word2vec-style)
        counts = torch.bincount(self.volume.view(-1), minlength=self.vocab_size).to(torch.float32)
        f = counts / counts.sum().clamp_min(1.0)
        # Original repo formula: 1 - (sqrt(f/t) + 1) * (t/f)
        t = float(subsample_t)
        discards = 1.0 - (torch.sqrt(f / t) + 1.0) * (t / f.clamp_min(1e-12))
        discards = discards.clamp(min=0.0, max=1.0)

        self.stats = TensorBlock2VecStats(vocab_size=self.vocab_size, counts=counts.to(torch.long), discards=discards)

    def __len__(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0) * (self.z1 - self.z0)

    def _idx_to_coords(self, index: int) -> Tuple[int, int, int]:
        # Map linear index -> (x,y,z) inside valid interior box
        z_span = (self.z1 - self.z0)
        y_span = (self.y1 - self.y0)
        x_span = (self.x1 - self.x0)

        z = index % z_span
        y = (index // z_span) % y_span
        x = (index // (z_span * y_span)) % x_span
        return x + self.x0, y + self.y0, z + self.z0

    def __getitem__(self, index: int):
        x, y, z = self._idx_to_coords(int(index))
        target = int(self.volume[x, y, z].item())

        # Subsample frequent tokens (resample a different coordinate)
        if np.random.rand() < float(self.stats.discards[target].item()):
            return self.__getitem__(np.random.randint(0, len(self)))

        ctx = []
        for dx, dy, dz in self._offsets:
            ctx.append(int(self.volume[x + dx, y + dy, z + dz].item()))

        # Return tensors compatible with SkipGramModel: (target: Long), (context: Long[context_size])
        return torch.tensor(target, dtype=torch.long), torch.tensor(ctx, dtype=torch.long)


