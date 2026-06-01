from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class MemoryConfig:
    max_memory: int = 3000
    threshold_quantile: float = 0.995
    threshold_scale: float = 1.25
    random_seed: int = 7


class NormalMemory:
    """Small PatchCore-style memory bank for normal-only anomaly scoring."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.memory: np.ndarray | None = None
        self.threshold: float | None = None
        self._nn: NearestNeighbors | None = None

    def fit(self, x: np.ndarray) -> "NormalMemory":
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or len(x) < 4:
            raise ValueError(f"Need at least four feature vectors, got {x.shape}")

        self.center = np.median(x, axis=0)
        q25, q75 = np.percentile(x, [25, 75], axis=0)
        self.scale = (q75 - q25).astype(np.float32)
        self.scale[self.scale < 1e-6] = 1.0
        z = self._standardize(x)

        train_nn = NearestNeighbors(n_neighbors=min(2, len(z)), algorithm="auto")
        train_nn.fit(z)
        distances, _ = train_nn.kneighbors(z)
        train_scores = distances[:, 1] if distances.shape[1] > 1 else distances[:, 0]
        threshold = np.quantile(train_scores, self.config.threshold_quantile)
        self.threshold = float(max(threshold * self.config.threshold_scale, 1e-6))

        if len(z) > self.config.max_memory:
            rng = np.random.default_rng(self.config.random_seed)
            idx = rng.choice(len(z), size=self.config.max_memory, replace=False)
            z = z[np.sort(idx)]
        self.memory = z.astype(np.float32)
        self._rebuild_index()
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self._nn is None:
            raise RuntimeError("NormalMemory is not fitted")
        z = self._standardize(np.asarray(x, dtype=np.float32))
        distances, _ = self._nn.kneighbors(z)
        return distances[:, 0]

    def is_anomaly(self, x: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("NormalMemory is not fitted")
        return self.score(x) > self.threshold

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None:
            raise RuntimeError("NormalMemory is not fitted")
        return (x - self.center) / self.scale

    def _rebuild_index(self) -> None:
        if self.memory is None:
            return
        self._nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        self._nn.fit(self.memory)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_nn"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._rebuild_index()

