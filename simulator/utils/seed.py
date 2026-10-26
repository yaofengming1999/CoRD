from typing import Optional
import numpy as np


class SeedManager:
    """Central RNG manager so all randomness is reproducible."""

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self._rng = np.random.default_rng(base_seed)

    def reset(self, seed: Optional[int] = None):
        s = seed if seed is not None else self.base_seed
        self._rng = np.random.default_rng(s)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def spawn(self) -> np.random.Generator:
        """Create a child RNG with a derived seed."""
        child_seed = int(self._rng.integers(0, 2**32))
        return np.random.default_rng(child_seed)
