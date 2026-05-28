"""Simple two-layer neural network for poker action selection."""

import numpy as np

from .features import FEATURE_DIM

HIDDEN_DIM = 64
ACTION_DIM = 7  # FOLD, CHECK, CALL, RAISE_10, RAISE_25, RAISE_50, ALL_IN


def _he_init(fan_in: int, fan_out: int) -> np.ndarray:
    return np.random.randn(fan_in, fan_out).astype(np.float32) * np.sqrt(2.0 / fan_in)


class SmallNN:
    """Two-layer feed-forward network: Linear -> ReLU -> Linear.

    Weights are stored flat so they can be mutated directly by ES.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()

        self.W1 = _he_init(FEATURE_DIM, HIDDEN_DIM)
        self.b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.W2 = _he_init(HIDDEN_DIM, ACTION_DIM)
        self.b2 = np.zeros(ACTION_DIM, dtype=np.float32)

        # Record shapes so we can pack/unpack consistently
        self._shapes = {
            "W1": self.W1.shape,
            "b1": self.b1.shape,
            "W2": self.W2.shape,
            "b2": self.b2.shape,
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (FEATURE_DIM,) -> logits: (ACTION_DIM,)"""
        x = np.asarray(x, dtype=np.float32)
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return (h @ self.W2 + self.b2).astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Weight serialisation (for ES noise injection)
    # ------------------------------------------------------------------

    @property
    def num_params(self) -> int:
        return (
            self.W1.size + self.b1.size + self.W2.size + self.b2.size
        )

    def get_weights(self) -> np.ndarray:
        """Return a flat 1-D vector of all parameters."""
        return np.concatenate(
            [self.W1.ravel(), self.b1, self.W2.ravel(), self.b2]
        )

    def set_weights(self, flat: np.ndarray):
        """Restore parameters from a flat vector."""
        cursor = 0
        for name in ("W1", "b1", "W2", "b2"):
            shape = self._shapes[name]
            size = int(np.prod(shape))
            setattr(self, name, flat[cursor : cursor + size].reshape(shape).copy())
            cursor += size
