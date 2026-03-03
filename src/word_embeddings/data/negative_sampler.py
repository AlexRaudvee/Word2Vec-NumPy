from __future__ import annotations
import numpy as np
from typing import Sequence


class NegativeSampler:
    """
    Draws negative samples according to unigram^power distribution.
    """

    def __init__(self, counts: Sequence[int], power: float = 0.75):
        counts = np.asarray(counts, dtype=np.float64)
        if counts.ndim != 1:
            raise ValueError("counts must be 1D")
        self.vocab_size = counts.shape[0]
        probs = counts ** power
        probs_sum = probs.sum()
        if probs_sum == 0:
            raise ValueError("All counts are zero.")
        self.probs = probs / probs_sum

    def sample(self, batch_size: int, num_negatives: int) -> np.ndarray:
        """
        Returns indices of shape (batch_size, num_negatives).
        """
        flat = np.random.choice(
            self.vocab_size,
            size=batch_size * num_negatives,
            replace=True,
            p=self.probs,
        )
        return flat.reshape(batch_size, num_negatives)