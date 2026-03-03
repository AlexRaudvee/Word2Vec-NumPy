from __future__ import annotations
import numpy as np
from typing import Iterable, List, Tuple


class SkipGramDataset:
    """
    Generates (center, context) pairs with dynamic window size.

    No subsampling; dynamic window per center:
    R is Uniform(1, max_window_size).
    """

    def __init__(
        self,
        sequences: Iterable[list[int]],
        max_window_size: int,
    ):
        self.max_window_size = max_window_size
        self._offsets: List[int] = []
        self._lengths: List[int] = []
        flat: List[int] = []
        for seq in sequences:
            self._offsets.append(len(flat))
            self._lengths.append(len(seq))
            flat.extend(seq)
        self.tokens = np.array(flat, dtype=np.int32)
        self.length = len(self.tokens)

    def iter_pairs(self) -> Iterable[Tuple[int, int]]:
        """
        Simple generator over all (center, context) pairs.
        """
        tokens = self.tokens
        num_sequences = len(self._offsets)

        for seq_idx in range(num_sequences):
            start = self._offsets[seq_idx]
            seq_len = self._lengths[seq_idx]
            end = start + seq_len

            for i in range(start, end):
                window = np.random.randint(1, self.max_window_size + 1)
                left = max(start, i - window)
                right = min(end - 1, i + window)
                for j in range(left, right + 1):
                    if j == i:
                        continue
                    yield tokens[i], tokens[j]
