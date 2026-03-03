from __future__ import annotations
from collections import defaultdict
from typing import Iterable, Dict, Tuple, List

import numpy as np


def build_cooccurrence_triplets(
    sequences: Iterable[list[int]],
    max_window_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build co-occurrence matrix in COO style triplets (i, j, X_ij).

    No distance weighting: each context within window adds +1.

    Returns:
        row_indices: np.ndarray[int32]
        col_indices: np.ndarray[int32]
        values: np.ndarray[float32]
    """
    cooc: Dict[Tuple[int, int], float] = defaultdict(float)

    for seq in sequences:
        L = len(seq)
        for center_pos in range(L):
            center = seq[center_pos]
            window = max_window_size  # fixed window for GloVe (still OK)
            left = max(0, center_pos - window)
            right = min(L - 1, center_pos + window)
            for ctx_pos in range(left, right + 1):
                if ctx_pos == center_pos:
                    continue
                context = seq[ctx_pos]
                cooc[(center, context)] += 1.0

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for (i, j), v in cooc.items():
        rows.append(i)
        cols.append(j)
        vals.append(v)

    return (
        np.array(rows, dtype=np.int32),
        np.array(cols, dtype=np.int32),
        np.array(vals, dtype=np.float32),
    )