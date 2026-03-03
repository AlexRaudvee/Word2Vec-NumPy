import numpy as np


def uniform_init(shape: tuple[int, ...], dim: int) -> np.ndarray:
    """
    Initialize embeddings with U(-0.5/dim, 0.5/dim)
    """
    limit = 0.5 / dim
    return np.random.uniform(-limit, limit, size=shape).astype(np.float32)