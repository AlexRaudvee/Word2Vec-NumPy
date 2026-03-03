from typing import Callable
import numpy as np


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Simple finite-difference gradient checker for small tensors
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + eps
        fx_plus = f(x)

        x[idx] = old_val - eps
        fx_minus = f(x)

        x[idx] = old_val
        grad[idx] = (fx_plus - fx_minus) / (2 * eps)
        it.iternext()
    return grad