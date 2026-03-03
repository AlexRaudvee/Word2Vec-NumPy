from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from ..models import EmbeddingParams


@dataclass
class AdagradState:
    W_in_sq: np.ndarray
    W_out_sq: np.ndarray
    b_in_sq: np.ndarray
    b_out_sq: np.ndarray


class AdagradOptimizer:
    def __init__(self, lr: float, epsilon: float = 1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.state: AdagradState | None = None

    def _init_state(self, params: EmbeddingParams) -> None:
        self.state = AdagradState(
            W_in_sq=np.zeros_like(params.W_in),
            W_out_sq=np.zeros_like(params.W_out),
            b_in_sq=np.zeros_like(params.b_in),
            b_out_sq=np.zeros_like(params.b_out),
        )

    def step(self, params: EmbeddingParams, grads: EmbeddingParams) -> None:
        """
        In-place parameter update.
        """
        if self.state is None:
            self._init_state(params)
        s = self.state

        # accumulate squared gradients
        s.W_in_sq += grads.W_in ** 2
        s.W_out_sq += grads.W_out ** 2
        s.b_in_sq += grads.b_in ** 2
        s.b_out_sq += grads.b_out ** 2

        # compute adjusted learning rates
        W_in_lr = self.lr / (np.sqrt(s.W_in_sq) + self.epsilon)
        W_out_lr = self.lr / (np.sqrt(s.W_out_sq) + self.epsilon)
        b_in_lr = self.lr / (np.sqrt(s.b_in_sq) + self.epsilon)
        b_out_lr = self.lr / (np.sqrt(s.b_out_sq) + self.epsilon)

        # apply parameter updates
        params.W_in -= W_in_lr * grads.W_in
        params.W_out -= W_out_lr * grads.W_out
        params.b_in -= b_in_lr * grads.b_in
        params.b_out -= b_out_lr * grads.b_out