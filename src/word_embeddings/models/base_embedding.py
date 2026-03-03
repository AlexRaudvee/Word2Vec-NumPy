from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from ..utils import uniform_init


@dataclass
class EmbeddingParams:
    W_in: np.ndarray       # (V, D)
    W_out: np.ndarray      # (V, D)
    b_in: np.ndarray       # (V,)
    b_out: np.ndarray      # (V,)


class BaseEmbeddingModel:
    """
    Shared base for SGNS and GloVe.
    Holds parameters and common utilities.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.params = EmbeddingParams(
            W_in=uniform_init((vocab_size, embedding_dim), dim=embedding_dim),
            W_out=uniform_init((vocab_size, embedding_dim), dim=embedding_dim),
            b_in=np.zeros(vocab_size, dtype=np.float32),
            b_out=np.zeros(vocab_size, dtype=np.float32),
        )

    def get_input_vectors(self, indices: np.ndarray) -> np.ndarray:
        """
        indices: (B,)
        return: (B, D)
        """
        return self.params.W_in[indices]

    def get_output_vectors(self, indices: np.ndarray) -> np.ndarray:
        """
        indices: (B,)
        return: (B, D)
        """
        return self.params.W_out[indices]

    def combined_embeddings(self) -> np.ndarray:
        """
        For evaluation: W_in + W_out
        returns: (V, D)
        """
        return self.params.W_in + self.params.W_out

    def zero_grad_like(self) -> EmbeddingParams:
        return EmbeddingParams(
            W_in=np.zeros_like(self.params.W_in),
            W_out=np.zeros_like(self.params.W_out),
            b_in=np.zeros_like(self.params.b_in),
            b_out=np.zeros_like(self.params.b_out),
        )