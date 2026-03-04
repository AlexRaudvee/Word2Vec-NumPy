# src/word_embeddings/models/glove_model.py
from __future__ import annotations

import numpy as np

from .base_embedding import BaseEmbeddingModel, EmbeddingParams


class GloVeModel(BaseEmbeddingModel):
    """
    GloVe model, using the same BaseEmbeddingModel parameters:
        W_in, W_out, b_in, b_out

    Objective per co-occurrence (i, j, X_ij):
        J_ij = f(X_ij) * (w_i^T w'_j + b_i + b'_j - log X_ij)^2

    We will use:
        f(x) = min(1, (x / xmax)^alpha)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        xmax: float = 100.0,
        alpha: float = 0.75,
        seed: int | None = None,
    ):
        super().__init__(vocab_size=vocab_size, embedding_dim=embedding_dim, seed=seed)
        self.xmax = float(xmax)
        self.alpha = float(alpha)

    def loss_and_grad_cooc(
        self,
        row_indices: np.ndarray,  # (B,)
        col_indices: np.ndarray,  # (B,)
        x_ij: np.ndarray,         # (B,)
    ) -> tuple[float, EmbeddingParams]:
        """
        Compute GloVe loss and gradients for a mini-batch of COO triplets.

        Inputs:
            row_indices: i indices
            col_indices: j indices
            x_ij: co-occurrence counts X_ij

        Returns:
            mean loss over batch
            gradients (EmbeddingParams)
        """
        B = row_indices.shape[0]

        #gather parameters
        W_in = self.params.W_in
        W_out = self.params.W_out
        b_in = self.params.b_in
        b_out = self.params.b_out

        #embeddings for batch
        w_i = W_in[row_indices]   # (B, D)
        w_j = W_out[col_indices]  # (B, D)

        #biases for batch
        b_i = b_in[row_indices]   # (B,)
        b_j = b_out[col_indices]  # (B,)

        # weights f(x)
        x = x_ij.astype(np.float32)  # (B,)
        f_x = np.where(
            x < self.xmax,
            (x / self.xmax) ** self.alpha,
            1.0,
        )  # (B,)
        # print(f"f_x: {f_x}")

        # predictions: dot + biases
        dots = np.sum(w_i * w_j, axis=1)  # (B,)
        log_x = np.log(x + 1)          # (B,)
        diff = dots + b_i + b_j - log_x   # (B,)
        # print(f"diff: {diff}")

        #loss: J = sum f(x) * diff^2
        loss_per = f_x * (diff ** 2)      # (B,)
        loss = float(np.mean(loss_per))

        #gradient wrt diff: g = dJ/d(diff) = 2 f(x) diff
        g = 2.0 * f_x * diff         # (B,) normalized by batch

        #gradients container
        grads = self.zero_grad_like()

        # for embeddings:
        # dJ/dw_i = g * w_j
        # dJ/dw_j = g * w_i
        grad_w_i = g[:, None] * w_j       # (B, D)
        grad_w_j = g[:, None] * w_i       # (B, D)

        #forbiases:
        # dJ/db_i = g
        # dJ/db_j = g
        grad_b_i = g                      # (B,)
        grad_b_j = g                      # (B,)

        #accumulate with add.at because indices repeat
        np.add.at(grads.W_in, row_indices, grad_w_i)
        np.add.at(grads.W_out, col_indices, grad_w_j)
        np.add.at(grads.b_in, row_indices, grad_b_i)
        np.add.at(grads.b_out, col_indices, grad_b_j)

        return loss, grads