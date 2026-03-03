# src/word_embeddings/models/sgns_model.py
from __future__ import annotations

import numpy as np

from .base_embedding import BaseEmbeddingModel, EmbeddingParams


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable-ish sigmoid
    return 1.0 / (1.0 + np.exp(-x))


class SGNSModel(BaseEmbeddingModel):
    """
    Skip-gram with Negative Sampling (SGNS) model.

    Uses:
        - W_in for center words
        - W_out for context words
        - b_in, b_out are unused (kept for interface compatibility)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_negatives: int,
        seed: int | None = None,
    ):
        super().__init__(vocab_size=vocab_size, embedding_dim=embedding_dim, seed=seed)
        self.num_negatives = int(num_negatives)

    def loss_and_grad(
        self,
        center_indices: np.ndarray,   # (B,)
        pos_indices: np.ndarray,      # (B,)
        neg_indices: np.ndarray,      # (B, K)
    ) -> tuple[float, EmbeddingParams]:
        """
        Compute SGNS loss and gradients for a mini-batch.

        Loss for one (c, o, {n_k}) triple:
            L = -log sigmoid(v_o' . v_c) - sum_k log sigmoid(-v_nk' . v_c)

        We return:
            - mean loss over batch
            - gradients for all parameters (EmbeddingParams)
        """
        B = center_indices.shape[0]
        K = neg_indices.shape[1]

        v_c = self.get_input_vectors(center_indices)          # (B, D)
        v_pos = self.get_output_vectors(pos_indices)          # (B, D)

        #negatives
        neg_flat = neg_indices.reshape(-1)                    # (B*K,)
        v_neg = self.get_output_vectors(neg_flat)             # (B*K, D)
        v_neg = v_neg.reshape(B, K, self.embedding_dim)       # (B, K, D)

        #forward. scores
        #positive scores (B,)
        pos_scores = np.sum(v_c * v_pos, axis=1)

        # negative scores: (B, K) each negative score s_neg[b, k] = v_c[b] . v_neg[b, k] broadcast v_c over K
        neg_scores = np.sum(v_neg * v_c[:, None, :], axis=2)

        pos_sig = sigmoid(pos_scores)
        neg_sig = sigmoid(-neg_scores)
        neg_sig_for_grad = sigmoid(neg_scores)

        eps = 1e-8
        #loss
        loss_per_example = -np.log(pos_sig + eps) - np.sum(np.log(neg_sig + eps), axis=1)
        loss = float(np.mean(loss_per_example))

        #backward
        grads = self.zero_grad_like()

        #derivatives wrt scores
        dL_dpos = pos_sig - 1.0                    # (B,)
        dL_dneg = neg_sig_for_grad                 # (B, K)

        # gradients wrt vectors:
        # v_c: (B, D)
        grad_v_c = dL_dpos[:, None] * v_pos + np.sum(
            dL_dneg[:, :, None] * v_neg, axis=1
        )  # (B, D)

        # v_pos: (B, D)
        grad_v_pos = dL_dpos[:, None] * v_c       # (B, D)

        # v_neg: (B, K, D)
        grad_v_neg = dL_dneg[:, :, None] * v_c[:, None, :]   # (B, K, D)

        # accumulate into embedding parameter gradients
        np.add.at(grads.W_in, center_indices, grad_v_c)
        np.add.at(grads.W_out, pos_indices, grad_v_pos)
        np.add.at(grads.W_out, neg_flat, grad_v_neg.reshape(B * K, self.embedding_dim))

        # normalize grads by batch size for stability
        grads.W_in /= B
        grads.W_out /= B
        
        return loss, grads