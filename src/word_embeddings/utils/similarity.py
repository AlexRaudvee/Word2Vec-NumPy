import numpy as np


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norms


def cosine_similarity_matrix(emb: np.ndarray) -> np.ndarray:
    """
    emb: (V, D)
    returns: (V, V)
    """
    emb_norm = normalize_rows(emb)
    return emb_norm @ emb_norm.T


def most_similar(
    emb: np.ndarray,
    idx: int,
    top_k: int = 10,
    exclude_self: bool = True,
) -> list[tuple[int, float]]:
    """
    Returns list of (index, similarity) sorted by similarity
    """
    emb_norm = normalize_rows(emb)
    sims = emb_norm @ emb_norm[idx]
    if exclude_self:
        sims[idx] = -np.inf
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in top_idx]