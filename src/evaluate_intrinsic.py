# src/evaluate_intrinsic.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from word_embeddings.utils import most_similar


def load_embeddings(checkpoint_dir: str, model_name: str):
    """
    Load W_in, W_out, vocab from checkpoint_dir

    Returns:
        embeddings: np.ndarray of shape (V, D)
        idx2word: list[str]
        word2idx: dict[str, int]
    """
    checkpoint_dir = Path(checkpoint_dir)
    W_in = np.load(checkpoint_dir / f"{model_name}_W_in.npy")
    W_out = np.load(checkpoint_dir / f"{model_name}_W_out.npy")

    with open(checkpoint_dir / f"{model_name}_vocab.json", "r") as f:
        idx2word = json.load(f)

    word2idx = {w: i for i, w in enumerate(idx2word)}
    embeddings = W_in + W_out
    return embeddings, idx2word, word2idx


# MEN similarity 


def eval_men(
    embeddings: np.ndarray,
    word2idx: Dict[str, int],
    men_path: str,
) -> Dict[str, float]:
    """
    Evaluate word similarity on MEN dataset.

    MEN format (simplified):
        word1 word2 score
    """
    sims_pred: List[float] = []
    sims_gold: List[float] = []
    total_pairs = 0
    used_pairs = 0

    with open(men_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            w1, w2 = parts[0], parts[1]
            try:
                score = float(parts[2])
            except ValueError:
                continue

            total_pairs += 1
            if w1 not in word2idx or w2 not in word2idx:
                continue

            i1 = word2idx[w1]
            i2 = word2idx[w2]
            v1 = embeddings[i1]
            v2 = embeddings[i2]

            cos = float(
                np.dot(v1, v2)
                / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            )

            sims_pred.append(cos)
            sims_gold.append(score)
            used_pairs += 1

    result: Dict[str, float] = {"total_pairs": total_pairs, "used_pairs": used_pairs}
    if len(sims_pred) > 1:
        corr, _ = spearmanr(sims_pred, sims_gold)
        result["spearman"] = float(corr)
    else:
        result["spearman"] = float("nan")

    return result


# MSR analogies


def eval_msr(
    embeddings: np.ndarray,
    word2idx: Dict[str, int],
    idx2word: List[str],
    msr_path: str,
    top_k: int = 1,
) -> Dict[str, float]:
    """
    Evaluate analogies on MSR dataset

    We assume a format like:
        a b c PATTERN d
    and we treat a,b,c,d as words: (a,b,c,d) = (0,1,2,last).
    """

    V, D = embeddings.shape
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    emb_normed = embeddings / norms

    total = 0
    used = 0
    correct = 0

    with open(msr_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            a, b, c, d = parts[0], parts[1], parts[2], parts[-1]
            total += 1

            if any(w not in word2idx for w in (a, b, c, d)):
                continue

            ia, ib, ic, id_true = (
                word2idx[a],
                word2idx[b],
                word2idx[c],
                word2idx[d],
            )

            # analogy vector
            vec = embeddings[ib] - embeddings[ia] + embeddings[ic]
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)

            # cosine similarity against all words
            sims = emb_normed @ vec_norm  # (V,)

            # exclude source words
            sims[ia] = -1e9
            sims[ib] = -1e9
            sims[ic] = -1e9

            # top-k predictions
            pred_indices = np.argpartition(-sims, top_k)[:top_k]
            used += 1
            if id_true in pred_indices:
                correct += 1

    acc = correct / used if used > 0 else float("nan")
    return {
        "total_analogies": total,
        "used_analogies": used,
        "top_k": top_k,
        "accuracy": float(acc),
    }


# Nearest Neighbors


def eval_neighbors(
    embeddings: np.ndarray,
    word2idx: Dict[str, int],
    idx2word: List[str],
    top_k: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Larger anchor set for nearest-neighbor coherence inspection.
    """
    anchors = [
        # gender / people
        "man",
        "woman",
        "king",
        "queen",
        "boy",
        "girl",
        # geography
        "france",
        "germany",
        "china",
        "japan",
        "europe",
        "america",
        # tech
        "computer",
        "software",
        "internet",
        "data",
        "network",
        # abstract
        "war",
        "peace",
        "music",
        "science",
        "art",
    ]

    results: Dict[str, List[Tuple[str, float]]] = {}

    for word in anchors:
        if word not in word2idx:
            continue
        idx = word2idx[word]
        neighbors = most_similar(embeddings, idx, top_k=top_k)
        results[word] = [
            (idx2word[i], float(sim)) for i, sim in neighbors
        ]

    return results

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    assert cfg.experiment.evaluation_type == "intrinsic"

    os.makedirs(cfg.paths.assets_dir, exist_ok=True)

    embeddings, idx2word, word2idx = load_embeddings(
        cfg.paths.checkpoint_dir, cfg.model.name
    )

    # MEN
    men_result = eval_men(embeddings, word2idx, cfg.evaluation.men_path)
    print("\nMEN similarity:", men_result)

    # MSR
    msr_result = eval_msr(embeddings, word2idx, idx2word, cfg.evaluation.msr_path)
    print("\nMSR analogy:", msr_result)

    # nearest neighbors
    nn_result = eval_neighbors(embeddings, word2idx, idx2word)

    # save metrics
    metrics = {
        "model_name": cfg.model.name,
        "men": men_result,
        "msr": msr_result,
        "neighbors": nn_result,
    }

    out_path = Path(cfg.paths.assets_dir) / f"intrinsic_{cfg.model.name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved intrinsic metrics to {out_path}")


if __name__ == "__main__":
    main()
