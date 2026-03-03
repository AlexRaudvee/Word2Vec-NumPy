# src/evaluate_visualization.py

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA


def load_embeddings(checkpoint_dir: str, model_name: str):
    checkpoint_dir = Path(checkpoint_dir)
    W_in = np.load(checkpoint_dir / f"{model_name}_W_in.npy")
    W_out = np.load(checkpoint_dir / f"{model_name}_W_out.npy")

    with open(checkpoint_dir / f"{model_name}_vocab.json", "r") as f:
        idx2word = json.load(f)

    word2idx = {w: i for i, w in enumerate(idx2word)}
    embeddings = W_in + W_out
    return embeddings, idx2word, word2idx


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    assert cfg.experiment.evaluation_type == "visualization"
    os.makedirs(cfg.paths.assets_dir, exist_ok=True)

    embeddings, idx2word, word2idx = load_embeddings(
        cfg.paths.checkpoint_dir, cfg.model.name
    )

    V, D = embeddings.shape
    n = min(int(cfg.evaluation.pca_num_words), V)
    # Skip index 0 (<unk>), take the next n words
    indices = np.arange(1, n)
    emb_sub = embeddings[indices]

    print(f"Running PCA on {emb_sub.shape[0]} words, dim={D} -> 2D...")
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb_sub)

    plt.figure(figsize=(10, 10))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5, s=10)

    # optionally label a small subset
    for i, idx in enumerate(indices[:100]):  # label first 100
        word = idx2word[idx]
        x, y = emb_2d[i]
        plt.text(x + 0.02, y + 0.02, word, fontsize=8, alpha=0.7)

    plt.title(f"PCA visualization ({cfg.model.name})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    out_path = Path(cfg.paths.assets_dir) / f"pca_{cfg.model.name}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA plot to {out_path}")


if __name__ == "__main__":
    main()
