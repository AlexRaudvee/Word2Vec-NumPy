import json
import os
from pathlib import Path
from typing import Dict, List

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


def build_toy_clusters() -> Dict[str, List[str]]:
    return {
        "gender_family": [
            "man",
            "woman",
            "men",
            "women",
            "boy",
            "girl",
            "king",
            "queen",
            "father",
            "mother",
        ],
        "numbers": [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten"
        ],
        "countries": [
            "france",
            "germany",
            "spain",
            "italy",
            "china",
            "japan",
            "india",
            "america",
        ],
        "tech": [
            "computer",
            "software",
            "internet",
            "data",
            "network",
            "code",
            "server",
            "system",
        ],
        "sports": [
            "football",
            "soccer",
            "basketball",
            "baseball",
            "tennis",
            "team",
            "game",
            "player",
        ],
    }


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    assert cfg.experiment.evaluation_type == "visualization"
    os.makedirs(cfg.paths.assets_dir, exist_ok=True)

    embeddings, idx2word, word2idx = load_embeddings(
        cfg.paths.checkpoint_dir, cfg.model.name
    )

    toy_clusters = build_toy_clusters()
    cluster_words: Dict[str, List[str]] = {}
    ordered_words: List[str] = []
    for cluster_name, words in toy_clusters.items():
        kept = [w for w in words if w in word2idx]
        if len(kept) >= 2:
            cluster_words[cluster_name] = kept
            ordered_words.extend(kept)

    if not ordered_words:
        raise ValueError("No toy words from predefined clusters were found in the model vocab.")

    indices = np.array([word2idx[w] for w in ordered_words], dtype=np.int64)
    emb_sub = embeddings[indices]
    D = embeddings.shape[1]

    print(f"Running PCA on {emb_sub.shape[0]} words, dim={D} -> 2D...")
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb_sub)

    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(cluster_words))))

    offset = 0
    for color_i, (cluster_name, words) in enumerate(cluster_words.items()):
        size = len(words)
        xs = emb_2d[offset : offset + size, 0]
        ys = emb_2d[offset : offset + size, 1]
        plt.scatter(xs, ys, alpha=0.8, s=40, color=colors[color_i], label=cluster_name)

        for j, word in enumerate(words):
            plt.text(xs[j] + 0.02, ys[j] + 0.02, word, fontsize=8, alpha=0.9)
        offset += size

    plt.title(f"PCA visualization ({cfg.model.name})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", fontsize=8)
    out_path = Path(cfg.paths.assets_dir) / f"pca_{cfg.model.name}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA plot to {out_path}")

    words_out_path = Path(cfg.paths.assets_dir) / f"pca_clusters_{cfg.model.name}.json"
    with words_out_path.open("w", encoding="utf-8") as f:
        json.dump(cluster_words, f, indent=2)
    print(f"Saved plotted toy clusters to {words_out_path}")


if __name__ == "__main__":
    main()
