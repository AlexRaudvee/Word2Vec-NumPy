# src/evaluate_extrinsic.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from word_embeddings.data.tokenizer import simple_tokenize

def load_embeddings(checkpoint_dir: str, model_name: str):
    """
    Load W_in, W_out, word2idx from checkpoint_dir

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


def doc_embedding(text: str, embeddings: np.ndarray, word2idx: Dict[str, int]) -> np.ndarray:
    tokens = simple_tokenize(text)
    vecs = []
    for t in tokens:
        if t in word2idx:
            vecs.append(embeddings[word2idx[t]])
    if not vecs:
        # no known words; return zero vector
        return np.zeros(embeddings.shape[1], dtype=np.float32)
    return np.mean(vecs, axis=0)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    assert cfg.experiment.evaluation_type == "extrinsic"
    os.makedirs(cfg.paths.assets_dir, exist_ok=True)

    embeddings, idx2word, word2idx = load_embeddings(
        cfg.paths.checkpoint_dir, cfg.model.name
    )
    D = embeddings.shape[1]

    print(f"Loaded embeddings: V={embeddings.shape[0]}, D={D}")

    print(f"Loading AG News from {cfg.evaluation.ag_news_dir} ...")
    ag_news = load_from_disk(cfg.evaluation.ag_news_dir)

    def build_matrix(split_name: str):
        split = ag_news[split_name]
        texts = split["text"]
        labels = split["label"]
        X = np.zeros((len(texts), D), dtype=np.float32)
        y = np.array(labels, dtype=np.int64)
        for i, txt in enumerate(texts):
            X[i] = doc_embedding(txt, embeddings, word2idx)
        return X, y

    X_train, y_train = build_matrix("train")
    X_test, y_test = build_matrix("test")

    print("Training logistic regression classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    metrics = {
        "model_name": cfg.model.name,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    print("\nAG News extrinsic metrics:", metrics)

    out_path = Path(cfg.paths.assets_dir) / f"ag_news_extrinsic_{cfg.model.name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved extrinsic metrics to {out_path}")


if __name__ == "__main__":
    main()
