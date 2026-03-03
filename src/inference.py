# src/word_embeddings/inference.py

import os
import json
import numpy as np

from word_embeddings.utils.similarity import most_similar


class EmbeddingInference:

    def __init__(self, checkpoint_dir: str, model_name: str):
        self.W_in = np.load(os.path.join(checkpoint_dir, f"{model_name}_W_in.npy"))
        self.W_out = np.load(os.path.join(checkpoint_dir, f"{model_name}_W_out.npy"))

        with open(os.path.join(checkpoint_dir, f"{model_name}_vocab.json"), "r") as f:
            self.idx2word = json.load(f)

        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.embeddings = self.W_in + self.W_out

    def query(self, word: str, top_k: int = 5):
        if word not in self.word2idx:
            print("Word not in vocabulary.")
            return

        idx = self.word2idx[word]
        neighbors = most_similar(self.embeddings, idx, top_k=top_k)

        print(f"\nNearest neighbors for '{word}':")
        for i, sim in neighbors:
            print(f"  {self.idx2word[i]}  ({sim:.4f})")


if __name__ == "__main__":
    model = EmbeddingInference(
        checkpoint_dir="checkpoints",
        model_name="sgns",  # or "glove"
    )

    while True:
        word = input("\nEnter word (or 'exit'): ")
        if word == "exit":
            break
        model.query(word)