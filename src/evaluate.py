# src/word_embeddings/evaluate.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from word_embeddings.utils import most_similar
from word_embeddings.training.trainer import TrainingMetrics


def evaluate_model(model, vocab, metrics: TrainingMetrics):

    print("\n===== Evaluation =====")
    losses = metrics.losses_logged
    memory_steps = metrics.memory_steps
    memory_usage_mb = metrics.memory_usage_mb
    epoch_times_sec = metrics.epoch_times_sec

    # Loss Curve
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Logged Step")
    plt.ylabel("Loss")
    plt.savefig(f"assets/training_loss_{model.__class__.__name__}.png")
    plt.close()

    # Memory Curve
    if len(memory_steps) > 0 and len(memory_usage_mb) > 0:
        mem_steps = np.asarray(memory_steps, dtype=np.int32)
        mem_usage = np.asarray(memory_usage_mb, dtype=np.float64)
        valid_mask = np.isfinite(mem_usage)

        if np.any(valid_mask):
            plt.figure()
            plt.plot(mem_steps[valid_mask], mem_usage[valid_mask], linewidth=1.2)
            plt.title("Memory Usage During Training")
            plt.xlabel("Training Step")
            plt.ylabel("RSS Memory (MB)")
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.savefig(f"assets/training_memory_{model.__class__.__name__}.png")
            plt.close()

    # Epoch Time Curve
    if len(epoch_times_sec) > 0:
        epochs = np.arange(1, len(epoch_times_sec) + 1)
        plt.figure()
        plt.plot(epochs, epoch_times_sec, marker="o")
        plt.title("Training Time Per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.savefig(f"assets/training_time_per_epoch_{model.__class__.__name__}.png")
        plt.close()

    # Nearest Neighbors
    embeddings = model.combined_embeddings()

    test_words = ["the", "man", "woman", "king", "computer", "valkyria"]

    for word in test_words:
        if word not in vocab.word2idx:
            continue

        idx = vocab.word2idx[word]
        neighbors = most_similar(embeddings, idx, top_k=5)

        print(f"\nNearest neighbors for '{word}':")
        for i, sim in neighbors:
            print(f"  {vocab.decode_index(i)}  ({sim:.4f})")

    # Example toy similarity pairs for checking 
    similarity_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("computer", "internet"),
    ]

    sims = []
    gold = [0.7, 0.7, 0.5]  # dummy target

    for w1, w2 in similarity_pairs:
        if w1 in vocab.word2idx and w2 in vocab.word2idx:
            i1 = vocab.word2idx[w1]
            i2 = vocab.word2idx[w2]
            v1 = embeddings[i1]
            v2 = embeddings[i2]
            cos = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
            )
            sims.append(cos)

    if len(sims) == len(gold):
        corr, _ = spearmanr(sims, gold)
        print(f"\nSpearman correlation (toy test): {corr:.4f}")
