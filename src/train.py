# src/train.py

from __future__ import annotations
import os
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

from word_embeddings.data.wikitext_loader import read_wikitext2
from word_embeddings.data.tokenizer import tokenize_corpus
from word_embeddings.data.vocabulary import Vocabulary
from word_embeddings.data.skipgram_dataset import SkipGramDataset
from word_embeddings.data.cooccurrence_builder import build_cooccurrence_triplets
from word_embeddings.data.negative_sampler import NegativeSampler

from word_embeddings.models.sgns_model import SGNSModel
from word_embeddings.models.glove_model import GloVeModel

from word_embeddings.optim.adagrad import AdagradOptimizer
from word_embeddings.training.trainer import Trainer
from evaluate import evaluate_model


def save_checkpoint(model, vocab, save_dir: str, model_name: str):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(save_dir, f"{model_name}_W_in.npy"), model.params.W_in)
    np.save(os.path.join(save_dir, f"{model_name}_W_out.npy"), model.params.W_out)
    np.save(os.path.join(save_dir, f"{model_name}_b_in.npy"), model.params.b_in)
    np.save(os.path.join(save_dir, f"{model_name}_b_out.npy"), model.params.b_out)

    with open(os.path.join(save_dir, f"{model_name}_vocab.json"), "w") as f:
        json.dump(vocab.idx2word, f)

    print(f"Checkpoint saved to {save_dir}")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.seed)

    # load data
    splits = read_wikitext2(cfg.data.root_dir)
    train_lines = splits["train"]
    # print(f"Loaded {len(train_lines)} lines from training data.")
    # print(train_lines)
    tokenized = tokenize_corpus(train_lines, lowercase=cfg.data.lowercase)
    print(f"Tokenized training data. {tokenized}")
    vocab = Vocabulary(
        max_size=cfg.data.vocab_size,
        min_count=cfg.data.min_count,
    )
    vocab.build(tokenized)
    # print(f"Vocabulary built. Size: {len(vocab)}. vocab: {vocab}")
    encoded_sequences = [vocab.encode_sequence(toks) for toks in tokenized]
    print(f"Encoded training data into indices. {encoded_sequences}")


    # model selection
    if cfg.model.name == "sgns":

        dataset = SkipGramDataset(
            sequences=encoded_sequences,
            max_window_size=cfg.data.max_window_size,
        )

        counts = np.zeros(len(vocab), dtype=np.int64)
        for seq in encoded_sequences:
            for idx in seq:
                counts[idx] += 1

        neg_sampler = NegativeSampler(
            counts=counts,
            power=cfg.model.neg_sampling_power,
        )

        model = SGNSModel(
            vocab_size=len(vocab),
            embedding_dim=cfg.model.embedding_dim,
            num_negatives=cfg.model.negative_samples,
            seed=cfg.seed,
        )

        optimizer = AdagradOptimizer(
            lr=cfg.training.learning_rate,
            epsilon=cfg.training.adagrad_epsilon,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            training_cfg=cfg.training,
            negative_sampler=neg_sampler,
        )

        training_metrics = trainer.train_sgns(dataset)

    elif cfg.model.name == "glove":

        rows, cols, vals = build_cooccurrence_triplets(
            sequences=encoded_sequences,
            max_window_size=cfg.data.max_window_size,
        )

        # print(f"co-occurence triplets build: ")
        # print(f"rows: {rows}")
        # print(f"cols: {cols}")
        # print(f"vals: {vals}")

        model = GloVeModel(
            vocab_size=len(vocab),
            embedding_dim=cfg.model.embedding_dim,
            xmax=cfg.model.xmax,
            alpha=cfg.model.alpha,
            seed=cfg.seed,
        )

        optimizer = AdagradOptimizer(
            lr=cfg.training.learning_rate,
            epsilon=cfg.training.adagrad_epsilon,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            training_cfg=cfg.training,
        )

        training_metrics = trainer.train_glove(rows, cols, vals)

    else:
        raise ValueError("Unknown model type.")

    # -------------------------
    # 3. Save Model
    # -------------------------
    save_checkpoint(
        model=model,
        vocab=vocab,
        save_dir=cfg.training.save_dir,
        model_name=cfg.model.name,
    )

    # -------------------------
    # 4. Evaluation
    # -------------------------
    os.makedirs("assets", exist_ok=True)
    evaluate_model(model, vocab, training_metrics)


if __name__ == "__main__":
    main()
