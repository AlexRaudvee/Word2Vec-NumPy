# src/word_embeddings/training/trainer.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import os
import subprocess
import sys
import time

import numpy as np
from tqdm import tqdm

from word_embeddings.models import SGNSModel
from word_embeddings.models import GloVeModel
from word_embeddings.optim import AdagradOptimizer
from word_embeddings.data import SkipGramDataset, NegativeSampler


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    adagrad_epsilon: float
    log_every: int
    memory_log_every_steps: int
    save_dir: str


@dataclass
class TrainingMetrics:
    losses_logged: list[float] = field(default_factory=list)
    epoch_times_sec: list[float] = field(default_factory=list)
    memory_steps: list[int] = field(default_factory=list)
    memory_usage_mb: list[float] = field(default_factory=list)


def _current_memory_mb() -> float:
    """
    Return current RSS memory usage in MB.
    """
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
        # `ps` reports RSS in KB on macOS/Linux.
        return float(output) / 1024.0
    except Exception:
        pass

    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes.
        if sys.platform == "darwin":
            return float(rss) / (1024.0 * 1024.0)
        return float(rss) / 1024.0
    except Exception:
        return float("nan")


class Trainer:
    """
    Trainer that can handle both SGNS and GloVe models with mini-batch, vectorized training.
    """

    def __init__(
        self,
        model: SGNSModel | GloVeModel,
        optimizer: AdagradOptimizer,
        training_cfg,
        negative_sampler: Optional[NegativeSampler] = None,
    ):
        self.model = model
        self.optimizer = optimizer

        # training_cfg is a DictConfig from Hydra; wrap into a simple object
        self.cfg = TrainingConfig(
            batch_size=int(training_cfg.batch_size),
            epochs=int(training_cfg.epochs),
            learning_rate=float(training_cfg.learning_rate),
            adagrad_epsilon=float(training_cfg.adagrad_epsilon),
            log_every=int(training_cfg.log_every),
            memory_log_every_steps=max(
                1,
                int(
                    getattr(
                        training_cfg,
                        "memory_log_every_steps",
                        training_cfg.log_every,
                    )
                ),
            ),
            save_dir=str(training_cfg.save_dir),
        )

        self.negative_sampler = negative_sampler



    # SGNS training
    def train_sgns(self, dataset: SkipGramDataset) -> TrainingMetrics:
        """
        Train SGNS model using streaming (center, context) pairs from SkipGramDataset.

        Returns:
            TrainingMetrics with logged losses, memory samples, and epoch times.
        """
        assert isinstance(self.model, SGNSModel), "train_sgns requires SGNSModel"
        assert self.negative_sampler is not None, "SGNS training needs NegativeSampler"

        batch_size = self.cfg.batch_size
        num_neg = getattr(self.model, "num_negatives", None)
        if num_neg is None:
            raise RuntimeError(
                "SGNSModel needs attribute 'num_negatives' or you must handle num_negatives externally."
            )

        metrics = TrainingMetrics()
        global_step = 0

        for epoch in range(self.cfg.epochs):
            epoch_start = time.perf_counter()
            print(f"Epoch {epoch + 1}/{self.cfg.epochs} (SGNS)")
            approx_pairs = int(dataset.length * dataset.max_window_size + 1)
            approx_steps = max(1, int(np.ceil(approx_pairs / batch_size)))
            pbar = tqdm(
                total=approx_steps,
                desc=f"epoch {epoch + 1}/{self.cfg.epochs}",
                unit="step",
            )
            # reset streaming buffers
            centers_batch: list[int] = []
            contexts_batch: list[int] = []
            epoch_losses: list[float] = []
            epoch_steps = 0

            # TODO: shuffle by iterating over indices

            for center, context in dataset.iter_pairs():
                centers_batch.append(int(center))
                contexts_batch.append(int(context))

                if len(centers_batch) >= batch_size:
                    global_step += 1
                    center_arr = np.asarray(centers_batch, dtype=np.int32)
                    pos_arr = np.asarray(contexts_batch, dtype=np.int32)

                    # (B, K)
                    neg_arr = self.negative_sampler.sample(batch_size, num_neg)
                    # avoid drawing true positive or center word as negatives
                    collision_mask = (neg_arr == pos_arr[:, None]) | (neg_arr == center_arr[:, None])
                    attempts = 0
                    max_attempts = 10
                    while np.any(collision_mask) and attempts < max_attempts:
                        neg_arr[collision_mask] = self.negative_sampler.sample(
                            int(np.sum(collision_mask)), 1
                        ).reshape(-1)
                        collision_mask = (neg_arr == pos_arr[:, None]) | (
                            neg_arr == center_arr[:, None]
                        )
                        attempts += 1
                    # print(f"center_arr: {center_arr}, \npos_arr: {pos_arr}, \nneg_arr: {neg_arr}")
                    loss, grads = self.model.loss_and_grad(
                        center_indices=center_arr,
                        pos_indices=pos_arr,
                        neg_indices=neg_arr,
                    )
                    self.optimizer.step(self.model.params, grads)
                    epoch_losses.append(loss)
                    epoch_steps += 1
                    pbar.update(1)
                    if global_step % self.cfg.memory_log_every_steps == 0:
                        metrics.memory_steps.append(global_step)
                        metrics.memory_usage_mb.append(_current_memory_mb())

                    if global_step % self.cfg.log_every == 0:
                        print(f"[epoch {epoch + 1}] step {global_step}: loss = {loss:.4f}")
                        metrics.losses_logged.append(loss)

                    centers_batch.clear()
                    contexts_batch.clear()

            # handle last partial batch of epoch
            if len(centers_batch) > 0:
                B = len(centers_batch)
                global_step += 1
                center_arr = np.asarray(centers_batch, dtype=np.int32)
                pos_arr = np.asarray(contexts_batch, dtype=np.int32)
                neg_arr = self.negative_sampler.sample(B, num_neg)
                collision_mask = (neg_arr == pos_arr[:, None]) | (neg_arr == center_arr[:, None])
                attempts = 0
                max_attempts = 10
                while np.any(collision_mask) and attempts < max_attempts:
                    neg_arr[collision_mask] = self.negative_sampler.sample(
                        int(np.sum(collision_mask)), 1
                    ).reshape(-1)
                    collision_mask = (neg_arr == pos_arr[:, None]) | (
                        neg_arr == center_arr[:, None]
                    )
                    attempts += 1

                loss, grads = self.model.loss_and_grad(
                    center_indices=center_arr,
                    pos_indices=pos_arr,
                    neg_indices=neg_arr,
                )
                self.optimizer.step(self.model.params, grads)
                epoch_losses.append(loss)
                epoch_steps += 1
                pbar.update(1)
                if global_step % self.cfg.memory_log_every_steps == 0:
                    metrics.memory_steps.append(global_step)
                    metrics.memory_usage_mb.append(_current_memory_mb())

                print(f"[epoch {epoch + 1}] step {global_step}: loss = {loss:.4f}")
                metrics.losses_logged.append(loss)

            if epoch_steps > 0:
                epoch_loss_arr = np.asarray(epoch_losses, dtype=np.float64)
                print(
                    f"[epoch {epoch + 1}] summary: "
                    f"mean={epoch_loss_arr.mean():.4f}, "
                    f"median={np.median(epoch_loss_arr):.4f}, "
                    f"p10={np.percentile(epoch_loss_arr, 10):.4f}, "
                    f"p90={np.percentile(epoch_loss_arr, 90):.4f}, "
                    f"steps={epoch_steps}"
                )
            epoch_time = time.perf_counter() - epoch_start
            metrics.epoch_times_sec.append(epoch_time)
            print(f"[epoch {epoch + 1}] time: {epoch_time:.2f}s")
            pbar.close()

        return metrics



    # GloVe training
    def train_glove(
        self,
        row_indices: np.ndarray,  # (N_cooc,)
        col_indices: np.ndarray,  # (N_cooc,)
        values: np.ndarray,       # (N_cooc,)
    ) -> TrainingMetrics:
        """
        Train GloVe model using precomputed COO co-occurrence triplets

        Returns:
            TrainingMetrics with logged losses, memory samples, and epoch times
        """
        assert isinstance(self.model, GloVeModel), "train_glove requires GloVeModel"

        N = row_indices.shape[0]
        batch_size = self.cfg.batch_size
        num_batches = int(np.ceil(N / batch_size))

        metrics = TrainingMetrics()
        global_step = 0

        for epoch in range(self.cfg.epochs):
            epoch_start = time.perf_counter()
            print(f"Epoch {epoch + 1}/{self.cfg.epochs} (GloVe)")

            # shuffle triplets each epoch
            perm = np.random.permutation(N)
            rows_shuffled = row_indices[perm]
            cols_shuffled = col_indices[perm]
            vals_shuffled = values[perm]

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(N, (batch_idx + 1) * batch_size)

                r_batch = rows_shuffled[start:end]
                c_batch = cols_shuffled[start:end]
                v_batch = vals_shuffled[start:end]

                global_step += 1

                loss, grads = self.model.loss_and_grad_cooc(
                    row_indices=r_batch,
                    col_indices=c_batch,
                    x_ij=v_batch,
                )
                self.optimizer.step(self.model.params, grads)
                if global_step % self.cfg.memory_log_every_steps == 0:
                    metrics.memory_steps.append(global_step)
                    metrics.memory_usage_mb.append(_current_memory_mb())

                if global_step % self.cfg.log_every == 0:
                    print(f"[epoch {epoch + 1}] step {global_step}: loss = {loss:.4f}")
                    metrics.losses_logged.append(loss)

            epoch_time = time.perf_counter() - epoch_start
            metrics.epoch_times_sec.append(epoch_time)
            print(f"[epoch {epoch + 1}] time: {epoch_time:.2f}s")

        return metrics
