# Word2Vec-NumPy

## Description
NumPy-based word embedding project (SGNS and GloVe) managed with `uv` for environments/dependencies and configured with Hydra (`configs/`) for reproducible training/evaluation runs. Training is launched via Hydra overrides, e.g. `uv run python src/train.py model=sgns training.epochs=10 data.vocab_size=30000` or configs can be changed right in yaml files inside `configs/`.

## Datasets
- **WikiText-2** (`data/wikitext-2`):
  - Purpose: primary corpus for learning word embeddings.
  - Usage: `train.txt` is tokenized, converted to vocabulary indices, then used for SGNS skip-gram pairs or GloVe co-occurrence statistics.
- **MEN** (`data/MEN/MEN_dataset_natural_form_full`):
  - Purpose: intrinsic semantic similarity benchmark.
  - Usage: model cosine similarities are compared against human similarity scores using Spearman correlation.
- **MSR Analogies** (`data/msr/msr.txt`):
  - Purpose: intrinsic analogy reasoning benchmark.
  - Usage: vector arithmetic (`b - a + c`) is used to predict `d`; reported as top-k analogy accuracy.
- **AG News** (`data/ag_news`):
  - Purpose: extrinsic downstream benchmark.
  - Usage: document vectors are built by averaging token embeddings, then a logistic regression classifier is trained/evaluated for news category classification.

## Models
- **SGNS** (Skip-Gram with Negative Sampling, Word2Vec-style):
  - Learns embeddings by predicting nearby context words from a center word.
  - Uses dynamic context windows and negative sampling (`negative_samples`, `neg_sampling_power`) for efficient training.
  - Best aligned with local context prediction objective.
- **GloVe** (Global Vectors):
  - Learns embeddings by factorizing global word-word co-occurrence statistics.
  - Optimizes a weighted least-squares objective with `xmax` and `alpha` weighting hyperparameters.
  - Best aligned with capturing global corpus co-occurrence structure.

## Results and Discussion

#### Training Behaviour

The training dynamics of the GloVe model are shown on [this figure](assets/training_loss_GloVeModel.png), where the loss rapidly decreases during the early iterations and stabilizes after approximately 50–100 logged steps. This behaviour indicates that the model successfully converges toward the optimal solution. The GloVe objective aims to approximate the log of word co-occurrence counts through the dot product of word vectors and bias terms, using a weighting function to reduce the influence of extremely frequent or rare co-occurrences.

The final loss values are relatively small (0.002). This is expected because the optimization objective minimizes a squared regression error rather than a probabilistic likelihood, meaning the loss approaches zero once the vector dot products approximate the log-co-occurrence statistics. Overall, the loss curve demonstrates stable convergence without evidence of divergence or unstable training. 

In contrast, the [SGNS model](assets/training_loss_SGNSModel.png) shows a noisier training curve where the loss fluctuates between approximately 2.1 and 2.6 while gradually decreasing. This behaviour is typical for SGNS because the model is trained as a binary classifier that distinguishes true word–context pairs from randomly sampled negative pairs using a logistic loss. Each training step uses randomly sampled negative examples, which introduces stochasticity into the gradient updates and causes the loss to fluctuate rather than smoothly converge. Despite these fluctuations, the overall downward trend of the curve indicates that the model is successfully learning meaningful word representations.

#### Computational Efficiency

The efficiency of the implementation was evaluated by measuring both memory usage during training and training time per epoch, shown in [this](assets/training_memory_GloVeModel.png) and [this](assets/training_memory_SGNSModel.png) figures

Memory consumption in case with [GloVe model](assets/training_memory_GloVeModel.png) remains relatively stable between approximately 130 MB and 320 MB. Periodic spikes occur due to batch processing and temporary allocation of intermediate arrays during gradient updates. Despite these fluctuations, the overall memory footprint remains modest given the vocabulary size and embedding dimensionality used in the experiments. In case with SGNS model we can see even more stable behaviour and lover MB usage (around 150 MB). This can be explained by the training mechanism of SGNS, where the model updates only a small subset of embeddings at each iteration through negative sampling. Instead of computing gradients over the entire vocabulary, the algorithm compares one positive word–context pair with a small number of randomly sampled negative examples. As a result, each training step updates only $K+1$ vectors rather than all vocabulary vectors, significantly reducing the computational cost and memory requirements

With GloVe model [raining time per epoch](assets/training_time_per_epoch_GloVeModel.png) averages roughly 120–130 seconds, with occasional increases to approximately 170–180 seconds in later epochs. These variations likely arise from operating system scheduling and temporary memory allocation overhead. In the case of the SGNS model, the training time per epoch is noticeably higher and shows a different pattern compared to GloVe. As illustrated in [this figure](assets/training_time_per_epoch_SGNSModel.png), the first epochs require approximately 680–730 seconds, indicating a relatively stable training time during the early stage of optimization. However, starting from around epoch 6, the training time increases significantly to roughly 980–1060 seconds per epoch (prabably because of stochastic nature of the negative sampling training procedure, which introduce variability in runtime across epochs). Nevertheless, the training process remains computationally manageable, demonstrating that both SGNS and GloVe can be implemented efficiently using vectorized NumPy operations even without GPU acceleration.

However it is wort to mention that GloVe model requires higher amount of epochs and batch size to produce more less meaningfull embedings, while SGNS model was trained only on 7 epochs and performs quite closely to GloVe.

#### Intrinsic Evaluation

Intrinsic evaluation measures how well the embeddings capture semantic relationships between words.

**Word Similarity**

The models were evaluated using the MEN similarity dataset, which measures correlation between cosine similarity of word vectors and human similarity judgments.

| Model	| Spearman Correlation |
|---|---|
| GloVe	| 0.269 |
| SGNS	| 0.152 |

The results indicate that GloVe significantly outperforms SGNS on semantic similarity. This difference is consistent with the underlying design of the two models. GloVe learns embeddings by factorizing global co-occurrence statistics, while SGNS learns representations by predicting nearby context words through local training signals. As a result, GloVe tends to capture global semantic relationships more effectively

**Analogy Task**

The embeddings were also evaluated on the MSR analogy dataset, which measures whether linear relationships between word vectors encode semantic or syntactic analogies.

| Model	| Accuracy |
|---|---|
|GloVe |	2.08% |
| SGNS |	1.95% |

Both models achieve relatively low analogy accuracy. This result is expected because the WikiText-2 corpus used for training is relatively small compared to datasets typically used for word embedding training. Analogy tasks usually require very large corpora to learn reliable linear relationships between word vectors

**Nearest Neighbour Analysis**

Nearest-neighbour inspection provides qualitative insight into the semantic structure of the embeddings. For GloVe, many semantic clusters appear coherent. For example:

king -> henry, charles, edward

queen -> elizabeth, victoria

france -> germany, italy, spain

These results indicate that the model captures historical and geopolitical relationships.

SGNS also captures some meaningful relationships, such as:

king -> queen, lord

france -> germany, italy

However, several neighbours appear less semantically coherent (e.g., unrelated proper nouns), suggesting weaker global structure compared to GloVe

#### Extrinsic Evaluation

Extrinsic evaluation measures how useful the learned embeddings are for downstream tasks. In this work, embeddings were evaluated using the AG News text classification dataset

|Model | Accuracy | Macro F1 |
|---|---|---|
|GloVe | 0.765 | 0.764 |
|SGNS | 0.650 | 0.648 |

The results show that GloVe embeddings significantly outperform SGNS in the downstream classification task. This suggests that embeddings trained using global co-occurrence statistics may encode semantic information that is more useful for document-level tasks, however, the training time is also higher

One possible explanation is that GloVe explicitly models global co-occurrence structure, while SGNS focuses primarily on predicting local context words through negative sampling. Negative sampling reduces computational cost by contrasting positive word pairs with randomly sampled negative examples rather than computing probabilities over the entire vocabulary

While this approach makes SGNS highly efficient, it may also limit the amount of global statistical information captured by the embeddings

#### Embedding Space Visualization

The PCA visualizations of the embedding spaces are shown on the following figures: [GloVe](assets/pca_glove.png), [SGNS](assets/pca_sgns.png)

The GloVe visualization shows a relatively well distributed embedding space, where words form several small clusters representing semantic categories. This indicates that the model has learned more less meaningful global structure. In contrast, the SGNS visualization shows a denser central cluster, suggesting that many word vectors are concentrated near the origin of the embedding space. This may indicate weaker separation between semantic groups (which is makes sense because model wasn't trained for a longer time). As an example, in both cases we can see that digits like 1,2,3 are clustered together, same happens with punctuation marks like "." and ",". This indicated that our models learn meaningfull embeding space, and therefore on larger corpus and with higher amount of training steps we can obtain usable embedding space.


## Project Overview
```text
.
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Locked dependency versions (uv)
├── README.md
├── .gitignore
├── LICENSE
│
├── configs/                    # Hydra configuration tree
│   ├── config.yaml             # Main default config
│   ├── data/
│   │   └── wikitext2.yaml      # Data/vocabulary/window settings for training purposes
│   ├── model/
│   │   ├── sgns.yaml           # SGNS hyperparameters
│   │   └── glove.yaml          # GloVe hyperparameters
│   ├── training/
│   │   └── base.yaml           # Batch size, epochs, LR, logging
│   └── experiment/
│       ├── sgns_intrinsic.yaml
│       ├── sgns_extrinsic.yaml
│       ├── sgns_visualization.yaml
│       ├── glove_intrinsic.yaml
│       ├── glove_extrinsic.yaml
│       └── glove_visualization.yaml
│
├── scripts/
│   └── data_download.py         # Downloads MEN, MSR, AG News, WikiText-2
│
├── src/
│   ├── train.py                 # Train SGNS/GloVe, save checkpoints, basic curves
│   ├── evaluate.py              # Training-time plots + quick qualitative checks
│   ├── evaluate_intrinsic.py    # MEN + MSR + nearest neighbors
│   ├── evaluate_extrinsic.py    # AG News classification benchmark
│   ├── evaluate_visualization.py# PCA visualization of embeddings
│   ├── inference.py             # Interactive nearest-neighbor lookup
│   └── word_embeddings/
│       ├── data/                # Tokenizer, vocabulary, dataset builders, loader
│       ├── models/              # SGNS and GloVe model definitions
│       ├── optim/               # AdaGrad optimizer
│       ├── training/            # Trainer + metric tracking
│       └── utils/               # Similarity/init/gradient-check utilities
│
├── data/                        # Downloaded datasets (local)
├── checkpoints/                 # Saved weights + vocab JSON
├── assets/                      # Plots and evaluation JSON outputs
└── outputs/                     # Hydra run logs by date/time
```

## Set up
From project root.

### Unix / macOS

1. Install `uv`:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Verify installation:
   ```bash
   uv --version
   ```
3. Create virtual environment:
   ```bash
   uv venv
   ```
4. Activate environment:
   ```bash
   source .venv/bin/activate
   ```
5. Install dependencies:
   ```bash
   uv sync
   ```

### Windows (PowerShell)

1. Install `uv` (recommended via `winget`):
   ```powershell
   winget install --id=astral-sh.uv -e
   ```
2. Verify installation:
   ```powershell
   uv --version
   ```
3. Create virtual environment:
   ```powershell
   uv venv
   ```
4. Activate environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
5. Install dependencies:
   ```powershell
   uv sync
   ```

### Windows (CMD)

1. Create virtual environment:
   ```bat
   uv venv
   ```
2. Activate environment:
   ```bat
   .venv\Scripts\activate.bat
   ```
3. Install dependencies:
   ```bat
   uv sync
   ```

## How to run
Chronological workflow from fresh clone:

1. Download all required datasets:
   ```bash
   uv run python scripts/data_download.py
   ```

2. Ensure WikiText training filename matches loader expectation:
   ```bash
   test -f data/wikitext-2/train.txt || cp data/wikitext-2/train.txt data/wikitext-2/train_.txt
   ```

3. (Optional) Quick loader sanity check:
   ```bash
   uv run python -c "import sys; sys.path.append('src'); from word_embeddings.data.wikitext_loader import read_wikitext2; d=read_wikitext2('data'); print({k: len(v) for k, v in d.items()})"
   ```

4. Train SGNS:
   ```bash
   uv run python src/train.py model=sgns
   ```
   Hydra example (override config values at runtime):
   ```bash
   uv run python src/train.py model=sgns training.epochs=10 training.batch_size=64 data.max_window_size=8
   ```

5. Train GloVe:
   ```bash
   uv run python src/train.py model=glove
   ```
   Hydra example (switch model + tune hyperparameters):
   ```bash
   uv run python src/train.py model=glove model.embedding_dim=200 model.xmax=50 training.learning_rate=0.03
   ```

6. Run intrinsic evaluation:
   ```bash
   uv run python src/evaluate_intrinsic.py experiment=sgns_intrinsic model=sgns
   uv run python src/evaluate_intrinsic.py experiment=glove_intrinsic model=glove
   ```

7. Run extrinsic evaluation (AG News):
   ```bash
   uv run python src/evaluate_extrinsic.py experiment=sgns_extrinsic model=sgns
   uv run python src/evaluate_extrinsic.py experiment=glove_extrinsic model=glove
   ```

8. Generate embedding visualization (PCA):
   ```bash
   uv run python src/evaluate_visualization.py experiment=sgns_visualization model=sgns
   uv run python src/evaluate_visualization.py experiment=glove_visualization model=glove
   ```

9. Run interactive inference (nearest neighbors):
   ```bash
   uv run python src/inference.py
   ```
   Notes:
   - `src/inference.py` currently defaults to `model_name="sgns"`.
   - To query GloVe, change `model_name` in `src/inference.py` to `"glove"`.

### Where outputs are written
- Checkpoints: `checkpoints/`
- Metrics and figures: `assets/`
- Hydra logs: `outputs/YYYY-MM-DD/HH-MM-SS/`
