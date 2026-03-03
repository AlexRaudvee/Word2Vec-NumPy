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

## Results


## Discussion


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
