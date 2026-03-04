#!/usr/bin/env bash

# for debug
set -e

# train sgns model
uv run python src/train.py model=sgns training.learning_rate=0.02

# train glove model
uv run python src/train.py model=glove training.learning_rate=0.01

# intrinsic evaluation
uv run python src/evaluate_intrinsic.py experiment=sgns_intrinsic model=sgns
uv run python src/evaluate_intrinsic.py experiment=glove_intrinsic model=glove

# extrinsic evaluation
uv run python src/evaluate_extrinsic.py experiment=sgns_extrinsic model=sgns
uv run python src/evaluate_extrinsic.py experiment=glove_extrinsic model=glove

# visualization 
uv run python src/evaluate_visualization.py experiment=sgns_visualization model=sgns
uv run python src/evaluate_visualization.py experiment=glove_visualization model=glove

