from __future__ import annotations
from pathlib import Path
from typing import Iterable


def read_wikitext2(root_dir: str) -> dict[str, Iterable[str]]:
    """
    Expects files in:
    root_dir/wikitext-2/train.txt
    root_dir/wikitext-2/valid.txt
    root_dir/wikitext-2/test.txt
    """
    root = Path(root_dir) / "wikitext-2"
    splits = {
        "train": root / "train.txt", # train_.txt for debug purpooses
        "valid": root / "valid.txt",
        "test": root / "test.txt",
    }
    data = {}
    for split, path in splits.items():
        with path.open("r", encoding="utf-8") as f:
            data[split] = [line.strip() for line in f if line.strip() != ""]

    return data