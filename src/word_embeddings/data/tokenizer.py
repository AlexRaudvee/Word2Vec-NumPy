from __future__ import annotations
import re
from typing import Iterable, List


_TOKEN_RE = re.compile(r"<unk>|\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def tokenize_corpus(lines: Iterable[str], lowercase: bool = True) -> list[list[str]]:
    return [simple_tokenize(line, lowercase=lowercase) for line in lines]
