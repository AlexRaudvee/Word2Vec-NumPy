from __future__ import annotations
from collections import Counter
from typing import Iterable, Dict, List


class Vocabulary:
    def __init__(self, max_size: int, min_count: int = 1, unk_token: str = "<unk>"):
        self.max_size = max_size
        self.min_count = min_count
        self.unk_token = unk_token

        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []

    def build(self, tokenized_corpus: Iterable[list[str]]) -> None:
        counter = Counter()
        for tokens in tokenized_corpus:
            counter.update(tokens)
        
        #sort
        sorted_tokens = sorted(
            [
                t
                for t, c in counter.items()
                if c >= self.min_count and t != self.unk_token
            ],
            key=lambda t: (-counter[t], t),
        )

        #reserve index 0 for unk
        vocab_tokens = [self.unk_token] + sorted_tokens[: self.max_size - 1]

        self.word2idx = {w: i for i, w in enumerate(vocab_tokens)}
        self.idx2word = vocab_tokens

    def __len__(self) -> int:
        return len(self.idx2word)

    def encode_token(self, token: str) -> int:
        return self.word2idx.get(token, self.word2idx[self.unk_token])

    def encode_sequence(self, tokens: list[str]) -> list[int]:
        return [self.encode_token(t) for t in tokens]

    def decode_index(self, idx: int) -> str:
        return self.idx2word[idx]
