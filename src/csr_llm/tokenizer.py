"""Character-level tokenizer for arithmetic task."""

from __future__ import annotations

import json
from pathlib import Path


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]

# All characters that appear in arithmetic: digits, operators, equals, newline
VOCAB_CHARS = list("0123456789+-*/=\n")


class CharTokenizer:
    """Simple character-level tokenizer for arithmetic."""

    def __init__(self):
        self._tok2id = {}
        self._id2tok = {}

        # Special tokens first
        for tok in SPECIAL_TOKENS:
            idx = len(self._tok2id)
            self._tok2id[tok] = idx
            self._id2tok[idx] = tok

        # Character tokens
        for ch in VOCAB_CHARS:
            idx = len(self._tok2id)
            self._tok2id[ch] = idx
            self._id2tok[idx] = ch

    def token_to_id(self, token: str) -> int | None:
        return self._tok2id.get(token)

    def get_vocab_size(self) -> int:
        return len(self._tok2id)

    def encode(self, text: str) -> list[int]:
        ids = []
        for ch in text:
            if ch in self._tok2id:
                ids.append(self._tok2id[ch])
            # Skip unknown characters
        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []
        for i in ids:
            tok = self._id2tok.get(i, "")
            if tok not in SPECIAL_TOKENS:
                chars.append(tok)
        return "".join(chars)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"tok2id": self._tok2id}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok._tok2id = data["tok2id"]
        tok._id2tok = {int(v): k for k, v in tok._tok2id.items()}
        return tok


def train_tokenizer(
    corpus: list[str],
    vocab_size: int = 8192,
    save_path: str | Path = "tokenizer.json",
) -> CharTokenizer:
    """Create and save a character-level tokenizer."""
    tokenizer = CharTokenizer()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    return tokenizer


def load_tokenizer(path: str | Path) -> CharTokenizer:
    """Load a saved tokenizer."""
    return CharTokenizer.load(str(path))


def encode(tokenizer: CharTokenizer, text: str) -> list[int]:
    """Encode text to token IDs."""
    return tokenizer.encode(text)


def decode(tokenizer: CharTokenizer, ids: list[int]) -> str:
    """Decode token IDs to text."""
    return tokenizer.decode(ids)


def get_special_ids(tokenizer: CharTokenizer) -> dict[str, int]:
    """Get special token IDs."""
    return {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}
