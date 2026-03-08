"""BPE tokenizer for arithmetic task."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


def train_tokenizer(
    corpus: list[str],
    vocab_size: int = 8192,
    save_path: str | Path = "tokenizer.json",
) -> Tokenizer:
    """Train a BPE tokenizer on arithmetic corpus."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # Train from iterator
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))

    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a saved tokenizer."""
    return Tokenizer.from_file(str(path))


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode text to token IDs."""
    return tokenizer.encode(text).ids


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    """Decode token IDs to text."""
    return tokenizer.decode(ids)


def get_special_ids(tokenizer: Tokenizer) -> dict[str, int]:
    """Get special token IDs."""
    return {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}
