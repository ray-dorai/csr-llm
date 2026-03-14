"""Generate training examples from a parent model.

This is the "self-replication" step: each parent model generates
training data that will be used to train its offspring.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from csr_llm.model import TinyGPT
from csr_llm.task import ParsedExample, parse_generated_output, summarize_parsed
from csr_llm.tokenizer import decode, encode, load_tokenizer


def generate_examples(
    model: TinyGPT,
    tokenizer,
    prefix: str,
    n_examples: int = 100,
    max_tokens_per_example: int = 20,
    temperature: float = 1.0,
    device: str = "cuda",
) -> str:
    """Prompt a model to generate arithmetic examples.

    The model is given a prefix of correct examples and asked to
    continue generating in the same format.

    Returns raw text output.
    """
    model = model.to(device)
    model.eval()

    # Encode prefix
    prefix_ids = encode(tokenizer, prefix)
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    # Generate enough tokens for n_examples
    # Each example is roughly "N+M=K\n" = ~6-8 tokens
    total_new = n_examples * max_tokens_per_example

    output_ids = model.generate(
        input_ids,
        max_new_tokens=total_new,
        temperature=temperature,
    )

    # Decode only the generated part (not the prefix)
    generated_ids = output_ids[0, len(prefix_ids) :].tolist()
    raw_text = decode(tokenizer, generated_ids)

    return raw_text


def generate_and_parse(
    model: TinyGPT,
    tokenizer,
    prefix: str,
    n_examples: int = 100,
    max_tokens_per_example: int = 20,
    temperature: float = 1.0,
    device: str = "cuda",
) -> tuple[str, list[ParsedExample], dict]:
    """Generate, parse, and summarize in one call.

    Returns:
        raw_text: The raw generated string
        parsed: List of ParsedExample objects
        summary: Dict of statistics
    """
    raw = generate_examples(
        model, tokenizer, prefix, n_examples, max_tokens_per_example, temperature, device
    )
    parsed = parse_generated_output(raw)
    summary = summarize_parsed(parsed)

    return raw, parsed, summary


def save_generation(
    model_id: str,
    round_dir: Path,
    raw_text: str,
    parsed: list[ParsedExample],
    summary: dict,
    prefix: str,
) -> None:
    """Save all generation artifacts to disk."""
    # Raw output
    (round_dir / "generated" / f"{model_id}.txt").write_text(raw_text)

    # Parsed + summary
    parsed_data = {
        "model_id": model_id,
        "examples": [
            {
                "raw": e.raw,
                "expression": e.expression,
                "answer": e.answer,
                "correct_answer": e.correct_answer,
                "is_valid": e.is_valid,
                "is_correct": e.is_correct,
            }
            for e in parsed
        ],
        **summary,
    }
    with open(round_dir / "parsed" / f"{model_id}.json", "w") as f:
        json.dump(parsed_data, f, indent=2)

    # Prefix used
    (round_dir / "generation_prefixes" / f"{model_id}.txt").write_text(prefix)
