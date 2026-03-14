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
    batch_size: int = 32,
) -> str:
    """Prompt a model to generate arithmetic examples.

    Generates each example independently in batches so that EOS on one
    example does not kill the rest. Previously a single long generation
    would stop at the first EOS (after ~1 pretraining chunk = ~16 examples),
    starving the evolutionary loop of training data.

    Returns raw text output (one attempted example per line).
    """
    model = model.to(device)
    model.eval()

    prefix_ids = encode(tokenizer, prefix)
    prefix_len = len(prefix_ids)
    eos_id = tokenizer.token_to_id("<eos>")

    all_lines = []
    remaining = n_examples

    while remaining > 0:
        bs = min(batch_size, remaining)
        # Repeat the prefix for each item in the batch
        input_ids = torch.tensor(
            [prefix_ids] * bs, dtype=torch.long, device=device
        )

        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens_per_example,
            temperature=temperature,
            eos_token_id=eos_id,
        )

        for b in range(bs):
            generated_ids = output_ids[b, prefix_len:].tolist()
            text = decode(tokenizer, generated_ids)
            # Take only the first line from each generation attempt
            first_line = text.split("\n")[0].strip()
            if first_line:
                all_lines.append(first_line)

        remaining -= bs

    return "\n".join(all_lines)


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
