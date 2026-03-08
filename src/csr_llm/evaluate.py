"""Evaluate offspring models on held-out test set.

The offspring's score becomes the parent's fitness.
This is the selection signal — parents that generated better
training data will have higher-scoring offspring.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from csr_llm.model import TinyGPT
from csr_llm.task import Problem
from csr_llm.tokenizer import decode, encode


def evaluate_offspring(
    model: TinyGPT,
    tokenizer,
    test_set: list[Problem],
    device: str = "cuda",
    max_new_tokens: int = 4,
) -> dict:
    """Evaluate a model on the held-out test set.

    For each problem, feed "N+M=" and check if greedy completion
    matches the correct answer.

    Returns:
        Dict with score, accuracy, per-problem results, and per-operation breakdown.
    """
    model = model.to(device)
    model.eval()

    results = []
    correct = 0
    per_op = {}

    with torch.no_grad():
        for prob in test_set:
            # Encode the prompt: "N+M="
            prompt = prob.expression + "="
            prompt_ids = encode(tokenizer, prompt)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # Greedy generation
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )

            # Decode generated part
            gen_ids = output_ids[0, len(prompt_ids) :].tolist()
            generated = decode(tokenizer, gen_ids).strip()

            # Extract first "word" (the answer before any newline or space)
            predicted = generated.split("\n")[0].split(" ")[0].strip()

            is_correct = predicted == prob.answer

            if is_correct:
                correct += 1

            # Per-operation tracking
            op = prob.operation
            if op not in per_op:
                per_op[op] = {"correct": 0, "total": 0}
            per_op[op]["total"] += 1
            if is_correct:
                per_op[op]["correct"] += 1

            results.append(
                {
                    "expression": prob.expression,
                    "expected": prob.answer,
                    "predicted": predicted,
                    "correct": is_correct,
                    "operation": op,
                }
            )

    total = len(test_set)

    # Compute per-operation accuracy
    for op in per_op:
        per_op[op]["accuracy"] = per_op[op]["correct"] / max(1, per_op[op]["total"])

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / max(1, total),
        "per_operation": per_op,
        "per_problem": results,
    }


def save_score(
    model_id: str,
    round_dir: Path,
    score: dict,
    metadata: dict,
) -> None:
    """Save evaluation results to disk."""
    data = {
        "model_id": model_id,
        **metadata,
        "offspring_score": {
            "correct": score["correct"],
            "total": score["total"],
            "accuracy": score["accuracy"],
            "per_operation": score["per_operation"],
        },
        # Don't save per_problem in the main file (too large)
        # Save it separately if needed
    }

    with open(round_dir / "scores" / f"{model_id}.json", "w") as f:
        json.dump(data, f, indent=2)

    # Save detailed per-problem results separately
    detail_dir = round_dir / "scores" / "detail"
    detail_dir.mkdir(exist_ok=True)
    with open(detail_dir / f"{model_id}.json", "w") as f:
        json.dump(score["per_problem"], f)
