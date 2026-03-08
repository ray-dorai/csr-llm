"""Arithmetic task: problem generation, parsing, and evaluation."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Problem:
    """A single arithmetic problem."""

    expression: str  # e.g. "3+5"
    answer: str  # e.g. "8"
    full: str  # e.g. "3+5=8"
    operation: str  # e.g. "+"


@dataclass
class ParsedExample:
    """A parsed training example from model generation."""

    raw: str
    expression: Optional[str]
    answer: Optional[str]
    correct_answer: Optional[str]
    is_valid: bool
    is_correct: bool


def generate_test_set(
    n: int,
    operations: list[str],
    operand_range: tuple[int, int],
    seed: int = 12345,
    multi_step: bool = False,
) -> list[Problem]:
    """Generate a fixed test set of arithmetic problems."""
    rng = random.Random(seed)
    problems = []
    lo, hi = operand_range

    for _ in range(n):
        op = rng.choice(operations)
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)

        # Avoid division by zero and non-integer division
        if op == "/":
            b = rng.randint(max(1, lo), hi)
            a = b * rng.randint(1, max(1, hi // max(1, b)))
        # Avoid negative results for subtraction (cleaner for tiny models)
        if op == "-" and a < b:
            a, b = b, a

        expr = f"{a}{op}{b}"
        answer = str(_eval_expr(expr))
        problems.append(Problem(expression=expr, answer=answer, full=f"{expr}={answer}", operation=op))

    return problems


def _eval_expr(expr: str) -> int:
    """Safely evaluate a simple arithmetic expression."""
    # Only allow digits and basic operators
    if not re.match(r"^[\d+\-*/]+$", expr):
        raise ValueError(f"Invalid expression: {expr}")
    try:
        result = eval(expr)  # Safe because we validated the format
        return int(result)
    except Exception:
        return 0


def save_test_set(problems: list[Problem], path: str | Path) -> None:
    """Save test set to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"expression": p.expression, "answer": p.answer, "full": p.full, "operation": p.operation} for p in problems]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_test_set(path: str | Path) -> list[Problem]:
    """Load test set from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [Problem(**d) for d in data]


def generate_training_corpus(
    n: int,
    operations: list[str],
    operand_range: tuple[int, int],
    seed: int = 42,
) -> list[str]:
    """Generate correct arithmetic examples for pretraining."""
    problems = generate_test_set(n, operations, operand_range, seed=seed)
    return [p.full for p in problems]


# --- Parsing model-generated output ---

# Pattern: one or more digits, operator, one or more digits, equals, answer
EXAMPLE_PATTERN = re.compile(r"^(\d+)([+\-*/])(\d+)=(-?\d+)$")


def parse_generated_output(raw_text: str) -> list[ParsedExample]:
    """Parse raw model output into structured examples."""
    examples = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        match = EXAMPLE_PATTERN.match(line)
        if match:
            a, op, b, claimed = match.groups()
            expr = f"{a}{op}{b}"
            correct = str(_eval_expr(expr))
            examples.append(
                ParsedExample(
                    raw=line,
                    expression=expr,
                    answer=claimed,
                    correct_answer=correct,
                    is_valid=True,
                    is_correct=(claimed == correct),
                )
            )
        else:
            examples.append(
                ParsedExample(
                    raw=line,
                    expression=None,
                    answer=None,
                    correct_answer=None,
                    is_valid=False,
                    is_correct=False,
                )
            )

    return examples


def summarize_parsed(examples: list[ParsedExample]) -> dict:
    """Summarize parsed generation output."""
    valid = [e for e in examples if e.is_valid]
    correct = [e for e in examples if e.is_correct]
    incorrect = [e for e in valid if not e.is_correct]
    invalid = [e for e in examples if not e.is_valid]

    return {
        "total": len(examples),
        "valid": len(valid),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "unparseable": len(invalid),
        "valid_rate": len(valid) / max(1, len(examples)),
        "correct_rate": len(correct) / max(1, len(examples)),
        "accuracy_of_valid": len(correct) / max(1, len(valid)),
        "sample_correct": [e.raw for e in correct[:5]],
        "sample_incorrect": [e.raw for e in incorrect[:5]],
        "sample_unparseable": [e.raw for e in invalid[:5]],
    }
