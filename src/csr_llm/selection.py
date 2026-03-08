"""Selection, reproduction, and island migration.

Implements the evolutionary operators:
- Island-based selection (top K per island)
- Cloning, mutation, recombination
- Prefix co-evolution
- Ring migration between islands
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from csr_llm.model import TinyGPT, mutate_model, recombine_models, save_model


@dataclass
class Individual:
    """A member of the population."""

    model_id: str
    island: int
    model: TinyGPT
    prefix: str
    fitness: float = 0.0
    generation_stats: dict = field(default_factory=dict)
    parent_id: Optional[str] = None
    creation_method: str = "init"
    mutation_sigma: float = 0.0
    lineage: list[str] = field(default_factory=list)

    @property
    def prefix_hash(self) -> str:
        return hashlib.md5(self.prefix.encode()).hexdigest()[:8]


def assign_islands(population: list[Individual], n_islands: int) -> None:
    """Assign individuals to islands (round-robin)."""
    for i, ind in enumerate(population):
        ind.island = i % n_islands


def select_survivors(
    population: list[Individual],
    n_islands: int,
    survivors_per_island: int,
) -> list[Individual]:
    """Select top K individuals per island."""
    survivors = []

    for island in range(n_islands):
        island_pop = [ind for ind in population if ind.island == island]
        island_pop.sort(key=lambda x: x.fitness, reverse=True)
        survivors.extend(island_pop[:survivors_per_island])

    return survivors


def reproduce(
    survivors: list[Individual],
    offspring_per_survivor: int,
    offspring_distribution: dict,
    mutation_cfg: dict,
    round_num: int,
    rng: random.Random,
) -> list[Individual]:
    """Create next generation from survivors.

    Each survivor produces offspring according to the distribution:
    - clone: exact copy
    - mutant_small: small weight perturbation
    - mutant_large: large weight perturbation
    - recombinant: layer swap with another survivor
    """
    next_gen = []
    model_counter = 0

    for parent in survivors:
        methods = []
        for method, count in offspring_distribution.items():
            methods.extend([method] * count)

        for method in methods:
            model_id = f"model_{model_counter:03d}"
            model_counter += 1

            if method == "clone":
                child_model = copy.deepcopy(parent.model)
                child_prefix = parent.prefix
                sigma = 0.0

            elif method == "mutant_small":
                sigma = mutation_cfg["sigma_small"]
                child_model = mutate_model(parent.model, sigma=sigma, seed=rng.randint(0, 2**31))
                child_prefix = mutate_prefix(parent.prefix, rng)

            elif method == "mutant_large":
                sigma = mutation_cfg["sigma_large"]
                child_model = mutate_model(parent.model, sigma=sigma, seed=rng.randint(0, 2**31))
                child_prefix = mutate_prefix(parent.prefix, rng, aggressive=True)

            elif method == "recombinant":
                # Pick another survivor to recombine with
                other = rng.choice([s for s in survivors if s.model_id != parent.model_id])
                child_model = recombine_models(
                    parent.model, other.model, seed=rng.randint(0, 2**31)
                )
                child_prefix = recombine_prefixes(parent.prefix, other.prefix, rng)
                sigma = 0.0

            else:
                raise ValueError(f"Unknown reproduction method: {method}")

            lineage = parent.lineage + [f"round_{round_num}:{parent.model_id}"]

            child = Individual(
                model_id=model_id,
                island=parent.island,
                model=child_model,
                prefix=child_prefix,
                parent_id=parent.model_id,
                creation_method=method,
                mutation_sigma=sigma,
                lineage=lineage,
            )
            next_gen.append(child)

    return next_gen


def migrate(
    population: list[Individual],
    n_islands: int,
    n_migrants: int,
    rng: random.Random,
) -> list[dict]:
    """Ring migration: move best individual from each island to the next."""
    migrations = []

    for src_island in range(n_islands):
        dst_island = (src_island + 1) % n_islands

        # Find best in source island
        src_pop = [ind for ind in population if ind.island == src_island]
        if not src_pop:
            continue

        src_pop.sort(key=lambda x: x.fitness, reverse=True)
        migrant = src_pop[0]

        # Find worst in destination island
        dst_pop = [ind for ind in population if ind.island == dst_island]
        if not dst_pop:
            continue

        dst_pop.sort(key=lambda x: x.fitness)
        replaced = dst_pop[0]

        # Swap
        migrant.island = dst_island
        replaced.island = src_island

        migrations.append(
            {
                "from_island": src_island,
                "to_island": dst_island,
                "migrant_id": migrant.model_id,
                "replaced_id": replaced.model_id,
            }
        )

    return migrations


# --- Prefix evolution ---


def mutate_prefix(prefix: str, rng: random.Random, aggressive: bool = False) -> str:
    """Mutate a generation prefix by swapping/adding/removing examples."""
    lines = [l for l in prefix.strip().split("\n") if l.strip()]

    if not lines:
        return prefix

    # Higher mutation rates for aggressive mutation
    swap_p = 0.5 if aggressive else 0.3
    add_p = 0.3 if aggressive else 0.2
    remove_p = 0.3 if aggressive else 0.2

    # Swap: replace one example with a random correct one
    if rng.random() < swap_p and lines:
        idx = rng.randint(0, len(lines) - 1)
        lines[idx] = _random_correct_example(rng)

    # Add: append a new example
    if rng.random() < add_p and len(lines) < 8:
        lines.append(_random_correct_example(rng))

    # Remove: drop one example
    if rng.random() < remove_p and len(lines) > 2:
        idx = rng.randint(0, len(lines) - 1)
        lines.pop(idx)

    return "\n".join(lines) + "\n"


def recombine_prefixes(prefix_a: str, prefix_b: str, rng: random.Random) -> str:
    """Combine examples from two parents' prefixes."""
    lines_a = [l for l in prefix_a.strip().split("\n") if l.strip()]
    lines_b = [l for l in prefix_b.strip().split("\n") if l.strip()]

    # Take roughly half from each
    combined = []
    for line in lines_a + lines_b:
        if rng.random() < 0.5:
            combined.append(line)

    # Ensure minimum length
    if len(combined) < 2:
        combined = lines_a[:2] if len(lines_a) >= 2 else lines_a + lines_b[:2]

    # Cap at 8
    if len(combined) > 8:
        combined = rng.sample(combined, 8)

    return "\n".join(combined) + "\n"


def _random_correct_example(rng: random.Random) -> str:
    """Generate a random correct arithmetic example for prefix mutation."""
    ops = ["+", "-"]
    op = rng.choice(ops)
    a = rng.randint(0, 9)
    b = rng.randint(0, 9)
    if op == "-" and a < b:
        a, b = b, a
    result = a + b if op == "+" else a - b
    return f"{a}{op}{b}={result}"


# --- Serialization ---


def save_round_state(
    population: list[Individual],
    round_dir: Path,
    round_num: int,
    save_weights: bool = True,
) -> None:
    """Save population state for a round."""
    meta = []
    for ind in population:
        entry = {
            "model_id": ind.model_id,
            "island": ind.island,
            "prefix": ind.prefix,
            "prefix_hash": ind.prefix_hash,
            "fitness": ind.fitness,
            "parent_id": ind.parent_id,
            "creation_method": ind.creation_method,
            "mutation_sigma": ind.mutation_sigma,
            "lineage": ind.lineage,
            "generation_stats": ind.generation_stats,
        }
        meta.append(entry)

        if save_weights:
            weight_dir = round_dir / "weights"
            weight_dir.mkdir(exist_ok=True)
            save_model(ind.model, weight_dir / f"{ind.model_id}.pt")

    with open(round_dir / "population_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
