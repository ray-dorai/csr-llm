"""Tests for selection, reproduction, and prefix evolution."""

import random

import torch

from csr_llm.model import TinyGPT
from csr_llm.selection import (
    Individual,
    assign_islands,
    migrate,
    mutate_prefix,
    recombine_prefixes,
    reproduce,
    select_survivors,
)


def _make_pop(n=16, n_islands=4):
    pop = []
    for i in range(n):
        m = TinyGPT(vocab_size=32, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=8)
        ind = Individual(
            model_id=f"m{i:03d}", island=0, model=m,
            prefix="1+1=2\n", fitness=float(i),
        )
        pop.append(ind)
    assign_islands(pop, n_islands)
    return pop


def test_assign_islands():
    pop = _make_pop(16, 4)
    for isl in range(4):
        count = sum(1 for ind in pop if ind.island == isl)
        assert count == 4


def test_select_survivors():
    pop = _make_pop(16, 4)
    survivors = select_survivors(pop, n_islands=4, survivors_per_island=1)
    assert len(survivors) == 4
    # Each survivor should be the best in its island
    for isl in range(4):
        island_pop = [ind for ind in pop if ind.island == isl]
        best = max(island_pop, key=lambda x: x.fitness)
        surv = [s for s in survivors if s.island == isl]
        assert len(surv) == 1
        assert surv[0].model_id == best.model_id


def test_reproduce_count():
    pop = _make_pop(16, 4)
    survivors = select_survivors(pop, 4, 1)
    rng = random.Random(42)
    dist = {"clone": 1, "mutant_small": 2, "recombinant": 1}
    next_gen = reproduce(survivors, 4, dist, {"sigma_small": 0.01, "sigma_large": 0.05}, 1, rng)
    assert len(next_gen) == 4 * 4  # 4 survivors × 4 offspring each


def test_migrate():
    pop = _make_pop(16, 4)
    rng = random.Random(42)
    migrations = migrate(pop, 4, 1, rng)
    assert len(migrations) > 0


def test_mutate_prefix_valid():
    rng = random.Random(42)
    prefix = "3+5=8\n7-2=5\n1+1=2\n"
    for _ in range(20):
        mutated = mutate_prefix(prefix, rng)
        lines = [l for l in mutated.strip().split("\n") if l.strip()]
        assert len(lines) >= 2
        assert len(lines) <= 8


def test_recombine_prefixes():
    rng = random.Random(42)
    a = "1+1=2\n2+2=4\n3+3=6\n"
    b = "4+4=8\n5+5=10\n6+6=12\n"
    combined = recombine_prefixes(a, b, rng)
    lines = [l for l in combined.strip().split("\n") if l.strip()]
    assert 2 <= len(lines) <= 8
