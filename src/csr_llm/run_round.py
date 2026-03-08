"""Run one complete round of the CSR loop.

One round:
1. Each parent generates training examples
2. Each offspring is trained on its parent's examples
3. Each offspring is evaluated on the held-out test set
4. Selection: keep top performers per island
5. Reproduction: create next generation
6. Migration: move best between islands
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path

import torch

from csr_llm.config import get_round_dir, get_temperature_for_round
from csr_llm.evaluate import evaluate_offspring, save_score
from csr_llm.generate import generate_and_parse, save_generation
from csr_llm.selection import (
    Individual,
    migrate,
    reproduce,
    save_round_state,
    select_survivors,
)
from csr_llm.task import Problem
from csr_llm.train_offspring import train_offspring

logger = logging.getLogger(__name__)


def run_round(
    population: list[Individual],
    test_set: list[Problem],
    base_checkpoint: str | Path,
    tokenizer,
    cfg: dict,
    round_num: int,
    rng,
) -> tuple[list[Individual], dict]:
    """Execute one full round of CSR.

    Args:
        population: Current generation of models
        test_set: Held-out evaluation problems
        base_checkpoint: Path to base model weights
        tokenizer: The tokenizer
        cfg: Full config
        round_num: Current round number
        rng: Random number generator

    Returns:
        next_population: The next generation
        summary: Round summary dict
    """
    round_dir = get_round_dir(cfg, round_num)
    device = cfg["experiment"]["device"]
    gen_cfg = cfg["generation"]
    evo_cfg = cfg["evolution"]

    temperature = get_temperature_for_round(cfg, round_num)

    t_start = time.time()
    all_scores = []

    logger.info(f"=== Round {round_num} | Pop: {len(population)} | Temp: {temperature} ===")

    # --- STEP 1 + 2 + 3: Generate, Train, Evaluate for each model ---
    for i, ind in enumerate(population):
        t_model = time.time()

        # STEP 1: Generate training examples
        raw, parsed, gen_summary = generate_and_parse(
            model=ind.model,
            tokenizer=tokenizer,
            prefix=ind.prefix,
            n_examples=gen_cfg["n_examples"],
            max_tokens_per_example=gen_cfg["max_tokens_per_example"],
            temperature=temperature,
            device=device,
        )

        ind.generation_stats = gen_summary
        save_generation(ind.model_id, round_dir, raw, parsed, gen_summary, ind.prefix)

        # STEP 2: Train offspring
        offspring = train_offspring(
            base_checkpoint=base_checkpoint,
            parsed_examples=parsed,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
        )

        # STEP 3: Evaluate offspring
        score = evaluate_offspring(
            model=offspring,
            tokenizer=tokenizer,
            test_set=test_set,
            device=device,
            max_new_tokens=cfg["evaluation"]["max_new_tokens"],
        )

        ind.fitness = score["correct"]
        all_scores.append(score["correct"])

        metadata = {
            "round": round_num,
            "island": ind.island,
            "parent_id": ind.parent_id,
            "creation_method": ind.creation_method,
            "mutation_sigma": ind.mutation_sigma,
            "generation_stats": gen_summary,
            "survived": False,  # Updated after selection
            "rank": 0,
        }
        save_score(ind.model_id, round_dir, score, metadata)

        # Free offspring from GPU
        del offspring
        torch.cuda.empty_cache()

        elapsed = time.time() - t_model
        logger.info(
            f"  [{i+1}/{len(population)}] {ind.model_id} "
            f"gen={gen_summary['correct']}/{gen_summary['valid']} "
            f"score={score['correct']}/{score['total']} "
            f"({elapsed:.1f}s)"
        )

    # --- STEP 4: Select ---
    survivors = select_survivors(
        population,
        n_islands=evo_cfg["n_islands"],
        survivors_per_island=evo_cfg["survivors_per_island"],
    )

    survivor_ids = {s.model_id for s in survivors}
    for ind in population:
        if ind.model_id in survivor_ids:
            # Update the saved score file
            score_path = round_dir / "scores" / f"{ind.model_id}.json"
            if score_path.exists():
                with open(score_path) as f:
                    data = json.load(f)
                data["survived"] = True
                with open(score_path, "w") as f:
                    json.dump(data, f, indent=2)

    # --- STEP 5: Reproduce ---
    next_population = reproduce(
        survivors=survivors,
        offspring_per_survivor=evo_cfg["offspring_per_survivor"],
        offspring_distribution=evo_cfg["offspring_distribution"],
        mutation_cfg=evo_cfg["mutation"],
        round_num=round_num,
        rng=rng,
    )

    # --- STEP 6: Migrate ---
    migrations = migrate(
        next_population,
        n_islands=evo_cfg["n_islands"],
        n_migrants=evo_cfg["migration"]["n_migrants_per_round"],
        rng=rng,
    )

    # --- Save state ---
    save_round_state(
        population,
        round_dir,
        round_num,
        save_weights=cfg["checkpointing"]["save_parent_weights"],
    )

    # --- Build summary ---
    t_end = time.time()
    duration = t_end - t_start

    gen_correct = [ind.generation_stats.get("correct", 0) for ind in population]
    gen_valid = [ind.generation_stats.get("valid", 0) for ind in population]

    summary = {
        "round": round_num,
        "timestamp_start": t_start,
        "timestamp_end": t_end,
        "duration_seconds": round(duration, 1),
        "gpu_hours": round(duration / 3600, 2),
        "generation_stats": {
            "mean_valid_lines": round(statistics.mean(gen_valid), 1),
            "median_valid_lines": round(statistics.median(gen_valid), 1),
            "min_valid_lines": min(gen_valid),
            "max_valid_lines": max(gen_valid),
            "mean_correct_lines": round(statistics.mean(gen_correct), 1),
            "median_correct_lines": round(statistics.median(gen_correct), 1),
            "min_correct_lines": min(gen_correct),
            "max_correct_lines": max(gen_correct),
        },
        "offspring_scores": {
            "mean": round(statistics.mean(all_scores), 1),
            "median": round(statistics.median(all_scores), 1),
            "std": round(statistics.stdev(all_scores), 1) if len(all_scores) > 1 else 0,
            "min": min(all_scores),
            "max": max(all_scores),
            "best_model_id": max(population, key=lambda x: x.fitness).model_id,
            "worst_model_id": min(population, key=lambda x: x.fitness).model_id,
        },
        "selection": {
            "survivors": [
                {
                    "model_id": s.model_id,
                    "island": s.island,
                    "fitness": s.fitness,
                    "generation_correct": s.generation_stats.get("correct", 0),
                    "prefix_hash": s.prefix_hash,
                    "lineage": s.lineage,
                }
                for s in survivors
            ],
            "migrations": migrations,
        },
        "diversity_metrics": {
            "unique_prefix_hashes": len(set(ind.prefix_hash for ind in population)),
            "island_mean_scores": [
                round(
                    statistics.mean(
                        [ind.fitness for ind in population if ind.island == isl] or [0]
                    ),
                    1,
                )
                for isl in range(evo_cfg["n_islands"])
            ],
        },
    }

    with open(round_dir / "round_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"  Round {round_num} complete: "
        f"best={summary['offspring_scores']['max']} "
        f"mean={summary['offspring_scores']['mean']} "
        f"duration={duration:.0f}s"
    )

    return next_population, summary
