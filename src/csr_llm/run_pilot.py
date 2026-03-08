"""Run the $10 pilot experiment (5 rounds with go/no-go checks).

Usage:
    python -m csr_llm.run_pilot --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch

from csr_llm.config import get_output_dir, load_config, save_config
from csr_llm.evaluate import evaluate_offspring
from csr_llm.model import create_model, load_model, mutate_model
from csr_llm.pretrain import pretrain
from csr_llm.run_round import run_round
from csr_llm.selection import Individual, assign_islands
from csr_llm.task import generate_test_set, load_test_set, save_test_set
from csr_llm.tokenizer import load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CSR-LLM Pilot Experiment")
    parser.add_argument("--config", type=str, default="configs/pilot.yaml")
    parser.add_argument("--resume-round", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = get_output_dir(cfg)
    save_config(cfg, out_dir / "config")

    seed = cfg["experiment"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = cfg["experiment"]["device"]

    # ── Phase 1: Pretrain base model ──────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1: Pretraining base model")
    logger.info("=" * 60)

    base_dir = out_dir / "base_model"
    base_ckpt = base_dir / "checkpoint" / "model.pt"

    if base_ckpt.exists() and args.resume_round > 0:
        logger.info(f"Reusing existing base model: {base_ckpt}")
    else:
        pretrain(cfg)

    tokenizer = load_tokenizer(base_dir / "tokenizer.json")

    # Validate base model
    logger.info("Validating base model...")
    val = _validate_base(base_ckpt, tokenizer, cfg)
    with open(base_dir / "validation_stats.json", "w") as f:
        json.dump(val, f, indent=2)
    logger.info(f"Base model: {val['valid_rate']:.0%} valid, {val['correct_rate']:.0%} correct")

    if val["valid_rate"] < 0.3:
        logger.error("Base model valid rate < 30%. Needs more pretraining. Aborting.")
        sys.exit(1)

    # ── Phase 2: Test set ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2: Held-out test set")
    logger.info("=" * 60)

    test_path = out_dir / "test_set.json"
    tcfg = cfg["task"]
    ops, op_range = _get_task_params(tcfg)

    if test_path.exists() and args.resume_round > 0:
        test_set = load_test_set(test_path)
    else:
        test_set = generate_test_set(tcfg["test_set_size"], ops, tuple(op_range), tcfg["test_set_seed"])
        save_test_set(test_set, test_path)

    # Baseline score
    base_model = load_model(base_ckpt, cfg).to(device)
    base_score = evaluate_offspring(base_model, tokenizer, test_set, device=device)
    logger.info(f"Base model score: {base_score['correct']}/{base_score['total']}")
    del base_model
    torch.cuda.empty_cache()

    # ── Phase 3: Initialize population ────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3: Initializing population")
    logger.info("=" * 60)

    evo = cfg["evolution"]
    base = load_model(base_ckpt, cfg)
    prefix = cfg["generation"]["initial_prefix"]

    population = []
    for i in range(evo["population_size"]):
        sigma = evo["mutation"]["sigma_small"]
        model = mutate_model(base, sigma=sigma, seed=rng.randint(0, 2**31))
        population.append(
            Individual(
                model_id=f"model_{i:03d}",
                island=0,
                model=model,
                prefix=prefix,
                creation_method="init_mutant",
                mutation_sigma=sigma,
            )
        )

    assign_islands(population, evo["n_islands"])
    del base
    logger.info(f"Population: {len(population)} models, {evo['n_islands']} islands")

    # ── Phase 4: Run rounds ───────────────────────────────────
    summaries = []
    t0 = time.time()
    go_cfg = cfg.get("go_no_go", {})

    for round_num in range(1, evo["n_rounds"] + 1):
        if round_num < args.resume_round:
            continue

        logger.info("=" * 60)
        logger.info(f"ROUND {round_num}/{evo['n_rounds']}")
        logger.info("=" * 60)

        population, summary = run_round(
            population, test_set, base_ckpt, tokenizer, cfg, round_num, rng
        )
        summaries.append(summary)

        # Go/no-go checks
        if round_num == 1 and "round_1" in go_cfg:
            _check_r1(summary, go_cfg["round_1"], base_score["correct"])
        if round_num == 3 and "round_3" in go_cfg and len(summaries) >= 3:
            _check_r3(summaries, go_cfg["round_3"])
        if round_num == 5 and "round_5" in go_cfg and len(summaries) >= 5:
            _check_r5(summaries, go_cfg["round_5"])

    # ── Phase 5: Final summary ────────────────────────────────
    total_sec = time.time() - t0
    exp_summary = _build_summary(cfg, summaries, base_score["correct"], total_sec)

    with open(out_dir / "experiment_summary.json", "w") as f:
        json.dump(exp_summary, f, indent=2)

    _print_report(exp_summary)


# ── Helpers ───────────────────────────────────────────────────


def _get_task_params(tcfg):
    if "difficulty_schedule" in tcfg:
        first = tcfg["difficulty_schedule"][0]
        return first["operations"], first["operand_range"]
    return tcfg["operations"], tcfg["operand_range"]


def _validate_base(ckpt, tokenizer, cfg):
    from csr_llm.generate import generate_and_parse

    model = load_model(ckpt, cfg)
    prefix = cfg["generation"]["initial_prefix"]
    raw, parsed, summary = generate_and_parse(
        model, tokenizer, prefix, n_examples=100, temperature=1.0, device=cfg["experiment"]["device"]
    )
    del model
    torch.cuda.empty_cache()
    return {"sample_output": raw[:500], **summary}


def _check_r1(s, checks, base_score):
    issues = []
    gen = s["generation_stats"]
    sc = s["offspring_scores"]
    if gen["mean_valid_lines"] / 100 < checks.get("min_valid_generation_rate", 0.5):
        issues.append(f"Low valid rate: {gen['mean_valid_lines']}/100")
    if sc["std"] < checks.get("min_score_std", 10):
        issues.append(f"Low variance: std={sc['std']}")
    if sc["max"] < base_score:
        issues.append(f"Best ({sc['max']}) below base ({base_score})")
    if issues:
        logger.warning("⚠️  ROUND 1 ISSUES: " + "; ".join(issues))
    else:
        logger.info("✅ Round 1 checks passed")


def _check_r3(summaries, checks):
    s1, s3 = summaries[0], summaries[2]
    issues = []
    if checks.get("require_best_score_increase") and s3["offspring_scores"]["max"] <= s1["offspring_scores"]["max"]:
        issues.append("Best score flat/declining")
    if checks.get("require_mean_score_increase") and s3["offspring_scores"]["mean"] <= s1["offspring_scores"]["mean"]:
        issues.append("Mean score flat/declining")
    if checks.get("require_generation_quality_increase") and s3["generation_stats"]["mean_correct_lines"] <= s1["generation_stats"]["mean_correct_lines"]:
        issues.append("Generation quality flat/declining")
    if issues:
        logger.warning("⚠️  ROUND 3 ISSUES: " + "; ".join(issues))
    else:
        logger.info("✅ Round 3 checks passed")


def _check_r5(summaries, checks):
    bests = [s["offspring_scores"]["max"] for s in summaries]
    imp = (bests[-1] - bests[0]) / max(1, bests[0]) * 100
    threshold = checks.get("min_improvement_pct", 10)
    if imp < threshold:
        logger.warning(f"⚠️  ROUND 5: Only {imp:.1f}% improvement (need {threshold}%). DO NOT PROCEED.")
    else:
        logger.info(f"✅ Round 5: {imp:.1f}% improvement. PROCEED to main run.")


def _build_summary(cfg, summaries, base_score, total_sec):
    bests = [s["offspring_scores"]["max"] for s in summaries]
    means = [s["offspring_scores"]["mean"] for s in summaries]

    proceed = len(bests) >= 2 and bests[-1] > bests[0] and means[-1] > means[0] and bests[-1] > base_score

    return {
        "experiment_id": cfg["experiment"]["name"],
        "total_rounds": len(summaries),
        "total_gpu_hours": round(total_sec / 3600, 2),
        "total_cost_usd": round(total_sec / 3600 * 0.40, 2),
        "base_model_score": base_score,
        "trajectory": {
            "best_per_round": bests,
            "mean_per_round": means,
            "median_per_round": [s["offspring_scores"]["median"] for s in summaries],
            "worst_per_round": [s["offspring_scores"]["min"] for s in summaries],
        },
        "generation_trajectory": {
            "mean_correct_per_round": [s["generation_stats"]["mean_correct_lines"] for s in summaries],
            "mean_valid_per_round": [s["generation_stats"]["mean_valid_lines"] for s in summaries],
        },
        "pilot_decision": {
            "proceed": proceed,
            "best_improvement_pct": round((bests[-1] - bests[0]) / max(1, bests[0]) * 100, 1) if bests else 0,
            "recommendation": "PROCEED to main run" if proceed else "STOP and redesign",
        },
    }


def _print_report(s):
    logger.info("=" * 60)
    logger.info("FINAL REPORT")
    logger.info("=" * 60)
    logger.info(f"Rounds:     {s['total_rounds']}")
    logger.info(f"GPU hours:  {s['total_gpu_hours']}")
    logger.info(f"Cost:       ${s['total_cost_usd']}")
    logger.info(f"Base score: {s['base_model_score']}")
    logger.info(f"Best trajectory: {s['trajectory']['best_per_round']}")
    logger.info(f"Mean trajectory: {s['trajectory']['mean_per_round']}")
    logger.info(f"Gen quality:     {s['generation_trajectory']['mean_correct_per_round']}")
    logger.info("-" * 60)
    d = s["pilot_decision"]
    logger.info(f"Improvement: {d['best_improvement_pct']}%")
    logger.info(f"Decision:    {d['recommendation']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
