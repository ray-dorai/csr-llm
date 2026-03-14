"""Self-Directed Curriculum Test: Does failure-driven self-correction outperform random?

Three conditions, all LoRA on frozen base, all oracle-verified labels:

  self_correct : model generates examples → oracle corrects failures → train on corrections
  self_filter  : model generates examples → oracle filters → train on correct only (STaR-style)
  random       : sample correct examples from fixed pool (no model generation)

Protocol per condition per seed:
  - 20 rounds × 100 LoRA steps = 2000 total gradient steps
  - Evaluate on 500-problem held-out test after each round
  - Oracle: Python eval() — no label noise, answers are always right

Question: which condition reaches high accuracy in fewest steps?

Prediction:
  self_correct should win early (concentrates compute on failures)
  self_filter should win late (quality improves as model improves)
  random is the flat baseline

Cost: ~9 runs × ~90s each + pretrain ≈ 15 min ≈ $0.10 on A40
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from csr_llm.config import get_output_dir, load_config
from csr_llm.evaluate import evaluate_offspring
from csr_llm.generate import generate_examples
from csr_llm.lora import apply_lora
from csr_llm.model import TinyGPT, load_model
from csr_llm.pretrain import pretrain
from csr_llm.task import generate_test_set, generate_training_corpus, parse_generated_output
from csr_llm.tokenizer import encode, load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ArithmeticLoRADataset(Dataset):
    """Pack training examples into chunked sequences for LoRA fine-tuning."""

    def __init__(self, examples: list[str], tokenizer, max_len: int = 128):
        self.samples = []
        bos = tokenizer.token_to_id("<bos>")
        eos = tokenizer.token_to_id("<eos>")
        pad = tokenizer.token_to_id("<pad>")

        all_ids = []
        for ex in examples:
            all_ids.extend(encode(tokenizer, ex.strip() + "\n"))

        if not all_ids:
            return

        chunk_size = max_len - 2
        if len(all_ids) < chunk_size:
            chunk = [bos] + all_ids + [eos]
            chunk = chunk + [pad] * (max_len - len(chunk))
            self.samples.append(torch.tensor(chunk, dtype=torch.long))
        else:
            for start in range(0, len(all_ids) - chunk_size + 1, chunk_size):
                chunk = [bos] + all_ids[start:start + chunk_size] + [eos]
                if len(chunk) < max_len:
                    chunk = chunk + [pad] * (max_len - len(chunk))
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s[:-1], s[1:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lora_steps(
    model: TinyGPT,
    examples: list[str],
    tokenizer,
    cfg: dict,
    device: str,
    opt: torch.optim.Optimizer,
    n_steps: int,
) -> float | None:
    """Run n_steps of LoRA training on examples. Returns final loss."""
    dataset = ArithmeticLoRADataset(examples, tokenizer, cfg["model"]["max_seq_len"])
    if len(dataset) == 0:
        return None

    trainable = [p for p in model.parameters() if p.requires_grad]
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True, drop_last=False)

    model.train()
    step = 0
    last_loss = None
    while step < n_steps:
        for x, y in loader:
            if step >= n_steps:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            last_loss = loss.item()
            step += 1
    return last_loss


# ---------------------------------------------------------------------------
# Example collection per condition
# ---------------------------------------------------------------------------

def collect_examples(
    condition: str,
    model: TinyGPT,
    tokenizer,
    cfg: dict,
    device: str,
    rng: random.Random,
    random_pool: list[str],
) -> tuple[list[str], dict]:
    """Collect training examples for one round.

    self_correct: oracle-corrected versions of the model's own mistakes
    self_filter:  only the model's generations that were already correct
    random:       randomly sampled correct examples from a fixed pool
    """
    ccfg = cfg["curriculum_test"]
    n_examples = ccfg["n_examples_per_round"]
    n_generate = ccfg["n_generate_per_round"]
    prefix = cfg["generation"]["initial_prefix"]
    gen_temp = ccfg["generation_temp"]
    max_tokens = cfg["generation"]["max_tokens_per_example"]

    stats = {"n_generated": 0, "n_valid": 0, "n_correct": 0, "n_failures": 0}

    if condition == "random":
        examples = rng.sample(random_pool, min(n_examples, len(random_pool)))
        return examples, stats

    raw = generate_examples(
        model, tokenizer, prefix,
        n_examples=n_generate,
        max_tokens_per_example=max_tokens,
        temperature=gen_temp,
        device=device,
    )
    parsed = parse_generated_output(raw)

    stats["n_generated"] = n_generate
    stats["n_valid"] = sum(1 for p in parsed if p.is_valid)
    stats["n_correct"] = sum(1 for p in parsed if p.is_correct)
    stats["n_failures"] = sum(1 for p in parsed if p.is_valid and not p.is_correct)

    if condition == "self_correct":
        # Oracle-corrected failures: expression is from model, answer is from oracle
        candidates = [
            f"{p.expression}={p.correct_answer}"
            for p in parsed if p.is_valid and not p.is_correct
        ]
    else:  # self_filter (STaR-style)
        candidates = [p.raw for p in parsed if p.is_correct]

    if len(candidates) > n_examples:
        candidates = rng.sample(candidates, n_examples)

    return candidates, stats


# ---------------------------------------------------------------------------
# One (condition, seed) trial
# ---------------------------------------------------------------------------

def run_one(
    condition: str,
    seed: int,
    base_ckpt: Path,
    tokenizer,
    cfg: dict,
    test_set: list,
    random_pool: list[str],
    device: str,
) -> list[dict]:
    """Run one (condition, seed) trial. Returns per-round results."""
    ccfg = cfg["curriculum_test"]
    lora_rank = ccfg["lora_rank"]
    n_rounds = ccfg["n_rounds"]
    n_steps = ccfg["n_steps_per_round"]

    rng = random.Random(seed)

    # Fresh LoRA on frozen base — base weights never change
    model = load_model(base_ckpt, cfg)
    model = apply_lora(model, rank=lora_rank, alpha=float(lora_rank * 2))
    model = model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=ccfg["lora_lr"], weight_decay=0.0)

    rounds = []
    cumulative_steps = 0

    for rnd in range(1, n_rounds + 1):
        examples, stats = collect_examples(
            condition, model, tokenizer, cfg, device, rng, random_pool
        )

        last_loss = None
        if examples:
            last_loss = train_lora_steps(model, examples, tokenizer, cfg, device, opt, n_steps)
            cumulative_steps += n_steps

        score = evaluate_offspring(model, tokenizer, test_set, device=device)

        result = {
            "round": rnd,
            "cumulative_steps": cumulative_steps,
            "score": score["correct"],
            "accuracy": round(score["accuracy"], 4),
            "n_trained": len(examples),
            **{f"gen_{k}": v for k, v in stats.items()},
        }
        if last_loss is not None:
            result["last_loss"] = round(last_loss, 4)
        rounds.append(result)

        gen_info = (
            f"valid={stats['n_valid']} corr={stats['n_correct']} fail={stats['n_failures']}"
            if condition != "random" else "pool"
        )
        logger.info(
            f"[{condition} s={seed}] r={rnd:2d}/{n_rounds} "
            f"score={score['correct']:3d}/500  trained={len(examples):3d}  {gen_info}"
        )

    return rounds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/curriculum_test.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = get_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg["experiment"]["device"]

    random.seed(cfg["experiment"]["seed"])
    torch.manual_seed(cfg["experiment"]["seed"])

    # ── 1. Pretrain base model ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1: Pretraining base model")
    logger.info("=" * 60)

    base_dir = out_dir / "base_model"
    base_ckpt = base_dir / "checkpoint" / "model.pt"
    pretrain(cfg)
    tokenizer = load_tokenizer(base_dir / "tokenizer.json")

    # ── 2. Test set + random pool ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2: Test set and data pool")
    logger.info("=" * 60)

    tcfg = cfg["task"]
    ops = tcfg["operations"]
    op_range = tuple(tcfg["operand_range"])

    test_set = generate_test_set(tcfg["test_set_size"], ops, op_range, seed=tcfg["test_set_seed"])
    random_pool = generate_training_corpus(n=10000, operations=ops, operand_range=op_range, seed=99999)

    base_model = load_model(base_ckpt, cfg).to(device)
    base_score = evaluate_offspring(base_model, tokenizer, test_set, device=device)
    logger.info(f"Base model: {base_score['correct']}/500 ({base_score['accuracy']:.1%})")
    del base_model
    torch.cuda.empty_cache()

    # ── 3. Run all conditions ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3: Curriculum conditions")
    logger.info("=" * 60)

    ccfg = cfg["curriculum_test"]
    conditions = ccfg["conditions"]
    seeds = ccfg["seeds"]

    all_results: dict[str, list] = {}
    t0 = time.time()

    for condition in conditions:
        all_results[condition] = []
        for seed in seeds:
            logger.info(f"\n{'='*60}")
            logger.info(f"Condition: {condition}  seed: {seed}")
            logger.info(f"{'='*60}")
            rounds = run_one(
                condition, seed, base_ckpt, tokenizer, cfg,
                test_set, random_pool, device,
            )
            all_results[condition].append({"seed": seed, "rounds": rounds})
            torch.cuda.empty_cache()

    # ── 4. Analysis ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Analysis")
    logger.info("=" * 60)

    n_rounds = ccfg["n_rounds"]
    analysis: dict = {
        "base_score": base_score["correct"],
        "total_gpu_min": round((time.time() - t0) / 60, 1),
        "total_cost_usd": round((time.time() - t0) / 3600 * 0.40, 3),
        "conditions": {},
    }

    for condition in conditions:
        runs = all_results[condition]
        mean_by_round = []
        for rnd_idx in range(n_rounds):
            scores = [r["rounds"][rnd_idx]["score"] for r in runs]
            mean_by_round.append(round(statistics.mean(scores), 1))

        final_scores = [r["rounds"][-1]["score"] for r in runs]
        mean_final = round(statistics.mean(final_scores), 1)
        analysis["conditions"][condition] = {
            "mean_final_score": mean_final,
            "final_scores_per_seed": final_scores,
            "mean_by_round": mean_by_round,
        }

        logger.info(f"\n{condition}:")
        logger.info(f"  Curve:  {mean_by_round}")
        logger.info(f"  Final:  {final_scores}  mean={mean_final:.1f}/500")

    winner = max(conditions, key=lambda c: analysis["conditions"][c]["mean_final_score"])
    analysis["winner"] = winner

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Base model:    {base_score['correct']}/500")
    for cond in conditions:
        cdata = analysis["conditions"][cond]
        logger.info(f"{cond:14s}: {cdata['mean_final_score']:.1f}/500  curve={cdata['mean_by_round']}")
    logger.info(f"\nWINNER: {winner}")
    logger.info(f"Cost: ${analysis['total_cost_usd']:.3f} ({analysis['total_gpu_min']:.0f} min)")
    logger.info("=" * 60)

    output = {"analysis": analysis, "results": all_results}
    with open(out_dir / "curriculum_test_results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
