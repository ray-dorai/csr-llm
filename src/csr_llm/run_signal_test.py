"""Signal Test: Does parent generation quality predict offspring accuracy?

This is the minimal validation experiment before building an evolutionary loop.
The fundamental question: does the self-replication channel carry heritable signal?

Protocol:
1. Pretrain a base model on single-digit arithmetic.
2. Create N parent variants by fine-tuning base on varying amounts of correct
   arithmetic data (10 → 5000 examples, 3 seeds each = 24 parents).
   More data → higher parent accuracy → better generation quality.
3. Each parent generates 500 examples via batched independent generation.
4. For each parent, train two offspring from frozen base + LoRA:
   - Offspring_gen: LoRA trained on parent's generated valid examples
   - Offspring_rnd: LoRA trained on same-count perfectly correct random examples
5. Evaluate both offspring on held-out test set.
6. Report: Pearson r(parent_correct_rate, offspring_gen_score) and
   mean delta (offspring_gen - offspring_rnd).

Interpretation:
  r > 0.3 AND delta > 0 consistently → channel has signal, build evolution.
  r ≈ 0 OR delta ≤ 0            → fundamental redesign needed.

Cost: ~24 parents × ~90s each ≈ 35 min ≈ $0.23 on A40 @ $0.40/hr.
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
from csr_llm.generate import generate_and_parse
from csr_llm.lora import apply_lora, lora_param_count
from csr_llm.model import TinyGPT, create_model, load_model, save_model
from csr_llm.pretrain import pretrain
from csr_llm.task import ParsedExample, generate_test_set, generate_training_corpus
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
    """Training examples for LoRA fine-tuning, packed like pretraining."""

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
                chunk = [bos] + all_ids[start: start + chunk_size] + [eos]
                if len(chunk) < max_len:
                    chunk = chunk + [pad] * (max_len - len(chunk))
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s[:-1], s[1:]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def finetune_parent(base_ckpt: Path, train_examples: list[str], tokenizer,
                    cfg: dict, device: str, n_steps: int = 500) -> TinyGPT:
    """Full fine-tune of base model to create a parent variant."""
    model = load_model(base_ckpt, cfg).to(device)
    dataset = ArithmeticLoRADataset(train_examples, tokenizer, cfg["model"]["max_seq_len"])
    if len(dataset) == 0:
        return model

    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    model.train()
    step = 0
    while step < n_steps:
        for x, y in loader:
            if step >= n_steps:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
    return model


def train_lora_offspring(base_ckpt: Path, train_examples: list[str], tokenizer,
                         cfg: dict, device: str,
                         lora_rank: int = 8, n_steps: int = 300) -> TinyGPT:
    """Train base + LoRA adapter on given examples. Base weights stay frozen."""
    model = load_model(base_ckpt, cfg).to(device)
    model = apply_lora(model, rank=lora_rank, alpha=float(lora_rank * 2))

    dataset = ArithmeticLoRADataset(train_examples, tokenizer, cfg["model"]["max_seq_len"])
    if len(dataset) == 0:
        return model

    trainable = [p for p in model.parameters() if p.requires_grad]
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.0)

    model.train()
    step = 0
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
            step += 1
    return model


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/signal_test.yaml")
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

    # ── 2. Build test set and random pool ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2: Test set and random data pool")
    logger.info("=" * 60)

    tcfg = cfg["task"]
    scfg = cfg["signal_test"]
    ops = tcfg["operations"]
    op_range = tuple(tcfg["operand_range"])

    test_set = generate_test_set(tcfg["test_set_size"], ops, op_range, seed=tcfg["test_set_seed"])

    # Large pool of correct random examples for the random baseline
    random_pool = generate_training_corpus(n=10000, operations=ops, operand_range=op_range, seed=99999)
    rng_pool = random.Random(42)

    base_model = load_model(base_ckpt, cfg).to(device)
    base_score = evaluate_offspring(base_model, tokenizer, test_set, device=device)
    logger.info(f"Base model score: {base_score['correct']}/{base_score['total']}")
    del base_model
    torch.cuda.empty_cache()

    # ── 3. Create parent variants ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3: Creating parent variants")
    logger.info("=" * 60)

    parent_training_sizes = scfg["parent_training_sizes"]
    parent_seeds = scfg["parent_seeds"]
    prefix = cfg["generation"]["initial_prefix"]
    gen_temp = scfg["generation_temp"]
    n_gen = scfg["n_gen_examples"]
    parent_n_steps = scfg["parent_n_steps"]
    lora_rank = scfg["lora_rank"]
    lora_n_steps = scfg["lora_n_steps"]

    results = []
    trial_idx = 0
    total_trials = len(parent_training_sizes) * len(parent_seeds)
    t0 = time.time()

    # Full corpus to sample parent training data from
    full_corpus = generate_training_corpus(n=10000, operations=ops, operand_range=op_range, seed=1)

    for n_train in parent_training_sizes:
        for seed in parent_seeds:
            trial_idx += 1
            t_trial = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"Trial {trial_idx}/{total_trials}: n_train={n_train} seed={seed}")
            logger.info(f"{'='*60}")

            rng = random.Random(seed)
            parent_data = rng.sample(full_corpus, min(n_train, len(full_corpus)))

            # Fine-tune parent
            parent = finetune_parent(base_ckpt, parent_data, tokenizer, cfg,
                                     device, n_steps=parent_n_steps)

            # Score parent
            parent_score = evaluate_offspring(parent, tokenizer, test_set, device=device)
            logger.info(f"Parent score: {parent_score['correct']}/{parent_score['total']}")

            # Generate examples from parent
            raw, parsed, gen_summary = generate_and_parse(
                model=parent,
                tokenizer=tokenizer,
                prefix=prefix,
                n_examples=n_gen,
                max_tokens_per_example=cfg["generation"]["max_tokens_per_example"],
                temperature=gen_temp,
                device=device,
            )
            valid_examples = [e.raw for e in parsed if e.is_valid]
            n_valid = len(valid_examples)
            n_correct = gen_summary["correct"]
            logger.info(f"Generated: {n_correct} correct / {n_valid} valid / {n_gen} attempts")

            del parent
            torch.cuda.empty_cache()

            # Train offspring_gen: LoRA on parent's generated valid examples
            if n_valid > 0:
                offspring_gen = train_lora_offspring(
                    base_ckpt, valid_examples, tokenizer, cfg,
                    device, lora_rank=lora_rank, n_steps=lora_n_steps
                )
                score_gen = evaluate_offspring(offspring_gen, tokenizer, test_set, device=device)
                del offspring_gen
                torch.cuda.empty_cache()
            else:
                score_gen = {"correct": 0, "total": len(test_set)}
            logger.info(f"Offspring_gen score: {score_gen['correct']}/{score_gen['total']}")

            # Train offspring_rnd: LoRA on same-count random correct examples
            n_rnd = max(n_valid, 1)
            rnd_examples = rng_pool.sample(random_pool, min(n_rnd, len(random_pool)))
            offspring_rnd = train_lora_offspring(
                base_ckpt, rnd_examples, tokenizer, cfg,
                device, lora_rank=lora_rank, n_steps=lora_n_steps
            )
            score_rnd = evaluate_offspring(offspring_rnd, tokenizer, test_set, device=device)
            del offspring_rnd
            torch.cuda.empty_cache()
            logger.info(f"Offspring_rnd score: {score_rnd['correct']}/{score_rnd['total']}")

            delta = score_gen["correct"] - score_rnd["correct"]
            trial_time = time.time() - t_trial

            result = {
                "trial": trial_idx,
                "n_train": n_train,
                "seed": seed,
                "parent_score": parent_score["correct"],
                "parent_correct_rate": parent_score["correct"] / parent_score["total"],
                "gen_valid": n_valid,
                "gen_correct": n_correct,
                "gen_correct_rate": n_correct / max(n_valid, 1),
                "offspring_gen_score": score_gen["correct"],
                "offspring_rnd_score": score_rnd["correct"],
                "delta": delta,
                "trial_sec": round(trial_time, 1),
            }
            results.append(result)

            logger.info(
                f"Delta (gen - rnd): {delta:+d} | "
                f"Time: {trial_time:.0f}s | "
                f"Elapsed: {(time.time()-t0)/60:.1f}min"
            )

    # ── 4. Analysis ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Analysis")
    logger.info("=" * 60)

    parent_rates = [r["parent_correct_rate"] for r in results]
    offspring_gen_scores = [r["offspring_gen_score"] for r in results]
    deltas = [r["delta"] for r in results]

    pearson_r = _pearson(parent_rates, offspring_gen_scores)
    mean_delta = statistics.mean(deltas)
    pct_positive = sum(1 for d in deltas if d > 0) / len(deltas) * 100

    analysis = {
        "n_trials": len(results),
        "base_score": base_score["correct"],
        "pearson_r_parent_rate_vs_offspring_gen": round(pearson_r, 4),
        "mean_delta_gen_minus_rnd": round(mean_delta, 2),
        "pct_trials_gen_beats_rnd": round(pct_positive, 1),
        "mean_offspring_gen": round(statistics.mean(offspring_gen_scores), 1),
        "mean_offspring_rnd": round(statistics.mean([r["offspring_rnd_score"] for r in results]), 1),
        "total_gpu_min": round((time.time() - t0) / 60, 1),
        "total_cost_usd": round((time.time() - t0) / 3600 * 0.40, 3),
        "verdict": _verdict(pearson_r, mean_delta, pct_positive),
    }

    output = {"analysis": analysis, "trials": results}
    with open(out_dir / "signal_test_results.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Pearson r (parent_rate → offspring_gen): {analysis['pearson_r_parent_rate_vs_offspring_gen']:.3f}")
    logger.info(f"Mean delta (gen - rnd):                  {analysis['mean_delta_gen_minus_rnd']:+.1f}")
    logger.info(f"% trials gen beats rnd:                  {analysis['pct_trials_gen_beats_rnd']:.0f}%")
    logger.info(f"Mean offspring_gen score:                {analysis['mean_offspring_gen']:.1f}/500")
    logger.info(f"Mean offspring_rnd score:                {analysis['mean_offspring_rnd']:.1f}/500")
    logger.info(f"Cost: ${analysis['total_cost_usd']:.3f} ({analysis['total_gpu_min']:.0f} min)")
    logger.info("-" * 60)
    logger.info(f"VERDICT: {analysis['verdict']}")
    logger.info("=" * 60)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _verdict(r: float, mean_delta: float, pct_positive: float) -> str:
    if r > 0.3 and mean_delta > 0 and pct_positive > 55:
        return "SIGNAL EXISTS — build evolutionary loop"
    elif r > 0.1 or mean_delta > 0:
        return "WEAK SIGNAL — investigate before building evolution"
    else:
        return "NO SIGNAL — fundamental redesign needed"


if __name__ == "__main__":
    main()
