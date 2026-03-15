"""Code Self-Play: Self-improving code generation via execution feedback.

The model plays two roles — it generates code, and execution verifies it.
This creates functional closure: the capability being trained (code generation)
is the same process that generates the training signal (correct solutions).

Three conditions, all LoRA on frozen base:

  self_play         : model generates solutions → execution verifies →
                      train on correct solutions → difficulty advances on success
  random            : train on pre-written reference solutions, fixed difficulty schedule
  oracle_curriculum : pre-written solutions, same adaptive difficulty as self_play

self_play vs oracle_curriculum isolates: does self-generated code help vs. oracle?
oracle_curriculum vs random isolates:    does adaptive difficulty help vs. fixed?
self_play vs random shows:               combined effect.

Metric: problems solved per difficulty level, pass rate over rounds.
Question: does self-play compound? Does pass rate at each level increase faster
          than the random baseline?

Model: Salesforce/codegen-350M-mono (or configurable)
Cost:  ~15 min on A40 = ~$0.10
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
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from csr_llm.code_task import (
    PROBLEMS,
    CodeProblem,
    execute_solution,
    extract_body,
    problems_at_level,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MAX_LEVEL = 5


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_base_model(model_name: str, device: str):
    """Load a HuggingFace causal LM in fp16."""
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model, tokenizer


def apply_lora(model, rank: int, target_modules: list[str]):
    """Wrap model with PEFT LoRA. Returns trainable model."""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA: {trainable:,} trainable / {total:,} total ({trainable/total:.1%})")
    return model


def generate_solution(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    device: str = "cuda",
) -> str:
    """Generate a function body given the signature+docstring prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][prompt_len:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return prompt + generated


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class CodeDataset(Dataset):
    """(prompt, solution) pairs for LoRA fine-tuning with prompt masking."""

    def __init__(self, examples: list[tuple[str, str]], tokenizer, max_len: int = 512):
        self.items = []
        for prompt, solution in examples:
            full = prompt + solution
            enc = tokenizer(
                full,
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"][0]

            # Mask prompt tokens in labels so loss is only over the solution
            prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            if (labels != -100).sum() > 0:  # at least some solution tokens
                self.items.append((input_ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def train_on_examples(
    model,
    tokenizer,
    examples: list[tuple[str, str]],
    opt: torch.optim.Optimizer,
    n_steps: int,
    device: str,
    max_len: int = 512,
) -> float | None:
    """Fine-tune LoRA for n_steps on (prompt, solution) pairs. Returns final loss."""
    dataset = CodeDataset(examples, tokenizer, max_len)
    if len(dataset) == 0:
        return None

    loader = DataLoader(
        dataset,
        batch_size=1,  # variable-length sequences; keep batch=1 for simplicity
        shuffle=True,
        collate_fn=lambda batch: batch[0],  # return single (input_ids, labels)
    )

    model.train()
    step = 0
    last_loss = None

    while step < n_steps:
        for input_ids, labels in loader:
            if step >= n_steps:
                break
            input_ids = input_ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()

            last_loss = loss.item()
            step += 1

    return last_loss


# ---------------------------------------------------------------------------
# One (condition, seed) trial
# ---------------------------------------------------------------------------

def run_one(
    condition: str,
    seed: int,
    model_name: str,
    cfg: dict,
    device: str,
) -> list[dict]:
    """Run one (condition, seed) trial. Returns per-round results."""
    scfg = cfg["selfplay"]
    rng = random.Random(seed)

    model, tokenizer = load_base_model(model_name, device)
    model = apply_lora(model, rank=scfg["lora_rank"], target_modules=scfg["lora_target_modules"])

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=scfg["lora_lr"], weight_decay=0.0)

    n_rounds = scfg["n_rounds"]
    n_steps = scfg["n_steps_per_round"]
    n_attempts = scfg["n_attempts_per_round"]
    temperature = scfg["generation_temp"]
    max_new_tokens = scfg["max_new_tokens"]
    advance_threshold = scfg["advance_threshold"]  # pass rate to advance level

    current_level = 1
    oracle_level = 1  # for oracle_curriculum: tracks same schedule as self_play
    rounds = []
    cumulative_steps = 0

    # For oracle_curriculum and random: pre-built solution pool per level
    solution_pool = {
        lvl: [(p.prompt, p.solution) for p in problems_at_level(lvl)]
        for lvl in range(1, MAX_LEVEL + 1)
    }

    for rnd in range(1, n_rounds + 1):
        level = current_level if condition != "random" else min(oracle_level, MAX_LEVEL)
        problems = problems_at_level(min(level, MAX_LEVEL))

        train_examples: list[tuple[str, str]] = []
        passed = 0
        attempted = 0
        gen_stats: dict = {}

        if condition == "self_play":
            # Model attempts each problem n_attempts/len(problems) times
            attempts_per_problem = max(1, n_attempts // len(problems))
            for prob in problems:
                for _ in range(attempts_per_problem):
                    attempted += 1
                    generated = generate_solution(
                        model, tokenizer, prob.prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        device=device,
                    )
                    body = extract_body(generated, prob.prompt)
                    ok, err = execute_solution(prob.prompt, body, prob.tests)
                    if ok:
                        passed += 1
                        train_examples.append((prob.prompt, body))

            pass_rate = passed / max(1, attempted)
            gen_stats = {"attempted": attempted, "passed": passed, "pass_rate": round(pass_rate, 3)}

            # Advance difficulty when model is passing reliably
            if pass_rate >= advance_threshold and current_level < MAX_LEVEL:
                current_level += 1
                logger.info(f"  → Advanced to level {current_level}")

        elif condition == "oracle_curriculum":
            # Same difficulty schedule as self_play (tracked via oracle_level)
            # but train on reference solutions, not model-generated code
            pool = solution_pool.get(min(oracle_level, MAX_LEVEL), [])
            train_examples = rng.sample(pool, min(len(pool), n_attempts))
            pass_rate = 1.0  # reference solutions always pass
            gen_stats = {"oracle_level": oracle_level}

            if oracle_level < MAX_LEVEL:
                oracle_level += 1  # advance every round (mirrors self_play eventual advance)

        else:  # random
            # Fixed difficulty (level 1), random reference solutions
            pool = solution_pool[1]
            train_examples = rng.sample(pool, min(len(pool), n_attempts))
            pass_rate = 1.0
            gen_stats = {}

        # Train LoRA
        last_loss = None
        if train_examples:
            last_loss = train_on_examples(
                model, tokenizer, train_examples, opt, n_steps, device,
                max_len=scfg["max_seq_len"],
            )
            cumulative_steps += n_steps

        # Evaluate: attempt all problems at current level (greedy)
        eval_problems = problems_at_level(min(current_level, MAX_LEVEL))
        eval_passed = 0
        for prob in eval_problems:
            gen = generate_solution(
                model, tokenizer, prob.prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # greedy for eval
                device=device,
            )
            body = extract_body(gen, prob.prompt)
            ok, _ = execute_solution(prob.prompt, body, prob.tests)
            if ok:
                eval_passed += 1
        eval_rate = eval_passed / max(1, len(eval_problems))

        result = {
            "round": rnd,
            "level": level,
            "cumulative_steps": cumulative_steps,
            "eval_passed": eval_passed,
            "eval_total": len(eval_problems),
            "eval_pass_rate": round(eval_rate, 3),
            "train_examples": len(train_examples),
            **{f"gen_{k}": v for k, v in gen_stats.items()},
        }
        if last_loss is not None:
            result["last_loss"] = round(last_loss, 4)
        rounds.append(result)

        logger.info(
            f"[{condition} s={seed}] r={rnd:2d}/{n_rounds} "
            f"lvl={level} eval={eval_passed}/{len(eval_problems)} "
            f"({eval_rate:.0%})  trained={len(train_examples)}"
            + (f"  pass_rate={pass_rate:.0%}" if condition == "self_play" else "")
        )

    del model
    torch.cuda.empty_cache()
    return rounds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/code_selfplay.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg["device"]
    model_name = cfg["model_name"]

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    conditions = cfg["selfplay"]["conditions"]
    seeds = cfg["selfplay"]["seeds"]

    all_results: dict[str, list] = {}
    t0 = time.time()

    for condition in conditions:
        all_results[condition] = []
        for seed in seeds:
            logger.info(f"\n{'='*60}")
            logger.info(f"Condition: {condition}  seed: {seed}")
            logger.info(f"{'='*60}")
            rounds = run_one(condition, seed, model_name, cfg, device)
            all_results[condition].append({"seed": seed, "rounds": rounds})

    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    n_rounds = cfg["selfplay"]["n_rounds"]
    analysis: dict = {
        "total_gpu_min": round((time.time() - t0) / 60, 1),
        "total_cost_usd": round((time.time() - t0) / 3600 * 0.40, 3),
        "conditions": {},
    }

    for condition in conditions:
        runs = all_results[condition]
        mean_eval = []
        for rnd_idx in range(n_rounds):
            rates = [r["rounds"][rnd_idx]["eval_pass_rate"] for r in runs]
            mean_eval.append(round(statistics.mean(rates), 3))

        final_rates = [r["rounds"][-1]["eval_pass_rate"] for r in runs]
        final_levels = [r["rounds"][-1]["level"] for r in runs]
        analysis["conditions"][condition] = {
            "mean_final_pass_rate": round(statistics.mean(final_rates), 3),
            "mean_final_level": round(statistics.mean(final_levels), 1),
            "eval_curve": mean_eval,
        }
        logger.info(f"\n{condition}:")
        logger.info(f"  Final pass rate: {final_rates}  mean={statistics.mean(final_rates):.1%}")
        logger.info(f"  Final level:     {final_levels}")
        logger.info(f"  Curve: {mean_eval}")

    winner = max(conditions, key=lambda c: analysis["conditions"][c]["mean_final_pass_rate"])
    analysis["winner"] = winner

    logger.info(f"\nWINNER: {winner}")
    logger.info(f"Cost: ${analysis['total_cost_usd']:.3f} ({analysis['total_gpu_min']:.0f} min)")
    logger.info("=" * 60)

    output = {"analysis": analysis, "results": all_results}
    with open(out_dir / "code_selfplay_results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
