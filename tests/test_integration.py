"""Integration test: run the full CSR loop on CPU with tiny models.

This is the confidence test before spending money on RunPod.
If this passes, the pipeline works end-to-end.
"""

import json
import random
import tempfile
from pathlib import Path

import torch

from csr_llm.config import get_output_dir, get_round_dir, save_config
from csr_llm.evaluate import evaluate_offspring
from csr_llm.model import load_model, mutate_model
from csr_llm.pretrain import pretrain
from csr_llm.run_round import run_round
from csr_llm.selection import Individual, assign_islands
from csr_llm.task import generate_test_set, save_test_set
from csr_llm.tokenizer import load_tokenizer


def _tiny_config(tmp_dir: str) -> dict:
    """Minimal config that exercises the full pipeline on CPU."""
    return {
        "experiment": {
            "name": "integration-test",
            "seed": 42,
            "device": "cpu",
            "output_dir": f"{tmp_dir}/logs",
        },
        "model": {
            "architecture": "gpt2",
            "n_layers": 2,
            "n_heads": 2,
            "d_model": 32,
            "d_ff": 64,
            "vocab_size": 512,
            "max_seq_len": 64,
            "dropout": 0.0,
        },
        "tokenizer": {
            "type": "bpe",
            "vocab_size": 512,
            "min_frequency": 1,
            "special_tokens": ["<pad>", "<eos>", "<bos>"],
        },
        "pretrain": {
            "n_examples": 200,
            "n_epochs": 2,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "save_every_epoch": False,
        },
        "task": {
            "type": "arithmetic",
            "difficulty": "single_digit_add_sub",
            "operations": ["+", "-"],
            "operand_range": [0, 9],
            "test_set_size": 20,
            "test_set_seed": 12345,
        },
        "evolution": {
            "n_rounds": 2,
            "population_size": 4,
            "n_islands": 2,
            "models_per_island": 2,
            "survivors_per_island": 1,
            "offspring_per_survivor": 2,
            "offspring_distribution": {
                "clone": 1,
                "mutant_small": 1,
            },
            "mutation": {
                "sigma_small": 0.01,
                "sigma_large": 0.05,
            },
            "migration": {
                "n_migrants_per_round": 1,
                "direction": "ring",
            },
            "selection": {
                "metric": "offspring_accuracy",
                "direction": "maximize",
            },
        },
        "generation": {
            "n_examples": 10,
            "max_tokens_per_example": 15,
            "temperature": 1.0,
            "initial_prefix": "3+5=8\n9-2=7\n",
        },
        "offspring": {
            "base": "base_checkpoint",
            "n_steps": 5,
            "batch_size": 4,
            "learning_rate": 5e-4,
            "weight_decay": 0.0,
            "warmup_steps": 0,
        },
        "evaluation": {
            "method": "greedy_completion",
            "max_new_tokens": 4,
            "temperature": 0.0,
        },
        "checkpointing": {
            "save_parent_weights": False,
            "save_offspring_weights": False,
            "save_generation_raw": True,
            "save_generation_parsed": True,
            "save_scores_detailed": True,
            "save_prefixes": True,
            "compress_old_rounds": False,
        },
    }


def test_full_pipeline():
    """Run 2 rounds of CSR on CPU. If this passes, RunPod will work."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_config(tmp)
        out_dir = get_output_dir(cfg)
        save_config(cfg, out_dir / "config")

        seed = cfg["experiment"]["seed"]
        random.seed(seed)
        torch.manual_seed(seed)
        rng = random.Random(seed)
        device = cfg["experiment"]["device"]

        # --- Pretrain ---
        model, base_ckpt = pretrain(cfg)
        assert base_ckpt.exists()

        tok_path = out_dir / "base_model" / "tokenizer.json"
        assert tok_path.exists()
        tokenizer = load_tokenizer(tok_path)
        del model

        # --- Test set ---
        tcfg = cfg["task"]
        test_set = generate_test_set(
            tcfg["test_set_size"],
            tcfg["operations"],
            tuple(tcfg["operand_range"]),
            tcfg["test_set_seed"],
        )
        save_test_set(test_set, out_dir / "test_set.json")
        assert len(test_set) == 20

        # --- Baseline score ---
        base_model = load_model(base_ckpt, cfg)
        base_score = evaluate_offspring(base_model, tokenizer, test_set, device=device)
        assert base_score["total"] == 20
        assert 0 <= base_score["correct"] <= 20
        del base_model

        # --- Initialize population ---
        evo = cfg["evolution"]
        base = load_model(base_ckpt, cfg)
        prefix = cfg["generation"]["initial_prefix"]

        population = []
        for i in range(evo["population_size"]):
            m = mutate_model(base, sigma=evo["mutation"]["sigma_small"], seed=rng.randint(0, 2**31))
            population.append(
                Individual(
                    model_id=f"model_{i:03d}",
                    island=0,
                    model=m,
                    prefix=prefix,
                    creation_method="init_mutant",
                    mutation_sigma=evo["mutation"]["sigma_small"],
                )
            )
        assign_islands(population, evo["n_islands"])
        del base

        assert len(population) == 4
        assert sum(1 for ind in population if ind.island == 0) == 2
        assert sum(1 for ind in population if ind.island == 1) == 2

        # --- Run 2 rounds ---
        summaries = []
        for round_num in range(1, 3):
            population, summary = run_round(
                population, test_set, base_ckpt, tokenizer, cfg, round_num, rng
            )
            summaries.append(summary)

            # Round produced correct output structure
            round_dir = get_round_dir(cfg, round_num)
            assert (round_dir / "round_summary.json").exists()

            with open(round_dir / "round_summary.json") as f:
                saved = json.load(f)
            assert saved["round"] == round_num

            # Scores were recorded for each model
            score_files = list((round_dir / "scores").glob("model_*.json"))
            assert len(score_files) == 4

            # Generation output was saved
            gen_files = list((round_dir / "generated").glob("model_*.txt"))
            assert len(gen_files) == 4

            # Parsed output was saved
            parsed_files = list((round_dir / "parsed").glob("model_*.json"))
            assert len(parsed_files) == 4

            # Prefixes were saved
            prefix_files = list((round_dir / "generation_prefixes").glob("model_*.txt"))
            assert len(prefix_files) == 4

            # Population was correctly reproduced
            assert len(population) == 4

        # --- Verify summary structure ---
        assert len(summaries) == 2
        for s in summaries:
            assert "offspring_scores" in s
            assert "generation_stats" in s
            assert "selection" in s
            assert "diversity_metrics" in s
            assert s["offspring_scores"]["max"] >= s["offspring_scores"]["min"]
            assert s["generation_stats"]["mean_valid_lines"] >= 0
            assert len(s["selection"]["survivors"]) == 2  # 1 per island × 2 islands
