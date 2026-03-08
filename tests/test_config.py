"""Tests for config loading."""

import tempfile
from pathlib import Path

import yaml

from csr_llm.config import load_config, get_difficulty_for_round


def _write_config(tmp, cfg):
    p = Path(tmp) / "test.yaml"
    with open(p, "w") as f:
        yaml.dump(cfg, f)
    return str(p)


MINIMAL_CFG = {
    "experiment": {"name": "test", "seed": 42, "device": "cpu", "output_dir": "/tmp/test"},
    "model": {"architecture": "gpt2", "n_layers": 2, "n_heads": 2, "d_model": 32,
              "d_ff": 64, "vocab_size": 64, "max_seq_len": 16, "dropout": 0.0},
    "tokenizer": {"type": "bpe", "vocab_size": 64},
    "pretrain": {"n_examples": 100, "n_epochs": 1, "batch_size": 8,
                 "learning_rate": 1e-3, "weight_decay": 0, "warmup_steps": 0},
    "task": {"type": "arithmetic", "difficulty": "easy", "operations": ["+"],
             "operand_range": [0, 9], "test_set_size": 50, "test_set_seed": 42},
    "evolution": {"n_rounds": 2, "population_size": 8, "n_islands": 2,
                  "models_per_island": 4, "survivors_per_island": 2,
                  "offspring_per_survivor": 2, "offspring_distribution": {"clone": 1, "mutant_small": 1},
                  "mutation": {"sigma_small": 0.01, "sigma_large": 0.05},
                  "migration": {"n_migrants_per_round": 1, "direction": "ring"},
                  "selection": {"metric": "offspring_accuracy", "direction": "maximize"}},
    "generation": {"n_examples": 10, "max_tokens_per_example": 20, "temperature": 1.0,
                   "initial_prefix": "1+1=2\n"},
    "offspring": {"base": "base_checkpoint", "n_steps": 10, "batch_size": 4,
                  "learning_rate": 5e-4, "weight_decay": 0, "warmup_steps": 0},
    "evaluation": {"method": "greedy_completion", "max_new_tokens": 4, "temperature": 0.0},
    "checkpointing": {"save_parent_weights": False, "save_offspring_weights": False,
                      "save_generation_raw": True, "save_generation_parsed": True,
                      "save_scores_detailed": True, "save_prefixes": True,
                      "compress_old_rounds": False},
}


def test_load_valid():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_config(tmp, MINIMAL_CFG)
        cfg = load_config(path)
        assert cfg["experiment"]["name"] == "test"


def test_population_math():
    """population_size must equal n_islands * models_per_island."""
    cfg = MINIMAL_CFG.copy()
    assert cfg["evolution"]["population_size"] == cfg["evolution"]["n_islands"] * cfg["evolution"]["models_per_island"]
