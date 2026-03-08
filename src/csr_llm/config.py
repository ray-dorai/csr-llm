"""Configuration loading, validation, and access."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config and validate required fields."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    """Check that required top-level keys exist."""
    required = ["experiment", "model", "task", "evolution", "generation", "offspring", "evaluation"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    evo = cfg["evolution"]
    pop = evo["population_size"]
    n_islands = evo["n_islands"]
    per_island = evo["models_per_island"]
    survivors = evo["survivors_per_island"]
    offspring = evo["offspring_per_survivor"]

    assert pop == n_islands * per_island, (
        f"population_size ({pop}) != n_islands ({n_islands}) * models_per_island ({per_island})"
    )
    assert survivors * n_islands * offspring == pop, (
        f"survivors * islands * offspring_per_survivor must equal population_size"
    )


def save_config(cfg: dict, path: str | Path) -> None:
    """Save config as both YAML and JSON for logging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path.with_suffix(".yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    with open(path.with_suffix(".json"), "w") as f:
        json.dump(cfg, f, indent=2)


def get_output_dir(cfg: dict) -> Path:
    """Get and create the output directory."""
    out = Path(cfg["experiment"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_round_dir(cfg: dict, round_num: int) -> Path:
    """Get and create directory for a specific round."""
    d = get_output_dir(cfg) / f"round_{round_num:03d}"
    d.mkdir(parents=True, exist_ok=True)
    for sub in ["generated", "parsed", "offspring", "scores", "generation_prefixes"]:
        (d / sub).mkdir(exist_ok=True)
    return d


def get_difficulty_for_round(cfg: dict, round_num: int) -> dict:
    """Get task difficulty settings for a given round."""
    task = cfg["task"]
    if "difficulty_schedule" in task:
        for entry in task["difficulty_schedule"]:
            lo, hi = entry["rounds"]
            if lo <= round_num <= hi:
                return entry
        # Fall back to last entry
        return task["difficulty_schedule"][-1]
    else:
        return {
            "difficulty": task["difficulty"],
            "operations": task["operations"],
            "operand_range": task["operand_range"],
        }


def get_temperature_for_round(cfg: dict, round_num: int) -> float:
    """Get generation temperature for a given round."""
    evo = cfg["evolution"]
    if "temperature_schedule" in evo:
        for entry in evo["temperature_schedule"]:
            lo, hi = entry["rounds"]
            if lo <= round_num <= hi:
                return entry["generation_temp"]
    return cfg["generation"].get("temperature", 1.0)
