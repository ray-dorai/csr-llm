"""Analyze experiment logs and generate plots + summary tables.

Usage:
    python -m csr_llm.analyze --logs-dir logs/pilot-001/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def load_summaries(logs_dir: Path) -> list[dict]:
    """Load all round summaries in order."""
    summaries = []
    for d in sorted(logs_dir.glob("round_*")):
        summary_path = d / "round_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summaries.append(json.load(f))
    return summaries


def print_trajectory(summaries: list[dict]) -> None:
    """Print score trajectory as ASCII chart."""
    if not summaries:
        print("No round data found.")
        return

    bests = [s["offspring_scores"]["max"] for s in summaries]
    means = [s["offspring_scores"]["mean"] for s in summaries]
    gen_correct = [s["generation_stats"]["mean_correct_lines"] for s in summaries]

    max_val = max(bests) if bests else 1
    chart_width = 50

    print("\n" + "=" * 70)
    print("SCORE TRAJECTORY (best per round)")
    print("=" * 70)
    for i, val in enumerate(bests):
        bar_len = int(val / max(1, max_val) * chart_width)
        bar = "█" * bar_len
        print(f"  R{i+1:3d} │ {bar} {val}")
    print()

    print("SCORE TRAJECTORY (mean per round)")
    print("-" * 70)
    max_mean = max(means) if means else 1
    for i, val in enumerate(means):
        bar_len = int(val / max(1, max_mean) * chart_width)
        bar = "▓" * bar_len
        print(f"  R{i+1:3d} │ {bar} {val}")
    print()

    print("GENERATION QUALITY (mean correct examples per round)")
    print("-" * 70)
    max_gen = max(gen_correct) if gen_correct else 1
    for i, val in enumerate(gen_correct):
        bar_len = int(val / max(1, max_gen) * chart_width)
        bar = "░" * bar_len
        print(f"  R{i+1:3d} │ {bar} {val}")
    print()


def print_survivor_lineages(summaries: list[dict]) -> None:
    """Show which lineages dominate across rounds."""
    print("=" * 70)
    print("SURVIVOR LINEAGES")
    print("=" * 70)

    for s in summaries:
        r = s["round"]
        survivors = s["selection"]["survivors"]
        print(f"\n  Round {r}:")
        for sv in survivors:
            lin = " → ".join(sv.get("lineage", [])[-3:]) or "(init)"
            print(f"    {sv['model_id']} island={sv['island']} "
                  f"score={sv['fitness']} gen_correct={sv.get('generation_correct', '?')} "
                  f"prefix={sv.get('prefix_hash', '?')[:6]}")
            if lin != "(init)":
                print(f"      lineage: {lin}")


def print_island_comparison(summaries: list[dict]) -> None:
    """Compare performance across islands."""
    print("\n" + "=" * 70)
    print("ISLAND COMPARISON (mean score per island per round)")
    print("=" * 70)

    for s in summaries:
        r = s["round"]
        island_scores = s["diversity_metrics"].get("island_mean_scores", [])
        print(f"  R{r:3d}: " + "  ".join(f"I{i}={v:5.1f}" for i, v in enumerate(island_scores)))


def print_generation_evolution(logs_dir: Path) -> None:
    """Show how generation prefixes evolved."""
    print("\n" + "=" * 70)
    print("PREFIX EVOLUTION (survivors' prefixes across rounds)")
    print("=" * 70)

    for d in sorted(logs_dir.glob("round_*")):
        summary_path = d / "round_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            s = json.load(f)

        r = s["round"]
        unique_hashes = s["diversity_metrics"].get("unique_prefix_hashes", "?")
        print(f"\n  Round {r} ({unique_hashes} unique prefixes):")

        # Show top survivor's prefix
        prefix_dir = d / "generation_prefixes"
        survivors = s["selection"]["survivors"]
        if survivors and prefix_dir.exists():
            best = survivors[0]
            pf = prefix_dir / f"{best['model_id']}.txt"
            if pf.exists():
                prefix_text = pf.read_text().strip()
                print(f"    Best ({best['model_id']}): {prefix_text[:80]}")


def generate_plots(summaries: list[dict], output_dir: Path) -> None:
    """Generate matplotlib plots if available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rounds = [s["round"] for s in summaries]

    # Score trajectory
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, [s["offspring_scores"]["max"] for s in summaries], "o-", label="Best", linewidth=2)
    ax.plot(rounds, [s["offspring_scores"]["mean"] for s in summaries], "s-", label="Mean", linewidth=2)
    ax.plot(rounds, [s["offspring_scores"]["median"] for s in summaries], "^-", label="Median", linewidth=1)
    ax.fill_between(
        rounds,
        [s["offspring_scores"]["min"] for s in summaries],
        [s["offspring_scores"]["max"] for s in summaries],
        alpha=0.15,
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Offspring Score (correct / 500)")
    ax.set_title("CSR-LLM: Offspring Score Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "score_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Generation quality
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, [s["generation_stats"]["mean_correct_lines"] for s in summaries], "o-", label="Mean correct", color="green")
    ax.plot(rounds, [s["generation_stats"]["mean_valid_lines"] for s in summaries], "s-", label="Mean valid", color="blue")
    ax.set_xlabel("Round")
    ax.set_ylabel("Examples per model")
    ax.set_title("CSR-LLM: Generation Quality Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "generation_quality.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Island comparison
    n_islands = len(summaries[0]["diversity_metrics"].get("island_mean_scores", []))
    if n_islands > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        for isl in range(n_islands):
            ax.plot(
                rounds,
                [s["diversity_metrics"]["island_mean_scores"][isl] for s in summaries],
                "o-",
                label=f"Island {isl}",
            )
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean Score")
        ax.set_title("CSR-LLM: Island Performance Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "island_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze CSR-LLM experiment logs")
    parser.add_argument("--logs-dir", type=str, required=True)
    parser.add_argument("--plots", action="store_true", help="Generate matplotlib plots")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    summaries = load_summaries(logs_dir)

    if not summaries:
        print(f"No round summaries found in {logs_dir}")
        sys.exit(1)

    print(f"\nLoaded {len(summaries)} rounds from {logs_dir}\n")

    print_trajectory(summaries)
    print_survivor_lineages(summaries)
    print_island_comparison(summaries)
    print_generation_evolution(logs_dir)

    # Load experiment summary if available
    exp_summary_path = logs_dir / "experiment_summary.json"
    if exp_summary_path.exists():
        with open(exp_summary_path) as f:
            exp = json.load(f)
        print("\n" + "=" * 70)
        print(f"PILOT DECISION: {exp.get('pilot_decision', {}).get('recommendation', 'N/A')}")
        print("=" * 70)

    if args.plots:
        generate_plots(summaries, logs_dir / "plots")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
