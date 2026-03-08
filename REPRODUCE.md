# Reproducibility Guide

This document provides everything needed to reproduce the experiments
reported in the paper.

## Hardware Requirements

**Minimum**: 1x GPU with ≥16GB VRAM (tested on NVIDIA A40 48GB)
**Recommended**: 1x A40 or A100 on RunPod

The pilot experiment ($10 budget) requires approximately 15-20 GPU-hours.
The full experiment ($65 budget) requires approximately 150-180 GPU-hours.

## Software Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

```bash
# Create environment
conda create -n csr-llm python=3.10
conda activate csr-llm

# Install
pip install -e .

# Verify
pytest tests/ -v
```

## Reproducing the Pilot ($10 Experiment)

```bash
# 1. Run pilot (5 rounds, ~15-20 GPU-hours)
python -m csr_llm.run_pilot --config configs/pilot.yaml

# 2. Analyze results
python -m csr_llm.analyze --logs-dir logs/pilot-001/ --plots

# 3. Check go/no-go decision
cat logs/pilot-001/experiment_summary.json | python -m json.tool
```

Expected pilot duration: 15-20 hours on A40.
Expected cost on RunPod: ~$6-8.

## Reproducing the Full Experiment

```bash
# Only run this if the pilot's go/no-go check passed
python -m csr_llm.run_experiment --config configs/main.yaml

# Analyze
python -m csr_llm.analyze --logs-dir logs/main-001/ --plots
```

Expected duration: 150-180 hours on A40.
Expected cost on RunPod: ~$60-72.

## Resuming from Interruption

If the experiment is interrupted (GPU preemption, etc.):

```bash
# Resume from round N (reuses base model and previous round data)
python -m csr_llm.run_pilot --config configs/pilot.yaml --resume-round 3
```

## Verifying Results

### Key Metrics to Check

1. **Score trajectory**: `logs/*/experiment_summary.json` → `trajectory.best_per_round`
   Should show upward trend.

2. **Generation quality**: `trajectory.generation_trajectory.mean_correct_per_round`
   Should show improvement in number of correct examples generated.

3. **Diversity**: Check `round_summary.json` → `diversity_metrics.unique_prefix_hashes`
   Should remain >1 (no total collapse to single strategy).

### Expected Results (Pilot)

After 5 rounds with single-digit addition/subtraction:
- Round 1 best score: 30-60 / 500
- Round 5 best score: 60-150 / 500
- Improvement: >30%

These ranges are approximate. Random seeds affect exact numbers
but the upward trend should be consistent.

## Random Seeds

All experiments use the seed specified in the config file (default: 42).
The experiment is fully deterministic given the same seed, hardware,
and PyTorch version.

To verify seed reproducibility:
```bash
python -m csr_llm.run_pilot --config configs/pilot.yaml
# Compare logs/pilot-001/experiment_summary.json across two runs
```

## Log Format

See `CSR-LLM-pilot-spec.md` for complete log format documentation.
All logs are JSON for machine readability. Key files:

- `config.json` — Frozen experiment configuration
- `round_NNN/round_summary.json` — Per-round aggregated metrics
- `round_NNN/scores/model_NNN.json` — Per-model scores
- `round_NNN/parsed/model_NNN.json` — Per-model generation quality
- `experiment_summary.json` — Final results and go/no-go decision
