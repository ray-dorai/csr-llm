# CSR-LLM: Compartmentalized Self-Replication for Language Models

**Self-improving tiny language models via evolutionary self-replication**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)


## Overview

CSR-LLM applies the principle of **compartmentalized self-replication** (Ghadessy et al., 2001) — originally developed for directed evolution of polymerase enzymes — to the self-improvement of language models.

In CSR-LLM, each model in a population generates its own training data. That data is used to train a fresh offspring model. The offspring's performance becomes the parent's fitness score. Models that generate better training data produce better offspring and survive to the next round. Models that generate poor training data are eliminated.


```
┌─────────────────────────────────────────────────┐
│  Parent model generates training examples       │
│              ↓                                   │
│  Offspring trained on those examples             │
│              ↓                                   │
│  Offspring evaluated on held-out test set        │
│              ↓                                   │
│  Offspring score = parent fitness                │
│              ↓                                   │
│  Top parents survive → mutate → next generation  │
└─────────────────────────────────────────────────┘
```

## Key Ideas

- **Self-replication**: Each model's "gene" is its weights. "Copying" is generating training data and fine-tuning on it. The model that generates the best training data for itself is the one that survives.
- **Compartmentalization**: Each model trains its offspring in isolation — no data sharing between lineages within a round. This maintains the genotype-phenotype link.
- **Co-evolution of prompt and weights**: Following insights from PromptBreeder (Fernando et al., 2023), we evolve the generation prompt alongside model weights, creating a two-level self-referential loop.
- **Island model**: Population is split into semi-isolated islands with occasional migration, preserving diversity (inspired by FunSearch, Romera-Paredes et al., 2023).

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/csr-llm.git
cd csr-llm

# Install
pip install -e .

# Run the $10 pilot (5 rounds, ~15-20 GPU-hours on A40)
python -m csr_llm.run_pilot --config configs/pilot.yaml

# Analyze results
python -m csr_llm.analyze --logs-dir logs/pilot-001/

# If pilot succeeds, run main experiment
python -m csr_llm.run_experiment --config configs/main.yaml
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- 1x GPU with ≥16GB VRAM (A40 recommended, T4/A10 possible with smaller population)

## Results

Null.

## Acknowledgments

This work is inspired by:
- Ghadessy, Ong & Holliger (2001). "Directed evolution of polymerase function by compartmentalized self-replication." *PNAS* 98(8), 4552-4557.
- Wu et al. (2024). "Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap." *arXiv:2401.10034*.

## License

MIT. See [LICENSE](LICENSE).
