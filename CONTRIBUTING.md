# Contributing

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/csr-llm.git
cd csr-llm
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

```bash
black src/ tests/
ruff check src/ tests/
```

## Project Structure

```
csr-llm/
├── configs/                  # Experiment configurations
│   ├── pilot.yaml           # $10 pilot (5 rounds)
│   └── main.yaml            # $65 full experiment (100 rounds)
│
├── src/csr_llm/             # Main package
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   ├── config.py            # Config loading/validation
│   ├── model.py             # TinyGPT + mutation/recombination
│   ├── tokenizer.py         # BPE tokenizer
│   ├── task.py              # Arithmetic problem gen/parse/eval
│   ├── pretrain.py          # Base model pretraining
│   ├── generate.py          # Training data generation from models
│   ├── train_offspring.py   # Fine-tune offspring on generated data
│   ├── evaluate.py          # Score offspring on test set
│   ├── selection.py         # Selection, reproduction, migration
│   ├── run_round.py         # One complete CSR round
│   ├── run_pilot.py         # 5-round pilot with go/no-go
│   ├── run_experiment.py    # Full experiment runner
│   └── analyze.py           # Log analysis and plotting
│
├── tests/                   # Unit tests
│   ├── test_config.py
│   ├── test_model.py
│   ├── test_task.py
│   └── test_selection.py
│
├── paper/                   # LaTeX source (if applicable)
├── CSR-LLM-pilot-spec.md   # Detailed pilot specification
├── REPRODUCE.md             # Reproducibility instructions
├── README.md
├── LICENSE
├── pyproject.toml
└── .gitignore
```

## Adding New Tasks

To add a task beyond arithmetic:

1. Implement problem generation in `task.py` (or a new file)
2. Implement parsing for model-generated output
3. Implement evaluation scoring
4. Add a config in `configs/`

The evolutionary loop (`selection.py`, `run_round.py`) is task-agnostic.

## Adding New Evolutionary Operators

To add new mutation/recombination strategies:

1. Add the operator function in `model.py`
2. Register it in `selection.py`'s `reproduce()` function
3. Add the method name to the config's `offspring_distribution`
