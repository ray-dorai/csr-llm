"""Run the full experiment (80-100 rounds).

Usage:
    python -m csr_llm.run_experiment --config configs/main.yaml
"""

from __future__ import annotations

# This follows the same structure as run_pilot but with more rounds
# and progressive difficulty. Import and extend.

from csr_llm.run_pilot import main

if __name__ == "__main__":
    main()
