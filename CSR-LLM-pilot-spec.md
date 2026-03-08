# CSR-LLM $10 PILOT EXPERIMENT SPEC
## Self-Improving Tiny Language Models via Compartmentalized Self-Replication

---

## GOAL

Spend ~$10 on RunPod to determine whether a self-replicating
evolutionary loop produces measurable improvement in 5 rounds.
If yes: proceed with $65 main run.
If no: diagnose why and redesign before spending more.


---

## HARDWARE

- 1x A40 (48GB VRAM) on RunPod @ ~$0.40/hr
- Budget: $10 = ~25 GPU-hours
- Expected pilot duration: 15-20 hours (buffer for setup/debug)


---

## MODEL

- Architecture: GPT-2 style decoder-only transformer
- Size: ~10M parameters
  - 8 layers
  - 256 hidden dim
  - 4 attention heads
  - 1024 feedforward dim
  - ~8K token BPE vocabulary (train tokenizer on task data)
- Framework: PyTorch + HuggingFace Transformers
- Why this size: trains from checkpoint in ~60-90 seconds,
  small enough to run 64 variants sequentially in ~90 minutes


---

## TASK

Simple integer arithmetic. Chosen because:
- Clear right/wrong answers (no subjective scoring)
- Even bad models occasionally generate correct examples
- Smooth fitness landscape (more correct = better)
- Easy to verify programmatically
- No ambiguity in evaluation

Task format (plain text, one per line):
```
2+3=5
14-7=7
6*3=18
20/4=5
```

Difficulty for pilot: single-digit addition and subtraction only.
```
3+5=8
9-2=7
1+8=9
7-4=3
```

Held-out test set: 500 problems, generated once, frozen for
all rounds. Balanced across all single-digit add/sub combos.


---

## BASE MODEL PREPARATION (before pilot starts)

1. Train tokenizer on arithmetic corpus (digits, operators, equals, newlines)
2. Pretrain base model on 50K arithmetic examples for ~10 epochs
   - Just enough that it can produce syntactically valid "N+M=K" strings
   - It does NOT need to get answers right
   - It just needs to output the right format most of the time
3. Save this as BASE_CHECKPOINT
4. Verify: prompt base model to generate 100 examples, count how many
   are syntactically valid (target: >80%) and correct (expect: 5-20%)
5. Time how long fine-tuning BASE_CHECKPOINT for 200 steps takes
   - This determines round timing and feasibility

Estimated time: 2-3 hours, ~$1


---

## THE SELF-REPLICATION LOOP

```
For each round R = 1..5:

  STEP 1: GENERATE (the "self-replication" step)
  -----------------------------------------------
  For each model M_i in population (i = 1..64):
    - Prompt M_i to generate 100 arithmetic examples
    - Prompt format: feed it "3+5=8\n7-2=5\n" as prefix,
      let it continue generating line by line
    - Save raw output to: logs/round_{R}/generated/model_{i}.txt
    - Parse output: extract valid "N op M = K" lines
    - Save parsed examples to: logs/round_{R}/parsed/model_{i}.txt
    - Log: total lines generated, valid lines, correct lines

  STEP 2: BIRTH (train offspring from generated data)
  ---------------------------------------------------
  For each model M_i:
    - Start from BASE_CHECKPOINT (same starting point every time)
    - Fine-tune for 200 steps on M_i's generated examples only
    - Learning rate: 5e-4, batch size: 16
    - Save offspring checkpoint: logs/round_{R}/offspring/model_{i}/
    - This is the "compartment wall" — each offspring only sees
      training data from its own parent, nothing else

  STEP 3: TEST (evaluate offspring)
  ---------------------------------
  For each offspring O_i:
    - Run on the 500 held-out test problems
    - For each problem, feed "N+M=" and check if the model's
      top-1 completion matches the correct answer
    - Score = number correct out of 500
    - Save full results: logs/round_{R}/scores/model_{i}.json
      (includes per-problem correct/incorrect)

  STEP 4: SELECT
  --------------
  Rank all 64 models by their offspring's score.
  Keep top 8 (12.5% survival rate).

  STEP 5: REPRODUCE
  -----------------
  Each of the 8 survivors produces 8 new models:
    - 1 exact clone (no changes)
    - 4 mutants: add Gaussian noise to weights (σ = 0.01)
    - 2 recombinants: for each layer, randomly pick weights
      from this parent or another random top-8 parent (50/50)
    - 1 wild mutant: larger noise (σ = 0.05)
  
  Total: 8 × 8 = 64 models for next round.

  ALSO EVOLVE THE GENERATION PROMPT (from PromptBreeder insight):
  ---------------------------------------------------------------
  Each survivor also carries a "generation prefix" — the few-shot
  examples used to prompt it to generate training data.
  
  - Clone: same prefix
  - Mutants: randomly swap/add/remove one example in the prefix
  - Recombinants: combine prefix examples from two parents
  
  This means we're evolving WHAT the model generates AND
  HOW it's prompted to generate. Both are heritable traits.
```


---

## POPULATION STRUCTURE (from FunSearch insight)

Split 64 models into 4 islands of 16:

```
Island A: models 1-16
Island B: models 17-32
Island C: models 33-48
Island D: models 49-64
```

Selection happens within islands (top 2 per island survive).
After selection, 1 random migrant moves between adjacent islands.
This preserves diversity — different islands may discover
different strategies for generating good training data.


---

## TIMING ESTIMATE

Per round:
  - Generate: 64 models × ~30 sec each = ~32 min
  - Train offspring: 64 models × ~90 sec each = ~96 min
  - Evaluate: 64 models × ~15 sec each = ~16 min
  - Selection/reproduction: ~2 min
  - Total per round: ~2.5 hours

5 rounds = ~12.5 hours
Setup + base model prep = ~3 hours
Buffer = ~5 hours
Total = ~20 hours = ~$8

Leaves ~$2 buffer for debugging.


---

## LOGGING SPECIFICATION

Everything goes under: logs/

### Directory Structure

```
logs/
├── config.json                  # Full experiment config (frozen at start)
├── base_model/
│   ├── checkpoint/              # BASE_CHECKPOINT weights
│   ├── pretrain_log.jsonl       # Loss curve during pretraining
│   ├── validation_examples.txt  # 100 examples from base model
│   └── validation_stats.json    # {valid: N, correct: N, total: N}
│
├── test_set.json                # The 500 held-out problems (frozen)
│
├── round_0/                     # Initial population (before any selection)
│   ├── population_weights/      # 64 model checkpoints (or deltas from base)
│   └── population_metadata.json # How each was created (mutation params, etc)
│
├── round_1/
│   ├── generated/
│   │   ├── model_001.txt        # Raw generation output
│   │   ├── model_002.txt
│   │   └── ...
│   ├── parsed/
│   │   ├── model_001.json       # {examples: [...], valid: N, correct: N}
│   │   ├── model_002.json
│   │   └── ...
│   ├── offspring/
│   │   ├── model_001/           # Fine-tuned offspring checkpoint
│   │   └── ...                  # (can delete after scoring to save disk)
│   ├── scores/
│   │   ├── model_001.json       # {score: N, per_problem: [...]}
│   │   ├── model_002.json
│   │   └── ...
│   ├── generation_prefixes/
│   │   ├── model_001.txt        # The few-shot prefix used for this model
│   │   └── ...
│   ├── round_summary.json       # Aggregated stats (see below)
│   └── survivors.json           # Which models survived, their lineage
│
├── round_2/
│   └── ... (same structure)
├── round_3/
├── round_4/
├── round_5/
│
└── experiment_summary.json      # Final aggregated results
```


### config.json (frozen at experiment start)

```json
{
  "experiment_id": "csr-llm-pilot-001",
  "timestamp": "2026-02-07T...",
  "budget_usd": 10,
  "hardware": "A40",
  "model": {
    "architecture": "gpt2",
    "layers": 8,
    "hidden_dim": 256,
    "heads": 4,
    "ff_dim": 1024,
    "vocab_size": 8000,
    "approx_params": "10M"
  },
  "task": {
    "type": "arithmetic",
    "difficulty": "single_digit_add_sub",
    "test_set_size": 500
  },
  "evolution": {
    "population_size": 64,
    "num_islands": 4,
    "models_per_island": 16,
    "survivors_per_island": 2,
    "total_survivors": 8,
    "offspring_per_survivor": 8,
    "offspring_breakdown": {
      "clone": 1,
      "mutant_small": 4,
      "recombinant": 2,
      "mutant_large": 1
    },
    "mutation_sigma_small": 0.01,
    "mutation_sigma_large": 0.05,
    "migration_per_round": 1
  },
  "training": {
    "base_checkpoint": "logs/base_model/checkpoint/",
    "finetune_steps": 200,
    "learning_rate": 5e-4,
    "batch_size": 16,
    "examples_per_model": 100
  },
  "rounds": {
    "pilot": 5,
    "main_run": 80
  }
}
```


### round_summary.json (one per round)

```json
{
  "round": 1,
  "timestamp_start": "...",
  "timestamp_end": "...",
  "duration_seconds": 9000,
  "gpu_hours": 2.5,
  "cumulative_cost_usd": 3.0,

  "generation_stats": {
    "mean_valid_lines": 72.3,
    "median_valid_lines": 74,
    "min_valid_lines": 12,
    "max_valid_lines": 95,
    "mean_correct_lines": 18.4,
    "median_correct_lines": 16,
    "min_correct_lines": 0,
    "max_correct_lines": 52
  },

  "offspring_scores": {
    "mean": 47.2,
    "median": 44,
    "std": 18.3,
    "min": 8,
    "max": 112,
    "p25": 31,
    "p75": 59,
    "best_model_id": "model_037",
    "worst_model_id": "model_012"
  },

  "selection": {
    "survivors": [
      {
        "model_id": "model_037",
        "island": "C",
        "offspring_score": 112,
        "num_correct_generated": 52,
        "num_valid_generated": 91,
        "generation_prefix_hash": "a3f7...",
        "lineage": ["round_0:model_003", "round_1:model_037"]
      }
    ],
    "migrants": [
      {"from_island": "C", "to_island": "D", "model_id": "model_037"}
    ]
  },

  "diversity_metrics": {
    "unique_generation_strategies": 0,
    "island_score_variance": [12.1, 15.3, 22.7, 9.8],
    "cross_island_score_variance": 18.2
  }
}
```


### scores/model_NNN.json (per model per round)

```json
{
  "model_id": "model_037",
  "round": 1,
  "island": "C",
  "parent_id": "model_003",
  "creation_method": "mutant_small",
  "mutation_sigma": 0.01,
  "generation_prefix": "3+5=8\n9-2=7\n1+8=9\n",

  "generation": {
    "raw_lines": 100,
    "valid_lines": 91,
    "correct_lines": 52,
    "incorrect_lines": 39,
    "unparseable_lines": 9,
    "sample_correct": ["4+3=7", "8-5=3", "2+6=8"],
    "sample_incorrect": ["7+5=11", "3-8=5", "9+1=1"],
    "sample_unparseable": ["the answer is", "+++", ""]
  },

  "offspring_score": {
    "correct": 112,
    "total": 500,
    "accuracy": 0.224,
    "per_operation": {
      "addition": {"correct": 78, "total": 250, "accuracy": 0.312},
      "subtraction": {"correct": 34, "total": 250, "accuracy": 0.136}
    }
  },

  "survived": true,
  "rank": 1
}
```


### experiment_summary.json (written at end)

```json
{
  "experiment_id": "csr-llm-pilot-001",
  "total_rounds": 5,
  "total_gpu_hours": 14.2,
  "total_cost_usd": 5.68,

  "trajectory": {
    "best_score_per_round":  [42, 67, 89, 112, 134],
    "mean_score_per_round":  [23, 31, 44, 52, 68],
    "median_score_per_round": [19, 28, 41, 48, 63],
    "worst_score_per_round": [2, 5, 12, 18, 24]
  },

  "generation_quality_trajectory": {
    "mean_correct_generated_per_round": [11, 18, 27, 34, 41],
    "mean_valid_generated_per_round": [68, 72, 78, 81, 84]
  },

  "pilot_checkpoints": {
    "signal_detected": true,
    "best_round1_vs_round5": "+219%",
    "variance_sufficient": true,
    "round1_score_std": 18.3,
    "recommendation": "PROCEED with main run"
  },

  "interesting_observations": [
    "Island C consistently outperformed others",
    "Models that generated fewer but more accurate examples scored higher",
    "Generation prefixes evolved toward shorter, simpler examples"
  ]
}
```


---

## GO/NO-GO CRITERIA (checked after each round)

### After Round 1 (sanity check):
- [ ] Can at least 50/64 models generate syntactically valid arithmetic?
- [ ] Is there meaningful variance in offspring scores? (std > 10)
- [ ] Is the best offspring score higher than the base model score?
- [ ] Did the round complete in under 3 hours?

If ANY of these fail: STOP. Debug before continuing.

### After Round 3 (trend check):
- [ ] Is best score in round 3 > best score in round 1?
- [ ] Is mean score in round 3 > mean score in round 1?
- [ ] Is mean correct-generated in round 3 > round 1?

If ALL three are flat or declining: STOP. Redesign.

### After Round 5 (pilot decision):
- [ ] Clear upward trend in best scores across rounds?
- [ ] Clear upward trend in mean scores across rounds?
- [ ] Generation quality improving (more correct examples over time)?

If YES to all three: PROCEED to main $65 run.
If YES to some: PROCEED with modifications.
If NO to all: DO NOT PROCEED. Analyze logs, redesign.


---

## WHAT TO LOOK FOR IN THE LOGS

### The primary signal: do scores go up?
Plot best/mean/median offspring score per round.
Upward trend = the loop is working.
Flat = selection pressure exists but isn't being amplified.
Declining = something is broken.

### The secondary signal: does generation quality improve?
Plot mean_correct_generated per round.
If parents are getting better at generating correct training
examples, that's the equivalent of the polymerase genes
evolving toward easier replication in the Ghadessy paper.

### The tertiary signal: do generation prefixes evolve?
Look at the few-shot prefixes of survivors across rounds.
Are they converging on particular patterns?
Are they getting shorter? Simpler? More diverse?
This tells you whether the "how to generate" is being
selected alongside the "what to generate."

### The diversity signal: are islands differentiating?
Compare mean scores across islands.
If one island consistently dominates, migration is working.
If all islands track together, they're not independent enough.
If one island dies (all low scores), that's fine — it'll be
repopulated by migrants.

### The failure modes to watch for:
1. COLLAPSE: All models converge to identical outputs.
   Sign: score variance drops to near zero.
   Fix: increase mutation sigma, reduce selection pressure.

2. GARBAGE IN GARBAGE OUT: Models generate nonsense,
   offspring trained on nonsense score zero, nothing improves.
   Sign: valid_lines stays very low across rounds.
   Fix: base model needs more pretraining on format.

3. CORRECT BUT USELESS: Models generate correct examples
   but they're all trivially easy (e.g., "0+0=0" repeated).
   Sign: high correct_generated but low offspring scores.
   Fix: add diversity penalty to generation, or filter duplicates.

4. OVERFITTING TO PREFIX: All survivors have the same prefix,
   losing diversity in generation strategy.
   Sign: generation_prefix_hash is identical across survivors.
   Fix: force prefix diversity in selection.


---

## MINIMAL VIABLE CODE STRUCTURE

```
csr-llm/
├── config.py          # Load/validate config.json
├── model.py           # Model definition, load/save, mutation, recombination
├── tokenizer.py       # Train and load BPE tokenizer
├── pretrain.py        # Pretrain base model on arithmetic corpus
├── generate.py        # Prompt model to generate training examples
├── parse.py           # Parse raw output into valid examples
├── train_offspring.py # Fine-tune base checkpoint on parsed examples
├── evaluate.py        # Score offspring on held-out test set
├── select.py          # Selection, reproduction, migration
├── evolve_prefix.py   # Mutate/recombine generation prefixes
├── run_round.py       # Orchestrate one full round
├── run_pilot.py       # Run 5 rounds with go/no-go checks
├── analyze.py         # Generate summaries and plots from logs
└── logs/              # All output (see directory structure above)
```

Each script is standalone and idempotent. If a round crashes
midway, you can rerun from the failed step. Every script reads
from and writes to the logs/ directory.


---

## WHAT THIS PROVES IF IT WORKS

If scores go up across 5 rounds, we've demonstrated:

1. An LLM can generate training data that, when used to train
   a fresh copy of itself, produces a functional model.

2. Selection on offspring quality creates pressure on the parent's
   ability to generate good training data.

3. This pressure, iterated, produces parents that are increasingly
   good at generating training data for their own task.

4. The loop is self-reinforcing: better generators → better
   offspring → better generators survive → even better generators.

This is the LLM equivalent of the polymerase copying its own gene.
The "gene" is the model weights. The "copying" is generating
training data and fine-tuning on it. The "compartment" is the
isolation of each model's training run. The "selection" is
offspring accuracy on the test set.

It's not perfect self-replication (external scaffolding still
handles weight updates), but the creative work — deciding WHAT
to train on — is done entirely by the model. And that's the part
that evolves.


---

## WHAT TO CHANGE FOR THE $65 MAIN RUN

If the pilot succeeds:

1. Scale to 80-100 rounds
2. Progressive difficulty: add two-digit, multiplication, division
3. Increase population to 128 if round time allows
4. Add temperature scheduling (high early → low late)
5. Save more checkpoints for post-hoc analysis
6. Track weight divergence from base model over time
7. Track which specific mutations (weight perturbations) survive
   longest across lineages — are there "genetic dynasties"?
8. Test whether final evolved models generalize to unseen
   arithmetic types they were never selected on
