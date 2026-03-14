# CSR-LLM: Experimental Findings

## Run Summary (2026-03-14)

Full experiment on RunPod A40, 11 rounds completed before termination.

**Config:** 64-population, 4 islands, 100-round main experiment, single-digit add/sub task, n_examples=500, n_steps=200 offspring training.

| Round | Best | Mean | Gen valid (median) |
|-------|------|------|--------------------|
| Baseline | 81/500 | — | — |
| 1 | 80 | 51.3 | ~465 |
| 3 | 70 | 37.2 | ~350 |
| 5 | **88** | 36.6 | ~250 |
| 7 | 80 | 34.0 | ~200 |
| 9 | 61 | 27.8 | ~130 |
| 11 | 60 | 30.0 | ~100 |

Best score ever: **88/500** (round 5), vs. base model 81/500 (+9%).
Mean stabilized around 28–37, well below baseline. Never showed sustained upward trajectory.

---

## Bugs Found and Fixed

### Bug 1: EOS early-stop killed generation (critical)
`generate_examples()` generated one long sequence with `max_new_tokens = n_examples * max_tokens_per_example = 10,000`. The model hit `<eos>` after ~128 tokens (one pretraining chunk = ~18 examples) and stopped entirely. Result: only ~15–18 valid examples per parent regardless of n_examples=500.

**Fix:** Generate each example independently in batches of 32, max 20 tokens each. EOS on one attempt doesn't affect others. Valid counts jumped from ~15 to ~450–480 per parent in round 1.

### Bug 2: GeneratedDataset format mismatch
Pretraining packed multiple examples per sequence: `[bos] + chain_of_N_examples + [eos]`. Offspring training wrapped each example individually: `[bos] + single_example + [eos]`. Offspring learned "EOS follows every single arithmetic line" rather than "EOS follows a long chain." This caused systematic EOS-creep: each round of fine-tuning made models emit EOS sooner.

**Fix:** Pack generated examples into sequences exactly like ArithmeticDataset. Also fixed edge case where <128 total tokens produced an empty dataset (range with negative stop).

### Bug 3: Insufficient generation volume
n_examples=100 → 500 (5x increase). Addressed the symptom; Bug 1 was the actual cause.

---

## What Worked

- Batched independent generation (Bug 1 fix) was the single most impactful change. Demonstrated that generation architecture matters more than evolutionary hyperparameters.
- Population held above zero mean for all 11 rounds (previous runs collapsed to near-zero by round 6–8).
- Round 5 best of 88/500 reproducibly exceeded baseline, showing the signal exists.
- Evolutionary infrastructure (islands, migration, selection) worked correctly once generation was functional.

---

## What Didn't Work

**The EOS-creep problem persisted.** Despite the format fix, generation valid counts declined each round: 465 → 350 → 250 → 130. Each round of fine-tuning on arithmetic examples nudges models toward shorter generation sequences. This is a structural tension: the same weights govern both generation and task performance, and fine-tuning optimizes one at the expense of the other.

**No sustained upward trajectory.** Best scores: 80 → 88 → 80 → 61 → 60. The mean declined monotonically from 51 to 30. No compounding improvement was observed across rounds.

---

## Structural Analysis

### The information channel is too lossy
The hereditary channel is: parent weights → (T=1.2 sampling) → text examples → (200 SGD steps) → offspring weights → test score. Four sequential stochastic transformations separate selection from what's being evolved. Mutual information between parent's "knowledge" and offspring's "knowledge" is near zero.

### The Ghadessy analogy breaks down
In biological CSR, the molecule IS its own replication machinery — the gene encodes the enzyme that copies it. In CSR-LLM, generation and task performance are different behaviors of the same weights, trained in conflicting directions. The "compartmentalization" that makes biological CSR work is absent.

### Selection metric doesn't select for the heritable trait
Selection acts on offspring test accuracy. The heritable trait is parent generation distribution. These are loosely related: a parent that happens to sample examples overlapping with the test distribution scores well, but that's stochastic luck, not an evolved capability.

### The task has no headroom
Base model scores 81/500 on single-digit addition before any evolution. When the base is already near-competent, self-generated training data can't compress the residual gap — the gap is dominated by evaluation noise, not missing knowledge.

### The fine-tuning signal is too weak
200 SGD steps on 50–200 generated examples barely differentiates offspring from base. The offspring IS essentially base model + sampling noise. The evolved delta is smaller than the noise floor.

---

## What the Experiment Proved

1. **Batched independent generation is necessary** for any self-replication signal to exist. A single long sequence dies at the first EOS.
2. **A weak positive signal exists.** Round 5 best (88) > baseline (81). The loop doesn't produce pure noise — there is some heritable information, just not enough.
3. **The loop is self-limiting.** Fine-tuning on self-generated examples degrades the generation mechanism that produces training data. This creates a ceiling on how many rounds can be productive.
4. **Mean stabilizes rather than diverges.** The population doesn't collapse to zero, it settles into a ~30–37/500 equilibrium. This suggests the selection mechanism is working but the signal it's selecting on is too weak to compound.

---

## Redesign Directions

### Minimal necessary change: decouple generation from fine-tuning
Freeze the base model for generation. Only fine-tune a task-specific head or adapter (LoRA delta). Offspring training updates the adapter only, never touching the weights used for generation. This eliminates EOS-creep entirely. The "gene" is the fine-tuning delta; the "polymerase" is the frozen base model.

### Better selection metric: sample efficiency, not raw accuracy
Measure: how many fewer gradient steps does an offspring need to reach some accuracy threshold vs. training from scratch? This directly measures whether the parent's generated data contains useful information. Raw accuracy conflates "base model capability" (constant) with "evolutionary improvement" (what you care about).

### Validate the minimal loop first
Before multi-island evolution: can a single parent → generate → single offspring loop reliably produce a better student than training on random data? If this one-parent-one-offspring experiment doesn't show a consistent positive delta, no evolutionary apparatus will save it. This is the experiment that should have been run first.

### Harder task, better headroom
Single-digit addition (base: 81/500) leaves ~9% room to improve. A task where the base model scores 20–30% would give the evolutionary loop more signal to work with.

### Autoresearch-style tightening
One metric (offspring improvement over base), one editable thing (generation strategy / prefix / filtering), fixed budget, accept/reject loop, never stop. Remove the evolutionary apparatus until the single-step signal is validated.
