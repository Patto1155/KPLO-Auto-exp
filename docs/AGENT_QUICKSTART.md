# Research-agent quickstart

This repository is a hill climb. Do not redesign the harness during a research run.

## Five files to read

1. `program.md` — research contract.
2. `lab.json` — available profiles, metrics, budgets and seeds.
3. The mutable target (`chess_agent.py`, `train.py`, or an algorithm module).
4. The matching protected evaluator under `benchmark/`—read only.
5. `research/current_best.json` and the tail of `research/experiments.jsonl`.

## One experiment

```bash
# Make one small change to chess_agent.py, then:
python research.py experiment --profile chess-tactics \
  --algorithm short-name \
  --hypothesis "One falsifiable sentence"
```

The harness snapshots the candidate, runs the confirmed parent and candidate under the
same budget, and records the comparison. A rejected or inconclusive working-tree change
is rolled back but remains recoverable under `refs/nrl/experiments/<id>`.

For a quick-stage KEEP:

```bash
python research.py confirm exp_XXXX
```

For a very large improvement withheld as anomalous:

```bash
python research.py audit exp_XXXX
```

## RL reasoning research

The mutable surface is wider here than on chess, so change one thing at a time.

- `rl.py` — optimiser, learning rate, group and batch shape, entropy bonus, gradient
  clipping, KL anchor refresh interval, and which algorithm runs.
- `algorithms/` — the objectives themselves. KLPO exposes the KL estimator
  (`exact`, `k1`, `k2`, `k3`), the KL target, and the adaptive controller.
- `model.py` — policy width, depth, initialisation.
- `sampling.py` — how training explores. Evaluation decoding is protected.

```bash
python research.py experiment --profile rl-reasoning \
  --algorithm klpo --hypothesis "One falsifiable sentence"

# Reference numbers for every algorithm under one budget and seed set
python research.py algorithms --profile rl-reasoning
```

Seed variance on this profile is large. A single-seed quick KEEP means very little
until `confirm` reproduces it.

Regenerate the hill-climb chart:

```bash
python research.py plot --profile chess-tactics
```

## Rules

- Change only files listed as mutable in `AGENTS.md`.
- One hypothesis per experiment.
- Never edit scores, seeds, evaluation data, budgets or the ledger.
- Do not keep a change because it feels elegant. Keep it only when the primary metric
  improves and the protected decision protocol accepts it.
- Read failures before repeating an idea. Negative evidence is part of the research.
