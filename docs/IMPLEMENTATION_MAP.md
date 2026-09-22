# Upstream reuse and implementation map

## Reuse from `karpathy/autoresearch`

- Preserve the complete git history, README attribution, MIT licence notice, and
  `upstream` remote.
- Keep `prepare.py` as protected data/evaluation code.
- Keep `train.py` as the real single-GPU pretraining organism. Its fixed wall-clock
  budget and `val_bpb` output remain the production pretraining benchmark.
- Keep the small-repository and agent-readable `program.md` model.

## Add in this repository

- `research.py`: the only normal command-line entry point.
- `harness/`: paired execution, cache keys, git snapshots, decisions, ledger, and
  leaderboard generation.
- `benchmark/`: protected protocol, metric semantics, evaluator, and a deterministic
  CPU smoke benchmark used to test the harness without an H100.
- `research/`: generated scientific history. JSONL is authoritative; Markdown and
  leaderboard JSON are derived views.
- `algorithms/` and `rl.py`: deliberately small reference interfaces for comparable
  REINFORCE, GRPO, KLPO, and FlashREINFORCE experiments.

## Deliberate non-goals for the first complete loop

- No scheduler, distributed system, web UI, database, or multi-agent framework.
- No claim that the CPU smoke benchmark measures language-model quality.
- No automatic promotion after one noisy screening run. Quick runs may KEEP a patch
  provisionally; confirmation is required for champion promotion.

The protected evaluator invokes an experiment entry point and parses a strict JSON
result. The production profile points at `train.py`; the smoke profile makes lifecycle
tests cheap and deterministic. Both use the same experiment record and decision path.
