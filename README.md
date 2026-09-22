# Nano Research Lab

A benchmark-first autonomous AI research lab, forked from Andrej Karpathy's
[`autoresearch`](https://github.com/karpathy/autoresearch). The original git history,
training organism, attribution, and MIT licence are preserved.

Every meaningful change follows one path:

> hypothesis → change → controlled benchmark → evidence → keep/reject

Training completion is not a result. Every experiment is paired against its confirmed
parent under the same protocol and ends as `KEEP`, `REJECT`, `INCONCLUSIVE`, or
`INVALID`.

## Quick lifecycle demo

Requires Python 3.10+ and git; the smoke profile needs no GPU or third-party packages.

```bash
# Modify model.py, then benchmark the working-tree candidate.
python research.py experiment \
  --hypothesis "A positive quality bias improves the protected smoke score" \
  --description "Lifecycle validation"

# Confirm a provisional quick-stage winner across three seeds.
python research.py confirm exp_0001

# Re-run an exact historical candidate.
python research.py reproduce exp_0001
```

The harness snapshots the candidate as a git commit. A quick `KEEP` stays on the active
branch. `REJECT`, `INCONCLUSIVE`, and `INVALID` candidates are returned to the starting
commit, while their candidate commits remain recoverable under
`refs/nrl/experiments/<experiment_id>`.

## Real pretraining profile

The upstream single-GPU nanochat training loop is still `train.py`; protected data and
evaluation remain in `prepare.py`.

```bash
uv sync
uv run prepare.py
# Modify train.py, then:
python research.py experiment --profile pretrain --hypothesis "..."
```

The primary metric is validation bits per byte (`val_bpb`, lower is better), with the
fixed 300-second training budget inherited from upstream. Baseline results are cached
by champion commit, profile, budget, seed protocol, and protected protocol content.

## Boundaries

Mutable research code:

- `train.py`, `model.py`
- `rl.py`, `sampling.py`
- `algorithms/`

Protected control plane:

- `benchmark/`, `harness/`
- `prepare.py`, `research.py`, `lab.json`

Any candidate touching protected or unknown files is `INVALID`. Real held-out task data
should live outside this repository; only hashes and loader contracts belong under
`benchmark/heldout/`.

## Research history

- `research/experiments.jsonl` is authoritative.
- `research/leaderboard.json` is generated.
- `research/current_best.json` identifies the confirmed champion.
- `python research.py status` shows the latest result.

Each record includes parentage, commit, changed files, profile, budget, seeds,
environment, per-seed runs, aggregate metrics, deltas, validity, decision, and a
reproduction command. Confirmation is required before champion promotion.

Generate the autoresearch-style hill-climb chart from the authoritative ledger:

```bash
python research.py plot --profile chess-tactics
```

New research agents should begin with
[`docs/AGENT_QUICKSTART.md`](docs/AGENT_QUICKSTART.md). They only edit the profile's
mutable core file and submit every proposal through the same protected comparison path.

## Scope

The first milestone—the controlled experiment lifecycle—works locally with the smoke
profile. REINFORCE, GRPO, KLPO, and FlashREINFORCE reference interfaces are included so
the same harness can next be connected to a small verifiable-reward reasoning task.
They are baselines, not presumed winners.

Protected adapter specifications are also included for Minecraft Ender Dragon,
Minecraft progression, chess engine play, chess tactics, and 9x9 Go:

```bash
python research.py template list
python research.py template show minecraft_ender_dragon
python research.py template validate
```

These specifications fix budgets, seeds/opponents, primary metrics, paired evaluation,
artifacts and anti-gaming rules. Minecraft, full chess play and Go remain
`adapter-required`. Chess tactics is runnable now:

```bash
# Baseline evaluator
NRL_SEED=101 NRL_BUDGET='{"positions":64}' \
  python benchmark/chess_tactics_eval.py

# Modify chess_agent.py, then run the controlled comparison
python research.py experiment --profile chess-tactics \
  --algorithm chess-policy --hypothesis "..."

# Large anomalous gains are withheld until a multi-seed integrity audit
python research.py audit exp_0006
```

The current task is a deterministic suite of KQK/KRK positions with exactly one first
move forcing mate in one, two or three. This leaves a meaningful search frontier while
remaining cheap enough for rapid agent experiments. Full middlegame tactics, Stockfish
scoring and UCI match Elo remain later profiles.

See [`docs/IMPLEMENTATION_MAP.md`](docs/IMPLEMENTATION_MAP.md) for the upstream reuse
decision and [`program.md`](program.md) for the autonomous agent contract.

## Licence

MIT, matching upstream. See the upstream repository history and README attribution.
