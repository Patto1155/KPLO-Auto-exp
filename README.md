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

## RL reasoning profile

The second research mode is the one the harness was built for: a fixed checkpoint,
rollouts scored by a verifiable reward, RL training, and a held-out benchmark.

The task is extreme selection — given four digits, emit the smallest then the largest.
Answers are computed rather than stored, so there is no label file to read, and the
train/held-out partition is a protected hash of the prompt. Training code can only
reach the train split; `TrainingEnvironment.score` raises on a held-out problem and
meters every rollout against a fixed budget.

```bash
# Baseline evaluator (no GPU, no third-party packages)
NRL_SEED=101 NRL_BUDGET='{"rollouts":20000,"eval_problems":2048}' \
  python benchmark/rl_eval.py

# Modify rl.py, model.py, sampling.py or algorithms/, then:
python research.py experiment --profile rl-reasoning \
  --algorithm klpo --hypothesis "..."
```

The primary metric is `heldout_pass_at_1`: exact-match accuracy under greedy decoding,
on problems never scored during training. Decoding at evaluation time lives in the
protected evaluator, so a candidate cannot change how it is measured.

### Algorithm baselines

`python research.py algorithms --profile rl-reasoning` runs every algorithm from an
identical starting checkpoint, budget and seed set. Measured over eight seeds at 20k
rollouts:

| Algorithm | heldout_pass_at_1 | std | entropy | gen gap |
| --- | --- | --- | --- | --- |
| klpo | 0.2205 | 0.0710 | 0.1268 | 0.0083 |
| flashreinforce | 0.1784 | 0.0565 | 0.0248 | 0.0146 |
| grpo | 0.1733 | 0.0498 | 0.0088 | 0.0165 |
| reinforce | 0.1489 | 0.0460 | 0.0355 | 0.0103 |

Read these as the harness reports them, not as a ranking. Seed-to-seed spread is large
relative to the gaps: KLPO's margin over REINFORCE is roughly three standard errors of
the mean and is the only separation the data supports. KLPO versus FlashREINFORCE is
not a distinguishable difference at eight seeds. The one robust pattern is mechanistic
rather than ordinal — the runs that collapse entropy also generalise worst.

KLPO is experiment zero. It is expected to lose to something an agent writes later.

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
profile. The second milestone is the `rl-reasoning` profile above: REINFORCE, GRPO,
KLPO and FlashREINFORCE are implemented against a shared gradient interface and
measured on a verifiable-reward task under one budget. They are baselines, not
presumed winners.

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
