# Agent boundaries

Read `program.md` before research work.

Mutable research surface: `train.py`, `model.py`, `rl.py`, `sampling.py`,
`chess_agent.py`, and `algorithms/`. Protected surface: `benchmark/`, `harness/`, `research.py`,
`prepare.py`, `lab.json`, and primary metric/protocol settings.

Profiles and their mutable target:

| Profile | Primary metric | Edit |
| --- | --- | --- |
| `smoke` | `score` | `model.py` |
| `pretrain` | `val_bpb` | `train.py` |
| `chess-tactics` | `tactical_pass_at_1` | `chess_agent.py` |
| `rl-reasoning` | `heldout_pass_at_1` | `rl.py`, `model.py`, `sampling.py`, `algorithms/` |

Submit every meaningful research change through `python research.py experiment`.
Never claim an improvement from training completion alone.
