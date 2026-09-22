# Agent boundaries

Read `program.md` before research work.

Mutable research surface: `train.py`, `model.py`, `rl.py`, `sampling.py`,
`chess_agent.py`, and `algorithms/`. Protected surface: `benchmark/`, `harness/`, `research.py`,
`prepare.py`, `lab.json`, and primary metric/protocol settings.

Submit every meaningful research change through `python research.py experiment`.
Never claim an improvement from training completion alone.
