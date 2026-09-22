# Autonomous research contract

Your objective is to improve the primary benchmark under the fixed protected protocol.

Start by reading `docs/AGENT_QUICKSTART.md`. Choose exactly one profile and remain on
that profile's ancestry. For chess research, edit only `chess_agent.py`. For RL
reasoning research, edit `rl.py`, `model.py`, `sampling.py`, or `algorithms/`, and say
which algorithm you ran with `--algorithm`.

For every hypothesis:

1. State the hypothesis briefly.
2. Make the smallest useful change in a mutable file.
3. Run `python research.py experiment --hypothesis "..."`.
4. Inspect the baseline/candidate comparison and validity checks.
5. Retain improvements; allow the harness to revert regressions.
6. Record what a failure taught in `research/failed_ideas.md` only when the generated
   record is insufficient.
7. Continue independently within the experiment or compute budget.
8. Periodically run `python research.py plot --profile <profile>` to expose whether the
   session is genuinely hill-climbing or merely producing noise.

Do not edit `benchmark/`, `harness/`, `prepare.py`, `research.py`, or `lab.json` during
an experiment. Do not train on held-out material, select a metric after seeing results,
or claim progress without comparative benchmark evidence. Prefer experiments that
discriminate between competing explanations.

Quick benchmarks screen ideas. A quick KEEP is provisional. Run
`python research.py confirm <experiment_id>` before the change becomes the confirmed
champion. The numerical evaluator, not an LLM, determines the measured outcome.
