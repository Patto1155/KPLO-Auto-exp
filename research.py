import argparse
import json
from pathlib import Path

from benchmark.template_registry import load_template, summaries, template_paths
from harness.runner import algorithm_matrix, audit, confirm, reproduce, run_experiment
from harness.results import records
from harness.progress import render_progress

ROOT = Path(__file__).resolve().parent


def show(record):
    print(json.dumps(record, indent=2, sort_keys=True))
    if record.get("baseline_metrics") and record.get("candidate_metrics"):
        metric = record["primary_metric"]
        print("\nComparison\n----------")
        print(f"Experiment: {record['experiment_id']}")
        print(f"Baseline {metric}:  {record['baseline_metrics'][metric]:.6f}")
        print(f"Candidate {metric}: {record['candidate_metrics'][metric]:.6f}")
        print(f"Improvement:        {record['delta']['primary_improvement']:+.6f}")
        print(f"Decision:           {record['decision']}")


parser = argparse.ArgumentParser(description="Benchmark-first autonomous research harness")
sub = parser.add_subparsers(dest="command", required=True)
experiment = sub.add_parser("experiment")
experiment.add_argument("--hypothesis", required=True)
experiment.add_argument("--description", default="")
experiment.add_argument("--algorithm", default="unspecified")
experiment.add_argument("--profile", choices=("smoke", "pretrain", "chess-tactics", "rl-reasoning"))
confirmation = sub.add_parser("confirm"); confirmation.add_argument("experiment_id")
auditing = sub.add_parser("audit"); auditing.add_argument("experiment_id")
reproduction = sub.add_parser("reproduce"); reproduction.add_argument("experiment_id")
sub.add_parser("status")
plot = sub.add_parser("plot")
plot.add_argument("--profile", required=True)
plot.add_argument("--output")
matrix = sub.add_parser("algorithms", help="benchmark every RL algorithm under one shared budget")
matrix.add_argument("--profile", default="rl-reasoning")
template = sub.add_parser("template", help="inspect protected environment benchmark templates")
template_sub = template.add_subparsers(dest="template_command", required=True)
template_sub.add_parser("list")
template_show = template_sub.add_parser("show")
template_show.add_argument("template_id")
template_sub.add_parser("validate")
args = parser.parse_args()
if args.command == "experiment":
    show(run_experiment(ROOT, args.hypothesis, args.description, args.algorithm, args.profile))
elif args.command == "confirm":
    show(confirm(ROOT, args.experiment_id))
elif args.command == "audit":
    show(audit(ROOT, args.experiment_id))
elif args.command == "reproduce":
    show(reproduce(ROOT, args.experiment_id))
elif args.command == "algorithms":
    payload = algorithm_matrix(ROOT, args.profile)
    metric = payload["primary_metric"]
    print(f"\n{payload['profile']} | {metric} | seeds {payload['seeds']} | budget {payload['budget']}")
    print(f"{'Algorithm':<16}{metric:>22}{'std':>10}{'entropy':>10}{'gen gap':>10}")
    for row in payload["results"]:
        values = row["metrics"]
        print(f"{row['algorithm']:<16}{values[metric]:>22.4f}{values.get(metric + '_std', 0.0):>10.4f}"
              f"{values.get('entropy', 0.0):>10.4f}{values.get('generalisation_gap', 0.0):>10.4f}")
    print("\nWritten to research/algorithms.json")
elif args.command == "template":
    if args.template_command == "list":
        show({"templates": summaries()})
    elif args.template_command == "show":
        show(load_template(args.template_id))
    else:
        loaded = [load_template(path.stem)["id"] for path in template_paths()]
        show({"valid": True, "count": len(loaded), "templates": loaded})
elif args.command == "plot":
    destination = Path(args.output) if args.output else ROOT / "research" / f"progress-{args.profile}.svg"
    show(render_progress(ROOT, args.profile, destination))
else:
    history = records(ROOT); show({"experiments": len(history), "latest": history[-1] if history else None})
