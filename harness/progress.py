"""Dependency-free autoresearch-style progress charts."""

from __future__ import annotations

import html
from pathlib import Path

from harness.results import records


def render_progress(root: Path, profile: str, output: Path) -> dict:
    history = [
        row for row in records(root)
        if row.get("parameters", {}).get("profile") == profile
        and row.get("candidate_metrics")
        and row.get("primary_metric") in row["candidate_metrics"]
    ]
    if not history:
        raise ValueError(f"no quick-stage results for profile {profile!r}")
    config = __import__("json").loads((root / "lab.json").read_text())
    direction = config["profiles"][profile]["direction"]
    metric = config["profiles"][profile]["primary_metric"]
    scores = [float(row["candidate_metrics"][metric]) for row in history]
    baseline = float(history[0]["baseline_metrics"][metric])
    all_scores = [baseline, *scores]
    low, high = min(all_scores), max(all_scores)
    padding = max((high - low) * 0.12, 0.01)
    low -= padding; high += padding
    width, height = 1000, 560
    left, right, top, bottom = 86, 30, 58, 74
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(index: int) -> float:
        return left + (index / max(len(history), 1)) * plot_w

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * plot_h

    best = baseline
    running = [(x(0), y(best))]
    kept = 0
    for index, (row, score) in enumerate(zip(history, scores), 1):
        if row["decision"] == "KEEP":
            best = score
        if row["decision"] == "KEEP":
            kept += 1
        running.extend([(x(index), running[-1][1]), (x(index), y(best))])

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#24302b}.grid{stroke:#e5e9e7;stroke-width:1}.axis{stroke:#59635f;stroke-width:1.2}.discard{fill:#bcc5c1;opacity:.65}.keep{fill:#079455;stroke:#067647;stroke-width:1.5}.best{fill:none;stroke:#079455;stroke-width:2.2}</style>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-weight="600">{html.escape(profile)} progress: {len(history)} experiments, {kept} kept</text>',
    ]
    for tick in range(6):
        value = low + (high - low) * tick / 5
        yy = y(value)
        lines += [f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>',
                  f'<text x="{left-12}" y="{yy+4:.1f}" text-anchor="end" font-size="12">{value:.3f}</text>']
    lines += [f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
              f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>']
    points = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in running)
    lines.append(f'<polyline class="best" points="{points}"/>')
    for index, (row, score) in enumerate(zip(history, scores), 1):
        cls = "keep" if row["decision"] == "KEEP" else "discard"
        radius = 5 if cls == "keep" else 3
        lines.append(f'<circle class="{cls}" cx="{x(index):.1f}" cy="{y(score):.1f}" r="{radius}"/>')
        if cls == "keep" or row["decision"] == "INCONCLUSIVE":
            label = html.escape(f"{row['experiment_id']} {row.get('algorithm', '')}")
            lines.append(f'<text x="{x(index)+7:.1f}" y="{y(score)-8:.1f}" font-size="10" transform="rotate(-25 {x(index)+7:.1f} {y(score)-8:.1f})">{label}</text>')
    lines += [
        f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" font-size="13">Experiment</text>',
        f'<text x="18" y="{top+plot_h/2}" text-anchor="middle" font-size="13" transform="rotate(-90 18 {top+plot_h/2})">{html.escape(metric)} ({"higher" if direction == "max" else "lower"} is better)</text>',
        f'<circle class="discard" cx="{width-190}" cy="28" r="4"/><text x="{width-180}" y="32" font-size="11">Discarded</text>',
        f'<circle class="keep" cx="{width-105}" cy="28" r="5"/><text x="{width-95}" y="32" font-size="11">Kept</text>',
        '</svg>',
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return {"profile": profile, "experiments": len(history), "kept": kept, "best": best, "output": str(output)}
