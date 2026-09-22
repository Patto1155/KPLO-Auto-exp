from benchmark.metrics import improvement
from benchmark.protocol import CONFIRM_MIN_ABSOLUTE, MAX_RELATIVE_JUMP_BEFORE_AUDIT, QUICK_MIN_ABSOLUTE


def decide(baseline, candidate, metric, direction, stage):
    absolute = improvement(baseline[metric], candidate[metric], direction)
    relative = absolute / max(abs(baseline[metric]), 1e-12)
    delta = {"primary_improvement": absolute, "relative_improvement": relative}
    if abs(relative) > MAX_RELATIVE_JUMP_BEFORE_AUDIT:
        return "INCONCLUSIVE", delta, ["anomalous jump requires audit"]
    threshold = CONFIRM_MIN_ABSOLUTE if stage == "confirmation" else QUICK_MIN_ABSOLUTE
    decision = "KEEP" if absolute >= threshold else "REJECT" if absolute <= -threshold else "INCONCLUSIVE"
    return decision, delta, []
