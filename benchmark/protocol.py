"""Protected validity and decision policy."""

PROTOCOL_VERSION = "1.0.0"
PROTECTED_PATHS = (
    "benchmark",
    "harness",
    "prepare.py",
    "research.py",
    "lab.json",
)
MUTABLE_PATHS = ("train.py", "model.py", "rl.py", "sampling.py", "chess_agent.py", "algorithms")
QUICK_MIN_ABSOLUTE = 0.01
CONFIRM_MIN_ABSOLUTE = 0.01
MAX_RELATIVE_JUMP_BEFORE_AUDIT = 0.50
