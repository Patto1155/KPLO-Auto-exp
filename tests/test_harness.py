import unittest
import json
import tempfile
from pathlib import Path

import chess

from algorithms.grpo import GRPO
from algorithms.klpo import KLPO
from algorithms.reinforce import REINFORCE
from benchmark.metrics import improvement
from benchmark.template_registry import load_template, summaries
from harness.compare import decide
from harness.git_state import validate_candidate_paths
from rl import Rollout
from benchmark.chess_tactics_eval import forcing_moves, generate_suite
from harness.accept import promote
from harness.progress import render_progress


class HarnessTests(unittest.TestCase):
    def test_metric_directions(self):
        self.assertAlmostEqual(improvement(2.0, 1.8, "min"), 0.2)
        self.assertAlmostEqual(improvement(0.5, 0.6, "max"), 0.1)

    def test_decisions(self):
        self.assertEqual(decide({"score": 0.5}, {"score": 0.52}, "score", "max", "quick")[0], "KEEP")
        self.assertEqual(decide({"score": 0.5}, {"score": 0.48}, "score", "max", "quick")[0], "REJECT")
        self.assertEqual(decide({"score": 0.5}, {"score": 0.505}, "score", "max", "quick")[0], "INCONCLUSIVE")

    def test_protected_paths(self):
        self.assertTrue(validate_candidate_paths(["model.py"])[0])
        self.assertFalse(validate_candidate_paths(["benchmark/metrics.py"])[0])
        self.assertFalse(validate_candidate_paths(["README.md"])[0])

    def test_algorithms_share_interface(self):
        rollouts = [Rollout(-0.2, 1.0, -0.25, 0), Rollout(-0.4, 0.0, -0.3, 0)]
        for algorithm in (REINFORCE(), GRPO(), KLPO()):
            self.assertEqual(len(algorithm.losses(rollouts)), 2)

    def test_environment_templates_validate(self):
        expected = {
            "minecraft_ender_dragon",
            "minecraft_progression",
            "chess_engine",
            "chess_tactics",
            "go_9x9",
        }
        self.assertEqual({item["id"] for item in summaries()}, expected)
        for template_id in expected:
            template = load_template(template_id)
            self.assertIsInstance(template["metrics"]["primary"], str)
            self.assertTrue(template["metrics"]["primary"])
            self.assertTrue(template["fairness"])
            self.assertTrue(template["integrity"])

    def test_chess_suite_answers_are_legal_and_mate_one_labels_are_exact(self):
        suite = generate_suite(101, 8)
        self.assertEqual(len(suite), 8)
        for fen, expected, depth in suite:
            board = chess.Board(fen)
            self.assertIn(chess.Move.from_uci(expected), board.legal_moves)
            if depth <= 3:
                self.assertEqual(forcing_moves(board, depth), [expected])

    def test_champions_are_profile_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "research").mkdir()
            (root / "research" / "current_best.json").write_text(
                json.dumps({"experiment_id": "old", "git_commit": "abc", "confirmed": True})
            )
            promote(root, "chess-tactics", "chess-best", "def")
            payload = json.loads((root / "research" / "current_best.json").read_text())
            self.assertEqual(payload["profiles"]["smoke"]["git_commit"], "abc")
            self.assertEqual(payload["profiles"]["chess-tactics"]["git_commit"], "def")

    def test_progress_chart_requires_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "research").mkdir()
            (root / "research" / "experiments.jsonl").write_text("")
            (root / "lab.json").write_text(json.dumps({"profiles": {"x": {"direction": "max", "primary_metric": "score"}}}))
            with self.assertRaises(ValueError):
                render_progress(root, "x", root / "progress.svg")


if __name__ == "__main__":
    unittest.main()
