import math
import random
import unittest
import json
import tempfile
from pathlib import Path

import chess

from algorithms.flashreinforce import FlashREINFORCE
from algorithms.grpo import GRPO
from algorithms.klpo import KLPO
from algorithms.reinforce import REINFORCE
from benchmark.metrics import improvement
from benchmark.reasoning_env import (
    ANSWER_DIGITS,
    HeldoutAccessError,
    Problem,
    RolloutBudgetExceeded,
    TrainingEnvironment,
    all_problems,
    context,
    dense_reward,
    heldout_split,
    train_split,
)
from benchmark.rl_eval import decode
from benchmark.template_registry import load_template, summaries
from harness.compare import decide
from harness.git_state import validate_candidate_paths
from model import Policy
from rl import Rollout, Step, analytic_kl, sequence_kl, train_policy
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
        for algorithm in (REINFORCE(), GRPO(), KLPO(), FlashREINFORCE()):
            self.assertEqual(len(algorithm.losses(rollouts)), 2)
            self.assertEqual(len(algorithm.coefficients(rollouts)), 2)
            self.assertIsInstance(algorithm.kl_gradient_scale(), float)
            algorithm.observe(rollouts)

    def test_higher_reward_earns_a_larger_coefficient(self):
        rollouts = [Rollout(-0.2, 1.0, -0.2, 0), Rollout(-0.2, 0.0, -0.2, 0)]
        for algorithm in (REINFORCE(), GRPO(), KLPO(), FlashREINFORCE()):
            better, worse = algorithm.coefficients(rollouts)
            self.assertGreater(better, worse, algorithm.name)


class ReasoningEnvironmentTests(unittest.TestCase):
    def test_splits_partition_the_task_without_overlap(self):
        train = train_split()
        heldout = heldout_split()
        self.assertEqual(len(train) + len(heldout), len(all_problems()))
        self.assertEqual(set(train) & set(heldout), set())
        self.assertGreater(len(heldout), 1000)

    def test_answers_and_rewards_are_exact(self):
        problem = Problem((3, 9, 1, 7))
        self.assertEqual(problem.answer, (1, 9))
        self.assertEqual(dense_reward(problem, (1, 9)), 1.0)
        self.assertEqual(dense_reward(problem, (1, 0)), 0.5)
        self.assertEqual(dense_reward(problem, (0, 0)), 0.0)

    def test_context_pads_undecided_answer_slots(self):
        slots = context((1, 2, 3, 4), ())
        self.assertEqual(len(slots), 4 + ANSWER_DIGITS - 1)
        self.assertEqual(slots[:4], [1, 2, 3, 4])

    def test_budget_is_enforced(self):
        environment = TrainingEnvironment(2, 101)
        problem = environment.sample_problem(random.Random(0))
        environment.score(problem, (0, 0))
        environment.score(problem, (0, 0))
        with self.assertRaises(RolloutBudgetExceeded):
            environment.score(problem, (0, 0))

    def test_training_cannot_score_heldout_problems(self):
        environment = TrainingEnvironment(10, 101)
        with self.assertRaises(HeldoutAccessError):
            environment.score(heldout_split()[0], (0, 0))


class PolicyTests(unittest.TestCase):
    def test_distribution_is_normalised(self):
        policy = Policy.initialise(7)
        distribution = policy.distribution(context((1, 2, 3, 4), ()))
        self.assertAlmostEqual(sum(distribution), 1.0, places=9)
        self.assertTrue(all(value > 0.0 for value in distribution))

    def test_backward_matches_finite_differences(self):
        """The hand-written backward pass is the one thing nothing else would catch."""
        policy = Policy.initialise(3)
        slots = context((4, 1, 8, 2), ())
        action = 5

        def negative_logprob() -> float:
            return -math.log(policy.distribution(slots)[action])

        probabilities, cache = policy.forward(slots)
        dlogits = list(probabilities)
        dlogits[action] -= 1.0
        gradients = policy.zero_gradients()
        policy.accumulate(cache, dlogits, gradients)

        epsilon = 1e-6
        for tensor_index, row_index, column in ((1, 0, 0), (1, 3, 7), (3, 5, 2), (0, 4, 1)):
            row = policy.tensors()[tensor_index][row_index]
            original = row[column]
            row[column] = original + epsilon
            high = negative_logprob()
            row[column] = original - epsilon
            low = negative_logprob()
            row[column] = original
            numeric = (high - low) / (2 * epsilon)
            analytic = gradients[tensor_index][row_index][column]
            self.assertAlmostEqual(numeric, analytic, places=6)


class KLPOTests(unittest.TestCase):
    def _rollout(self, logprob: float, reference_logprob: float) -> Rollout:
        return Rollout(logprob, 1.0, reference_logprob, 0)

    def test_sampled_estimators_agree_at_zero_divergence(self):
        item = self._rollout(-1.0, -1.0)
        for estimator in ("k1", "k2", "k3"):
            self.assertAlmostEqual(sequence_kl(item, estimator), 0.0, places=9)

    def test_k3_is_non_negative_where_k1_is_signed(self):
        item = self._rollout(-1.5, -1.0)
        self.assertLess(sequence_kl(item, "k1"), 0.0)
        self.assertGreater(sequence_kl(item, "k3"), 0.0)

    def test_analytic_kl_is_zero_against_itself_and_positive_otherwise(self):
        probabilities = [0.7, 0.3]
        same = Rollout(-0.3, 1.0, -0.3, 0, (Step([0], 0, probabilities, probabilities, {}),))
        self.assertAlmostEqual(analytic_kl(same), 0.0, places=9)
        different = Rollout(-0.3, 1.0, -0.3, 0, (Step([0], 0, probabilities, [0.4, 0.6], {}),))
        self.assertGreater(analytic_kl(different), 0.0)

    def test_exact_estimator_uses_the_analytic_gradient_path(self):
        self.assertGreater(KLPO(estimator="exact").kl_gradient_scale(), 0.0)
        self.assertEqual(KLPO(estimator="k3").kl_gradient_scale(), 0.0)

    def test_adaptive_controller_tracks_the_kl_budget(self):
        probabilities = [0.9, 0.1]
        drifted = [Rollout(-0.1, 1.0, -0.1, 0, (Step([0], 0, probabilities, [0.1, 0.9], {}),))]
        tightening = KLPO(kl_coefficient=0.01, kl_target=0.001)
        tightening.observe(drifted)
        self.assertGreater(tightening.kl_coefficient, 0.01)

        aligned = [Rollout(-0.1, 1.0, -0.1, 0, (Step([0], 0, probabilities, probabilities, {}),))]
        relaxing = KLPO(kl_coefficient=0.01, kl_target=0.5)
        relaxing.observe(aligned)
        self.assertLess(relaxing.kl_coefficient, 0.01)

    def test_unknown_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            KLPO(estimator="k9")
        with self.assertRaises(ValueError):
            KLPO(baseline="magic")


class TrainingLoopTests(unittest.TestCase):
    def test_training_respects_the_budget_and_improves_reward(self):
        environment = TrainingEnvironment(1600, 101)
        policy = train_policy(environment, Policy.initialise(101), {}, 101, "klpo")
        self.assertLessEqual(environment.rollouts_used, 1600)
        self.assertGreater(environment.train_reward_mean, 0.0)
        emitted = decode(policy, Problem((1, 2, 3, 4)))
        self.assertEqual(len(emitted), ANSWER_DIGITS)
        self.assertTrue(all(0 <= digit <= 9 for digit in emitted))


class LedgerAndTemplateTests(unittest.TestCase):
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
