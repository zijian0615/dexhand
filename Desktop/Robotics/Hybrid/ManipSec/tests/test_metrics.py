from __future__ import annotations

import unittest

from manipsec_bench import EpisodeOutcome, RunCondition, summarize


def outcome(
    pair_id: str,
    condition: RunCondition,
    *,
    task_success: bool,
    attack_success: bool = False,
) -> EpisodeOutcome:
    adversarial = condition is RunCondition.ADVERSARIAL
    return EpisodeOutcome(
        pair_id=pair_id,
        case_id=f"{pair_id}-{condition}",
        task_id="pick-place",
        policy_id=f"policy-{pair_id}",
        condition=condition,
        seed=0,
        env_index=0,
        task_success=task_success,
        attack_success=attack_success,
        task_score=float(task_success),
        artifact_id="attack" if adversarial else None,
    )


class MetricTests(unittest.TestCase):
    def test_metrics_share_adversarial_denominator(self) -> None:
        rows = [
            outcome("a", RunCondition.CLEAN, task_success=True),
            outcome("b", RunCondition.CLEAN, task_success=True),
            outcome("c", RunCondition.CLEAN, task_success=True),
            outcome("d", RunCondition.CLEAN, task_success=False),
            outcome("a", RunCondition.ADVERSARIAL, task_success=True),
            outcome("b", RunCondition.ADVERSARIAL, task_success=True, attack_success=True),
            outcome("c", RunCondition.ADVERSARIAL, task_success=False),
            outcome("d", RunCondition.ADVERSARIAL, task_success=False, attack_success=True),
        ]

        result = summarize(rows)

        self.assertEqual(result.benign_tsr, 0.75)
        self.assertAlmostEqual(result.secure_success_rate, 1 / 3)
        self.assertAlmostEqual(result.attack_success_rate, 1 / 3)
        self.assertAlmostEqual(result.functional_failure_rate, 1 / 3)
        self.assertEqual(result.eligible_pairs, 3)
        self.assertEqual(result.adversarial_runs, 3)


if __name__ == "__main__":
    unittest.main()

