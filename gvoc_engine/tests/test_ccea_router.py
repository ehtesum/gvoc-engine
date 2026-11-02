"""
Unit tests for CCEACostAwareRouter decision-theoretic stopping policy.
"""

import unittest
from gvoc_engine.core.action_space import Action
from gvoc_engine.core.state import GroundedState
from gvoc_engine.interventions.context_preservation import (
    ContextKind,
    EvidenceRegion,
)
from gvoc_engine.controllers.ccea_router import CCEACostAwareRouter


class TestCCEACostAwareRouter(unittest.TestCase):
    def setUp(self):
        self.router = CCEACostAwareRouter(
            kappa=0.40,
            lambda_cost=0.25,
            tau_safe=0.45,
            tau_clean=0.75,
            max_steps=2,
            cost_per_acquisition=0.15
        )
        self.valid_evidence = EvidenceRegion(
            target_box=(100, 150, 140, 200),
            required_context=(
                (ContextKind.COLUMN_LABEL, (40, 150, 80, 200)),
                (ContextKind.ROW_LABEL, (100, 80, 140, 140)),
            ),
            evidence_id="valid_table"
        )

    def test_beneficial_intervention_selected_within_budget(self):
        # Ambiguous initial state where layout discrepancy is high and p_stop is moderate
        state = GroundedState(
            c_plan=0.60,
            c_exec=0.60,
            c_judg=0.60,
            c_consensus=0.70,
            d_layout=0.35,      # High layout error -> high expected gain
            d_evidence=0.30,
            b_remaining=2000,
            step_index=0
        )
        action, meta = self.router.select_action(
            state,
            remaining_budget=2000,
            evidence_region=self.valid_evidence,
            image_width=1000,
            image_height=800
        )
        self.assertEqual(action, Action.REGROUND)
        self.assertEqual(meta["decision_path"], "beneficial_context_crop_acquired")
        self.assertTrue(meta["acquisition_valid"])

    def test_clean_state_early_stopping(self):
        # Clean state with high confidence and minimal layout loss
        clean_state = GroundedState(
            c_plan=0.95,
            c_exec=0.90,
            c_judg=0.95,
            c_consensus=0.98,
            d_layout=0.02,
            d_evidence=0.03,
            b_remaining=2000,
            step_index=0
        )
        action, meta = self.router.select_action(
            clean_state,
            remaining_budget=2000,
            evidence_region=self.valid_evidence,
            image_width=1000,
            image_height=800
        )
        self.assertEqual(action, Action.STOP)
        self.assertEqual(meta["decision_path"], "clean_state_early_stopping")

    def test_invalid_crop_rejected_even_if_benefit_is_high(self):
        # High layout loss, but target box is out of bounds
        state = GroundedState(
            d_layout=0.50,
            d_evidence=0.40,
            b_remaining=2000,
            step_index=0
        )
        invalid_evidence = EvidenceRegion(
            target_box=(100, 150, 1500, 200),  # ymax=1500 > height=800
            required_context=(),
            evidence_id="oob_region"
        )
        action, meta = self.router.select_action(
            state,
            remaining_budget=2000,
            evidence_region=invalid_evidence,
            image_width=1000,
            image_height=800
        )
        # Should not acquire invalid crop; should fall back to STOP or ABSTAIN
        self.assertNotEqual(action, Action.REGROUND)
        self.assertFalse(meta["acquisition_valid"])
        self.assertIn("Target box invalid", meta["rejection_reason"])

    def test_step_limit_and_budget_exhaustion_stops(self):
        # Step limit reached (step_index = 2 >= max_steps = 2)
        exhausted_state = GroundedState(
            d_layout=0.40,
            step_index=2,
            b_remaining=100
        )
        action, meta = self.router.select_action(
            exhausted_state,
            remaining_budget=100,
            evidence_region=self.valid_evidence,
            image_width=1000,
            image_height=800
        )
        self.assertEqual(meta["decision_path"], "budget_or_step_limit_exhausted")
        self.assertIn(action, [Action.STOP, Action.ABSTAIN])

    def test_insufficient_evidence_triggers_abstention_when_gain_is_insufficient(self):
        # Very low confidence across all features (p_stop < tau_safe = 0.45)
        # and layout loss is zero (so acquisition gain is zero)
        state = GroundedState(
            c_plan=0.20,
            c_exec=0.20,
            c_judg=0.20,
            c_consensus=0.20,
            d_layout=0.01,
            d_evidence=0.01,
            b_remaining=2000,
            step_index=0
        )
        action, meta = self.router.select_action(
            state,
            remaining_budget=2000,
            evidence_region=self.valid_evidence,
            image_width=1000,
            image_height=800
        )
        self.assertEqual(action, Action.ABSTAIN)
        self.assertEqual(meta["decision_path"], "insufficient_evidence_safe_abstention")
        self.assertLess(meta["p_stop"], self.router.tau_safe)


if __name__ == "__main__":
    unittest.main()
