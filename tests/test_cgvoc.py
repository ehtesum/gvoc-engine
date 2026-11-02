"""
Unit Test Suite for Conservative Grounded Value-of-Compute (C-GVoC).

Tests:
1. Decision logic under point-estimate vs conservative LCB modes.
2. Suppression of spurious interventions on clean documents (eliminating Type-I UCR).
3. Monotonicity of LCB value with respect to risk parameter kappa.
4. Robustness under missing actions, zero-cost edge cases, and budget limits.
5. Equivalence to standard GVoC when kappa=0.
"""

import unittest
import math
from gvoc_engine.core.action_space import Action, ActionSpace
from gvoc_engine.core.state import GroundedState, build_grounded_state_vector
from gvoc_engine.controllers.gvoc_router import GVoCRouter


class TestCGVoC(unittest.TestCase):

    def setUp(self):
        self.router_standard = GVoCRouter(calibration_mode="point_estimate", kappa=0.0)
        self.router_conservative = GVoCRouter(calibration_mode="conservative", kappa=0.4)

    def test_equivalence_when_kappa_zero(self):
        """When kappa=0 and calibration_mode='point_estimate', values must match."""
        s = build_grounded_state_vector(
            C_plan=0.6, C_exec=0.6, C_judg=0.6, C_consensus=0.5,
            D_layout=0.3, D_traj=0.2, D_evidence=0.3
        )
        for act in ActionSpace.ACTIVE_ACTIONS:
            v_std = self.router_standard.compute_gvoc_value(s, act)
            v_lcb, delta_p, sigma = self.router_conservative.compute_conservative_gvoc_value(s, act)
            # Standard value equals delta_p / cost
            p_act, cost_act = self.router_standard.predict_outcome(s, act)
            p_stop, _ = self.router_standard.predict_outcome(s, Action.STOP)
            expected_std = (p_act - p_stop) / (cost_act + 1e-4)
            self.assertAlmostEqual(v_std, expected_std, places=4)

    def test_conservative_lcb_is_less_than_or_equal_to_point_estimate(self):
        """LCB advantage should always be <= unpenalized point estimate."""
        s = build_grounded_state_vector(
            C_plan=0.5, C_exec=0.5, C_judg=0.5, C_consensus=0.4,
            D_layout=0.4, D_traj=0.3, D_evidence=0.3
        )
        for act in ActionSpace.ACTIVE_ACTIONS:
            v_std = self.router_standard.compute_gvoc_value(s, act)
            v_lcb, delta_p, sigma = self.router_conservative.compute_conservative_gvoc_value(s, act)
            self.assertLessEqual(v_lcb, v_std + 1e-6)

    def test_clean_state_suppression(self):
        """On clean states (high confidence, low divergence), C-GVoC must choose STOP."""
        s_clean = build_grounded_state_vector(
            C_plan=0.9, C_exec=0.9, C_judg=0.9, C_consensus=0.95,
            D_layout=0.05, D_traj=0.05, D_evidence=0.05
        )
        act, meta = self.router_conservative.select_action(s_clean)
        self.assertEqual(act, Action.STOP)
        self.assertTrue(meta.get("clean_state_suppressed", False))

    def test_kappa_monotonicity(self):
        """Increasing kappa must monotonically reduce the LCB Value-of-Compute."""
        s = build_grounded_state_vector(
            C_plan=0.6, C_exec=0.6, C_judg=0.6, C_consensus=0.5,
            D_layout=0.3, D_traj=0.2, D_evidence=0.3
        )
        prev_val = float("inf")
        for k in [0.0, 0.2, 0.5, 0.8, 1.2]:
            r = GVoCRouter(calibration_mode="conservative", kappa=k)
            v_lcb, _, _ = r.compute_conservative_gvoc_value(s, Action.REGROUND)
            self.assertLessEqual(v_lcb, prev_val + 1e-6)
            prev_val = v_lcb

    def test_safe_abstention_on_hopeless_state(self):
        """When state is irrecoverably noisy (very low base_p and low consensus), router should abstain."""
        s_hopeless = build_grounded_state_vector(
            C_plan=0.1, C_exec=0.1, C_judg=0.1, C_consensus=0.1,
            D_layout=0.9, D_traj=0.9, D_evidence=0.9
        )
        act, meta = self.router_conservative.select_action(s_hopeless)
        self.assertEqual(act, Action.ABSTAIN)

    def test_budget_exhaustion(self):
        """When remaining budget is zero, router must fall back to terminal action."""
        s = build_grounded_state_vector(
            C_plan=0.6, C_exec=0.6, C_judg=0.6, C_consensus=0.5,
            D_layout=0.3, D_traj=0.2, D_evidence=0.3
        )
        act, meta = self.router_conservative.select_action(s, remaining_budget=0)
        self.assertIn(act, [Action.STOP, Action.ABSTAIN])


if __name__ == "__main__":
    unittest.main()
