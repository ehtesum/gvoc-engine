"""
Unit Tests for Risk-Bounded Counterfactual Intervention Policy.

Covers:
1. Paired-state alignment (frozen state verification).
2. Uncertainty estimation dynamics under telemetry distortion.
3. Negative estimated benefit handling (LCB clipping at 0).
4. Missing / terminal actions safety.
5. Distinction between uncertainty about action benefit vs. insufficient evidence.
"""

import unittest
import math
from gvoc_engine.core.action_space import Action, ActionSpace
from gvoc_engine.core.state import GroundedState, build_grounded_state_vector
from gvoc_engine.controllers.risk_bounded_router import RiskBoundedCounterfactualRouter


class TestRiskBoundedRouter(unittest.TestCase):

    def setUp(self):
        self.router = RiskBoundedCounterfactualRouter(kappa=0.35)

    def test_paired_state_alignment(self):
        """Verifies that candidate actions evaluate against identical frozen state baseline."""
        s = build_grounded_state_vector(
            C_plan=0.7, C_exec=0.6, C_judg=0.65, C_consensus=0.70,
            D_layout=0.25, D_traj=0.15, D_evidence=0.20
        )
        p_stop, _ = self.router.predict_point_outcomes(s, Action.STOP)
        for act in ActionSpace.ACTIVE_ACTIONS:
            score, delta_p, sigma, cost = self.router.compute_conservative_score(s, act)
            p_act, _ = self.router.predict_point_outcomes(s, act)
            # Delta P must strictly equal p_act - p_stop from the exact same state s
            self.assertAlmostEqual(delta_p, p_act - p_stop, places=5)
            self.assertGreater(cost, 0.0)

    def test_uncertainty_estimation_dynamics(self):
        """Verifies that epistemic standard error sigma increases with telemetry ambiguity."""
        s_confident = build_grounded_state_vector(
            C_plan=0.9, C_exec=0.9, C_judg=0.9, C_consensus=0.95,
            D_layout=0.05, D_traj=0.05, D_evidence=0.05
        )
        s_divergent = build_grounded_state_vector(
            C_plan=0.4, C_exec=0.4, C_judg=0.4, C_consensus=0.30,
            D_layout=0.60, D_traj=0.50, D_evidence=0.70
        )
        for act in ActionSpace.ACTIVE_ACTIONS:
            sigma_conf = self.router.estimate_improvement_uncertainty(s_confident, act)
            sigma_div = self.router.estimate_improvement_uncertainty(s_divergent, act)
            self.assertGreater(sigma_div, sigma_conf)
            self.assertGreater(sigma_conf, 0.0)

    def test_negative_estimated_benefit(self):
        """Verifies that when delta_p <= 0, LCB score is clipped cleanly at 0.0."""
        s = build_grounded_state_vector(
            C_plan=0.95, C_exec=0.95, C_judg=0.95, C_consensus=0.95,
            D_layout=0.01, D_traj=0.01, D_evidence=0.01
        )
        # For an action with no positive gain over an already perfect state
        score, delta_p, sigma, cost = self.router.compute_conservative_score(s, Action.VERIFY)
        self.assertGreaterEqual(score, 0.0)
        if delta_p - self.router.kappa * sigma <= 0:
            self.assertEqual(score, 0.0)

    def test_missing_and_terminal_actions(self):
        """Verifies that terminal actions and edge cases return safe default scores."""
        s = build_grounded_state_vector()
        score_stop, delta_stop, sigma_stop, _ = self.router.compute_conservative_score(s, Action.STOP)
        self.assertEqual(score_stop, 0.0)
        self.assertEqual(delta_stop, 0.0)
        self.assertEqual(sigma_stop, 0.0)
        
        score_abs, delta_abs, sigma_abs, _ = self.router.compute_conservative_score(s, Action.ABSTAIN)
        self.assertEqual(score_abs, 0.0)

    def test_distinguish_uncertainty_from_insufficient_evidence(self):
        """
        Critical check: When action benefit is uncertain:
        - If P_stop >= tau_safe, model must STOP (not abstain).
        - If P_stop < tau_safe, model must ABSTAIN.
        """
        # Case 1: Moderate evidence (P_stop >= 0.40), but actions provide no certified gain due to high uncertainty
        s_moderate = build_grounded_state_vector(
            C_plan=0.65, C_exec=0.65, C_judg=0.65, C_consensus=0.30,
            D_layout=0.08, D_traj=0.08, D_evidence=0.08, D_answer=0.05
        )
        router_strict = RiskBoundedCounterfactualRouter(kappa=1.5, tau_safe=0.40)
        act1, meta1 = router_strict.select_action(s_moderate)
        self.assertEqual(act1, Action.STOP)
        self.assertEqual(meta1["decision_path"], "conservative_stopping")

        # Case 2: Insufficient evidence (P_stop < 0.40) and no action achieves viability
        s_hopeless = build_grounded_state_vector(
            C_plan=0.1, C_exec=0.1, C_judg=0.1, C_consensus=0.10,
            D_layout=0.85, D_traj=0.85, D_evidence=0.85
        )
        act2, meta2 = router_strict.select_action(s_hopeless)
        self.assertEqual(act2, Action.ABSTAIN)
        self.assertEqual(meta2["decision_path"], "safe_abstention_insufficient_evidence")

    def test_budget_exhaustion(self):
        """When budget is zero, active interventions cannot be executed."""
        s = build_grounded_state_vector(
            C_plan=0.5, C_exec=0.5, C_judg=0.5, C_consensus=0.50,
            D_layout=0.40, D_traj=0.30, D_evidence=0.40
        )
        act, meta = self.router.select_action(s, remaining_budget=0)
        self.assertIn(act, [Action.STOP, Action.ABSTAIN])


if __name__ == "__main__":
    unittest.main()
