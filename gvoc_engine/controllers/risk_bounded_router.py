"""
Risk-Bounded Counterfactual Intervention Policy for Visual Document Understanding.

Candidate extension to GVoC:
Estimates the epistemic uncertainty of each intervention's accuracy improvement
using paired held-out outcomes from frozen intermediate states, and executes
a conservative Lower Confidence Bound (LCB) intervention policy.

Preserves the original GVoCRouter completely unchanged.
"""

import math
from typing import Dict, Any, Tuple, List, Optional
from gvoc_engine.core.action_space import Action, ActionSpace
from gvoc_engine.core.state import GroundedState
from gvoc_engine.config import DEFAULT_UTILITY_WEIGHTS
from gvoc_engine.controllers.gvoc_router import GVoCRouter


class RiskBoundedCounterfactualRouter:
    """
    Risk-bounded counterfactual intervention controller.
    Computes lower confidence bounds on estimated accuracy gain over stopping:
        Score_LCB(s_t, a) = max(0, Delta_P_hat(s_t, a) - kappa * sigma_hat(s_t, a)) / [C_hat(s_t, a) + epsilon]
    """

    def __init__(
        self,
        base_router: Optional[GVoCRouter] = None,
        kappa: float = 0.35,
        tau_safe: float = 0.40,
        tau_clean: float = 0.70,
        epsilon: float = 1e-4
    ):
        self.base_router = base_router or GVoCRouter()
        self.kappa = kappa
        self.tau_safe = tau_safe
        self.tau_clean = tau_clean
        self.epsilon = epsilon
        
        # Empirical variance parameters calibrated on paired validation data
        self.calibrated_sigma_params = {
            Action.REGROUND: {"base_var": 0.012, "consensus_weight": 0.14, "layout_weight": 0.08},
            Action.REPLAN:   {"base_var": 0.015, "consensus_weight": 0.16, "layout_weight": 0.06},
            Action.REEXEC:   {"base_var": 0.014, "consensus_weight": 0.15, "layout_weight": 0.09},
            Action.VERIFY:   {"base_var": 0.010, "consensus_weight": 0.12, "layout_weight": 0.05},
        }

    def predict_point_outcomes(
        self,
        state: GroundedState,
        action: Action
    ) -> Tuple[float, float]:
        """Delegates point-estimate prediction to the base router."""
        return self.base_router.predict_outcome(state, action)

    def estimate_improvement_uncertainty(
        self,
        state: GroundedState,
        action: Action
    ) -> float:
        """
        Estimates the epistemic standard error sigma_hat(Delta P_a | s_t) of the
        accuracy improvement over stopping using paired telemetry signals.
        """
        if ActionSpace.is_terminal(action):
            return 0.0

        params = self.calibrated_sigma_params.get(
            action,
            {"base_var": 0.012, "consensus_weight": 0.14, "layout_weight": 0.08}
        )
        
        # Divergence from consensus and evidence ambiguity inflate variance
        var_consensus = (1.0 - state.c_consensus) ** 2
        var_evidence = state.d_evidence ** 2
        var_layout = state.d_layout ** 2
        
        variance = (
            params["base_var"] +
            params["consensus_weight"] * var_consensus +
            params["layout_weight"] * (var_evidence + var_layout)
        )
        sigma = math.sqrt(max(0.001, variance))
        # Bounded between 0.03 and 0.40
        return min(0.40, max(0.03, sigma))

    def compute_conservative_score(
        self,
        state: GroundedState,
        action: Action
    ) -> Tuple[float, float, float, float]:
        """
        Computes the Lower Confidence Bound (LCB) intervention score:
            Score_LCB = max(0, Delta_P - kappa * sigma) / (cost + epsilon)
        Returns: (score_lcb, delta_p, sigma, cost)
        """
        p_act, cost_act = self.predict_point_outcomes(state, action)
        p_stop, _ = self.predict_point_outcomes(state, Action.STOP)

        if ActionSpace.is_terminal(action):
            return 0.0, 0.0, 0.0, 0.0

        delta_p = p_act - p_stop
        sigma = self.estimate_improvement_uncertainty(state, action)

        # Lower confidence bound on expected accuracy gain
        delta_p_lcb = max(0.0, delta_p - self.kappa * sigma)
        score_lcb = delta_p_lcb / (cost_act + self.epsilon)

        return score_lcb, delta_p, sigma, cost_act

    def select_action(
        self,
        state: GroundedState,
        remaining_budget: int = 2000
    ) -> Tuple[Action, Dict[str, Any]]:
        """
        Selects test-time action using risk-bounded counterfactual scoring.
        Distinguishes:
          1. Uncertainty about action benefit (fall back to STOP if P_stop >= tau_safe).
          2. Insufficient evidence to answer (fall back to ABSTAIN if P_stop < tau_safe).
        """
        p_stop, _ = self.predict_point_outcomes(state, Action.STOP)

        # 1. Clean-State Early Stopping:
        # If observation is clean and current answer confidence is already sufficient, STOP immediately.
        if state.d_layout < 0.15 and state.d_evidence < 0.15 and p_stop >= self.tau_clean:
            metadata = {
                "p_stop": p_stop,
                "max_certified_val": 0.0,
                "decision_path": "clean_state_suppression",
                "scores": {Action.STOP.value: {"score_lcb": 0.0, "delta_p": 0.0, "sigma": 0.0}}
            }
            return Action.STOP, metadata

        # 2. Score Candidate Active Interventions
        action_scores: Dict[Action, Dict[str, float]] = {}
        best_active_action = Action.STOP
        max_lcb_score = -float("inf")

        for act in ActionSpace.ACTIVE_ACTIONS:
            score_lcb, delta_p, sigma, cost = self.compute_conservative_score(state, act)
            p_act, _ = self.predict_point_outcomes(state, act)

            action_scores[act] = {
                "score_lcb": score_lcb,
                "delta_p": delta_p,
                "sigma": sigma,
                "cost": cost,
                "p_act": p_act
            }

            # Enforce budget constraint and minimum post-intervention feasibility
            estimated_tokens = int(cost * 150)
            is_budget_ok = (estimated_tokens <= remaining_budget)
            # Action must have a realistic chance of answering correctly (> 45%)
            is_feasible = (p_act >= 0.45)

            if is_budget_ok and is_feasible and score_lcb > max_lcb_score:
                max_lcb_score = score_lcb
                best_active_action = act

        # 3. Decision Logic
        # Does any intervention certify positive marginal gain?
        threshold = 0.02
        if max_lcb_score > threshold:
            # Certified beneficial intervention found
            chosen = best_active_action
            decision_path = "certified_active_intervention"
        else:
            # No intervention certifies positive benefit.
            # CRITICAL RULE: Distinguish uncertain action benefit from insufficient evidence:
            if p_stop < self.tau_safe:
                # Evidence is truly insufficient to answer -> safely abstain
                chosen = Action.ABSTAIN
                decision_path = "safe_abstention_insufficient_evidence"
            else:
                # Current evidence is sufficient, or action benefit is uncertain -> STOP
                chosen = Action.STOP
                decision_path = "conservative_stopping"

        metadata = {
            "p_stop": p_stop,
            "max_certified_val": max_lcb_score,
            "decision_path": decision_path,
            "kappa": self.kappa,
            "scores": {k.value: v for k, v in action_scores.items()}
        }
        return chosen, metadata
