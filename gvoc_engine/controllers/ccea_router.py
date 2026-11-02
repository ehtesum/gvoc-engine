"""
Context-Preserving, Cost-Aware Visual Evidence Acquisition (CCEA) Controller.

Implements transparent, decision-theoretic action selection and stopping:
    score(action) = conservative_estimated_accuracy_gain(action) - lambda_cost * incremental_cost(action)
Subject to context preservation, validity constraints, and remaining compute budget.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

from gvoc_engine.core.action_space import Action, ActionSpace
from gvoc_engine.core.state import GroundedState
from gvoc_engine.interventions.context_preservation import (
    EvidenceRegion,
    assess_crop,
    propose_context_preserving_crop,
    validate_box,
)


@dataclass
class ActionCandidate:
    action: Action
    p_hat: float
    cost_hat: float
    delta_p: float
    sigma_hat: float
    score: float
    is_valid: bool
    rejection_reason: Optional[str] = None


class CCEACostAwareRouter:
    """
    Decision-theoretic controller for Context-Preserving, Cost-Aware Visual Evidence Acquisition.
    
    Distinguishes:
    1. Insufficient evidence where intervention benefit justifies cost -> ACQUIRE
    2. Sufficient evidence or marginal benefit does not justify cost -> STOP
    3. Irrecoverably insufficient evidence within budget -> ABSTAIN
    """

    def __init__(
        self,
        kappa: float = 0.40,            # Epistemic conservatism penalty
        lambda_cost: float = 0.25,      # Trade-off weight on incremental compute cost
        tau_safe: float = 0.45,         # Threshold below which evidence is deemed inadequate
        tau_clean: float = 0.75,        # Threshold above which state is clean and needs no repair
        max_steps: int = 2,             # Maximum allowed intervention steps
        cost_per_acquisition: float = 0.15 # Normalized incremental cost of an acquisition
    ):
        self.kappa = kappa
        self.lambda_cost = lambda_cost
        self.tau_safe = tau_safe
        self.tau_clean = tau_clean
        self.max_steps = max_steps
        self.cost_per_acquisition = cost_per_acquisition

    def estimate_stop_probability(self, state: GroundedState) -> float:
        """Estimates intrinsic probability that the current answer is correct without further intervention."""
        # Baseline correctness estimated from intermediate semantic confidences and layout losses
        base_p = (
            state.c_plan * 0.25 +
            state.c_exec * 0.25 +
            state.c_judg * 0.20 +
            state.c_consensus * 0.15 -
            state.d_layout * 0.35 -
            state.d_evidence * 0.25
        )
        return max(0.05, min(0.98, base_p))

    def estimate_acquisition_gain(
        self,
        state: GroundedState,
        p_stop: float
    ) -> Tuple[float, float, float]:
        """
        Estimates the expected gain and uncertainty of acquiring a context-preserving crop.
        Returns: (delta_p_hat, sigma_hat, delta_p_lcb)
        """
        # Expected accuracy gain is higher when layout discrepancy or evidence misalignment is prominent
        gain_factor = (state.d_layout * 1.5) + (state.d_evidence * 1.2) - (state.c_judg * 0.2)
        delta_p_hat = max(0.01, min(0.40, gain_factor))

        # Epistemic standard error: uncertainty rises with consensus divergence and evidence ambiguity
        var_consensus = (1.0 - state.c_consensus) ** 2
        var_evidence = state.d_evidence ** 2
        var_layout = state.d_layout ** 2
        sigma_hat = math.sqrt(0.01 + 0.10 * var_consensus + 0.08 * var_evidence + 0.05 * var_layout)
        sigma_hat = min(0.35, max(0.03, sigma_hat))

        # Conservative lower confidence bound on accuracy improvement
        delta_p_lcb = max(0.0, delta_p_hat - self.kappa * sigma_hat)
        return delta_p_hat, sigma_hat, delta_p_lcb

    def select_action(
        self,
        state: GroundedState,
        remaining_budget: int = 2000,
        evidence_region: Optional[EvidenceRegion] = None,
        image_width: int = 1000,
        image_height: int = 1000,
    ) -> Tuple[Action, Dict[str, Any]]:
        """
        Selects optimal action under cost-aware stopping rule.
        
        Returns: (chosen_action, telemetry_metadata)
        """
        p_stop = self.estimate_stop_probability(state)
        candidates: Dict[str, Dict[str, Any]] = {}

        # 1. Finite Step & Budget Exhaustion Guard
        if state.step_index >= self.max_steps or remaining_budget < 150:
            decision = Action.ABSTAIN if p_stop < self.tau_safe else Action.STOP
            metadata = {
                "decision_path": "budget_or_step_limit_exhausted",
                "p_stop": p_stop,
                "step_index": state.step_index,
                "remaining_budget": remaining_budget,
                "candidates": {Action.STOP.value: {"score": 0.0}}
            }
            return decision, metadata

        # 2. Clean-State Pre-flight Suppression
        if state.d_layout < 0.15 and state.d_evidence < 0.15 and p_stop >= self.tau_clean:
            metadata = {
                "decision_path": "clean_state_early_stopping",
                "p_stop": p_stop,
                "reason": "Input evidence is already clean and sufficient",
                "candidates": {Action.STOP.value: {"score": 0.0}}
            }
            return Action.STOP, metadata

        # 3. Evaluate Acquisition Candidate
        delta_p_hat, sigma_hat, delta_p_lcb = self.estimate_acquisition_gain(state, p_stop)
        incremental_cost = self.cost_per_acquisition
        acquisition_score = delta_p_lcb - (self.lambda_cost * incremental_cost)

        # Pre-flight Crop Validity Check
        acquisition_valid = True
        rejection_reason = None

        if evidence_region and evidence_region.target_box:
            is_valid_tgt, tgt_err = validate_box(evidence_region.target_box, image_width, image_height)
            if not is_valid_tgt:
                acquisition_valid = False
                rejection_reason = f"Target box invalid: {tgt_err}"
            elif evidence_region.required_context:
                try:
                    proposed_crop = propose_context_preserving_crop(
                        evidence_region.target_box,
                        [b for _, b in evidence_region.required_context],
                        width=image_width,
                        height=image_height
                    )
                    assessment = assess_crop(proposed_crop, evidence_region, image_width, image_height)
                    if not assessment.valid:
                        acquisition_valid = False
                        rejection_reason = f"Crop omits required context: {assessment.missing_context}"
                except Exception as e:
                    acquisition_valid = False
                    rejection_reason = f"Failed to form valid context crop: {e}"

        candidates[Action.REGROUND.value] = {
            "p_hat": p_stop + delta_p_hat,
            "cost_hat": incremental_cost,
            "delta_p": delta_p_hat,
            "sigma": sigma_hat,
            "score": acquisition_score,
            "valid": acquisition_valid,
            "rejection_reason": rejection_reason
        }

        candidates[Action.STOP.value] = {
            "p_hat": p_stop,
            "cost_hat": 0.0,
            "delta_p": 0.0,
            "sigma": 0.0,
            "score": 0.0,
            "valid": True
        }

        candidates[Action.ABSTAIN.value] = {
            "p_hat": 0.0,
            "cost_hat": 0.0,
            "delta_p": -p_stop,
            "sigma": 0.0,
            "score": -1.0 if p_stop >= self.tau_safe else 0.05,
            "valid": True
        }

        # 4. Stopping Decision Logic
        threshold = 0.01  # Minimum certified positive margin
        if acquisition_valid and acquisition_score > threshold:
            chosen = Action.REGROUND
            decision_path = "beneficial_context_crop_acquired"
        else:
            # Intervention does not justify cost or is geometrically invalid
            if p_stop < self.tau_safe:
                chosen = Action.ABSTAIN
                decision_path = "insufficient_evidence_safe_abstention"
            else:
                chosen = Action.STOP
                decision_path = "sufficient_evidence_conservative_stop"

        metadata = {
            "decision_path": decision_path,
            "chosen_action": chosen.value,
            "p_stop": p_stop,
            "acquisition_score": acquisition_score,
            "acquisition_valid": acquisition_valid,
            "rejection_reason": rejection_reason,
            "candidates": candidates
        }
        return chosen, metadata


class CCEANonCircularRouter:
    """
    Non-Circular, Evidence-Grounded Router for CCEA (C5 Policy).

    SCIENTIFIC METHODOLOGY & CONTRACT:
    1. Heuristic scores are uncalibrated ranking indices (NOT probabilities or calibrated confidences).
    2. Image resolution (width x height) is NEVER used as an uncertainty signal and never forces acquisition.
    3. Answer fluency does NOT imply factual correctness; factual status is marked as 'UNKNOWN'.
    4. Explicit linguistic uncertainty markers ('unclear', 'blurry', etc.) or empty answers trigger REGROUND
       if and only if spatial localization is structurally valid.
    5. Invalid or missing localization follows an explicit safe fallback: ABSTAIN if uncertain, STOP if fluent.
    """

    def __init__(
        self,
        alpha_evidence: float = 0.80,    # Weight on observable evidence uncertainty
        lambda_cost: float = 0.25,       # Weight on normalized token cost
        tau_reground: float = 0.12,      # Priority threshold to acquire crop
        tau_stop: float = 0.02,          # Priority threshold below which STOP is selected
        max_steps: int = 2,
    ):
        self.alpha_evidence = alpha_evidence
        self.lambda_cost = lambda_cost
        self.tau_reground = tau_reground
        self.tau_stop = tau_stop
        self.max_steps = max_steps

    def evaluate_overview_answer(self, answer_text: Optional[str]) -> Tuple[float, str]:
        """
        Extracts observable evidence uncertainty index u_evidence in [0.0, 1.0].
        Returns (u_evidence, factual_status).
        NOTE: Fluent answers are classified as 'FLUENT_BUT_FACTUALLY_UNKNOWN' - fluency != correctness.
        """
        ans_clean = (answer_text or "").strip().lower()
        hedging_markers = ["unclear", "cannot determine", "not visible", "unreadable", "cut off", "blurry", "illegible"]

        if len(ans_clean) == 0:
            return 0.90, "EMPTY_OR_MISSING"

        if any(marker in ans_clean for marker in hedging_markers):
            return 0.75, "EXPLICIT_HEDGING"

        if len(ans_clean) < 2:
            return 0.50, "ABBREVIATED_UNKNOWN"

        return 0.15, "FLUENT_BUT_FACTUALLY_UNKNOWN"

    def evaluate_localization(
        self,
        evidence_region: Optional[EvidenceRegion],
        image_width: int,
        image_height: int,
    ) -> Tuple[float, bool, Optional[str]]:
        """
        Evaluates geometric localization quality without oracle knowledge.
        Returns (g_loc, is_valid, rejection_reason).
        """
        if not evidence_region or not evidence_region.target_box:
            return 0.05, False, "Missing evidence region or target box"

        is_valid_tgt, tgt_err = validate_box(evidence_region.target_box, image_width, image_height)
        if not is_valid_tgt:
            return 0.05, False, f"Target box invalid: {tgt_err}"

        if evidence_region.required_context:
            try:
                proposed_crop = propose_context_preserving_crop(
                    evidence_region.target_box,
                    [b for _, b in evidence_region.required_context],
                    width=image_width,
                    height=image_height,
                )
                assessment = assess_crop(proposed_crop, evidence_region, image_width, image_height)
                if not assessment.valid:
                    return 0.30, False, f"Crop omits required context: {assessment.missing_context}"
                return 0.85, True, None
            except Exception as e:
                return 0.10, False, f"Failed to form valid context crop: {e}"

        return 0.60, True, None

    def decide_acquisition(
        self,
        overview_answer: Optional[str],
        evidence_region: Optional[EvidenceRegion],
        image_width: int,
        image_height: int,
        estimated_crop_tokens: int = 1400,
        remaining_budget: int = 2000,
        step_index: int = 0,
    ) -> Tuple[Action, Dict[str, Any]]:
        """
        Executes non-circular routing decision.
        Returns: (chosen_action, metadata_dict)
        """
        u_evidence, factual_status = self.evaluate_overview_answer(overview_answer)
        g_loc, acquisition_valid, rejection_reason = self.evaluate_localization(
            evidence_region, image_width, image_height
        )

        c_norm = min(1.0, max(0.20, estimated_crop_tokens / 2860.0))
        acquisition_priority_score = (self.alpha_evidence * u_evidence * g_loc) - (self.lambda_cost * c_norm)

        # Baseline heuristic priority score (uncalibrated ranking index)
        baseline_heuristic_score = max(0.05, min(0.95, 1.0 - u_evidence))

        # 1. Resource and step limits
        if step_index >= self.max_steps or remaining_budget < 150:
            action = Action.ABSTAIN if u_evidence >= 0.70 else Action.STOP
            path = "step_or_budget_limit_exhausted"
        # 2. Invalid or missing localization safe fallback
        elif not acquisition_valid:
            action = Action.ABSTAIN if u_evidence >= 0.70 else Action.STOP
            path = "invalid_localization_safe_fallback"
        # 3. Explicit uncertainty with valid localization
        elif u_evidence >= 0.70 and acquisition_valid:
            action = Action.REGROUND
            path = "uncertain_answer_with_valid_localization_reground"
        # 4. Priority threshold
        elif acquisition_priority_score >= self.tau_reground:
            action = Action.REGROUND
            path = "high_priority_reground"
        # 5. Low uncertainty or cost-prohibitive
        else:
            action = Action.STOP
            path = "low_uncertainty_or_cost_prohibitive_stop"

        metadata = {
            "decision_path": path,
            "chosen_action": action.value,
            "factual_status": factual_status,
            "u_evidence": u_evidence,
            "g_loc": g_loc,
            "c_norm": c_norm,
            "acquisition_priority_score": acquisition_priority_score,
            "baseline_heuristic_score": baseline_heuristic_score,
            "acquisition_valid": acquisition_valid,
            "rejection_reason": rejection_reason,
            "is_calibrated_probability": False,
        }
        return action, metadata

