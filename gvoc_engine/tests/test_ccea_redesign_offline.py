"""
Targeted Offline Test Suite for CCEA Scientific Redesign Protocol.

Tests verify the non-circular routing logic, fair baseline mechanics,
honest error/uncertainty accounting, and failure mode handling.

STRICTLY OFFLINE: Zero live API calls, zero network sockets, zero credit spend.
"""

import unittest
from typing import Dict, Any, Optional

from gvoc_engine.core.action_space import Action
from gvoc_engine.interventions.context_preservation import (
    ContextKind,
    EvidenceRegion,
)
from gvoc_engine.controllers.ccea_router import CCEANonCircularRouter
from discovery.live_validation.scoring import (
    score_vdu_strict,
    score_vdu_unit_sensitive,
    compute_token_f1,
)


class MockProtocolRunner:
    """
    Mock runner simulating C1, C2, C3, C4, and C5 conditions offline
    with identical model configurations and scoring functions.
    """
    FROZEN_MODEL = "mistral/pixtral-12b-2409"
    FROZEN_TEMPERATURE = 0.0
    FROZEN_MAX_TOKENS = 60

    def __init__(self, router: Optional[CCEANonCircularRouter] = None):
        self.router = router or CCEANonCircularRouter()

    def run_c1_overview(self, overview_output: str, ground_truth: str) -> Dict[str, Any]:
        """C1: Exactly 1 call on full image."""
        pred = overview_output
        f1, em = score_vdu_strict(pred, ground_truth)
        return {
            "condition": "C1_full_overview",
            "model": self.FROZEN_MODEL,
            "calls": 1,
            "prediction": pred,
            "strict_em": float(em),
            "f1": f1,
        }

    def run_c2_fixed_crop(self, crop_output: str, ground_truth: str) -> Dict[str, Any]:
        """C2: Exactly 1 call on fixed context crop (diagnostic)."""
        pred = crop_output
        f1, em = score_vdu_strict(pred, ground_truth)
        return {
            "condition": "C2_fixed_crop_only",
            "model": self.FROZEN_MODEL,
            "calls": 1,
            "prediction": pred,
            "strict_em": float(em),
            "f1": f1,
        }

    def run_c3_overview_plus_fixed_crop(
        self, overview_output: str, crop_output: str, ground_truth: str
    ) -> Dict[str, Any]:
        """C3: Exactly 2 calls mandatory (Overview + Fixed Crop). Primary Fair Comparator."""
        pred = crop_output
        f1, em = score_vdu_strict(pred, ground_truth)
        return {
            "condition": "C3_overview_plus_fixed_crop",
            "model": self.FROZEN_MODEL,
            "calls": 2,
            "prediction": pred,
            "strict_em": float(em),
            "f1": f1,
        }

    def run_c4_resolution_only(
        self,
        overview_output: str,
        crop_output: str,
        width: int,
        height: int,
        ground_truth: str,
        tau_res_pixels: int = 2_500_000,
    ) -> Dict[str, Any]:
        """C4: Resolution-only baseline. Acquires crop iff pixels >= tau_res."""
        total_pixels = width * height
        if total_pixels >= tau_res_pixels:
            calls = 2
            pred = crop_output
            decision = "ACQUIRE_BY_RESOLUTION"
        else:
            calls = 1
            pred = overview_output
            decision = "STOP_BY_RESOLUTION"

        f1, em = score_vdu_strict(pred, ground_truth)
        return {
            "condition": "C4_resolution_only",
            "model": self.FROZEN_MODEL,
            "calls": calls,
            "decision": decision,
            "prediction": pred,
            "strict_em": float(em),
            "f1": f1,
        }

    def run_c5_revised_ccea(
        self,
        overview_output: str,
        crop_output: str,
        evidence_region: Optional[EvidenceRegion],
        width: int,
        height: int,
        ground_truth: str,
    ) -> Dict[str, Any]:
        """C5: Revised CCEA policy. 1 or 2 calls based on evidence signals."""
        action, meta = self.router.decide_acquisition(
            overview_answer=overview_output,
            evidence_region=evidence_region,
            image_width=width,
            image_height=height,
        )

        if action == Action.REGROUND:
            calls = 2
            pred = crop_output
        elif action == Action.STOP:
            calls = 1
            pred = overview_output
        else:  # Action.ABSTAIN
            calls = 1
            pred = ""  # Abstention scored as failure in standard accuracy

        f1, em = score_vdu_strict(pred, ground_truth)
        return {
            "condition": "C5_revised_ccea",
            "model": self.FROZEN_MODEL,
            "action": action.value,
            "calls": calls,
            "prediction": pred,
            "strict_em": float(em),
            "f1": f1,
            "is_abstention": (action == Action.ABSTAIN),
            "meta": meta,
        }



class TestCCEARedesignOffline(unittest.TestCase):
    """Offline test suite confirming scientific redesign requirements."""

    def setUp(self):
        self.router = CCEANonCircularRouter()
        self.runner = MockProtocolRunner(self.router)
        self.valid_evidence = EvidenceRegion(
            target_box=(100, 150, 140, 200),
            required_context=(
                (ContextKind.COLUMN_LABEL, (40, 150, 80, 200)),
                (ContextKind.ROW_LABEL, (100, 80, 140, 140)),
            ),
            evidence_id="valid_table_region"
        )

    def test_clean_fluent_answer_not_presumed_correct(self):
        """
        1. A clean-looking overview answer must not be presumed correct solely because it is fluent.
        Factual correctness must be marked as UNKNOWN, and fluency must not guarantee truth.
        """
        fluent_wrong_answer = "June 14, 1998"
        oracle_ground_truth = "September 21, 1994"

        # Router processes overview
        action, meta = self.router.decide_acquisition(
            overview_answer=fluent_wrong_answer,
            evidence_region=self.valid_evidence,
            image_width=1000,
            image_height=800,
        )

        # Telemetry contract verification
        self.assertEqual(meta["factual_status"], "FLUENT_BUT_FACTUALLY_UNKNOWN")
        self.assertFalse(meta["is_calibrated_probability"])
        self.assertEqual(action, Action.STOP)

        # Evaluator verification: Fluency does NOT prevent accurate scoring of factual incorrectness
        c1_res = self.runner.run_c1_overview(fluent_wrong_answer, oracle_ground_truth)
        self.assertEqual(c1_res["strict_em"], 0.0)
        self.assertLess(c1_res["f1"], 0.50)

    def test_high_resolution_does_not_force_reground(self):
        """
        2. A high-resolution image must not automatically trigger REGROUND.
        Resolution is strictly isolated; a confident fluent answer on a 7.5M px image selects STOP.
        """
        high_res_width = 3000
        high_res_height = 2500  # 7,500,000 pixels (previously forced d_layout = 0.28)
        fluent_answer = "Total operating revenue was $4,250,000."

        action, meta = self.router.decide_acquisition(
            overview_answer=fluent_answer,
            evidence_region=self.valid_evidence,
            image_width=high_res_width,
            image_height=high_res_height,
        )

        self.assertEqual(action, Action.STOP)
        self.assertEqual(meta["decision_path"], "low_uncertainty_or_cost_prohibitive_stop")
        self.assertLess(meta["acquisition_priority_score"], self.router.tau_reground)

    def test_low_resolution_with_explicit_uncertainty_triggers_reground(self):
        """
        3. A low-resolution image with explicit answer uncertainty and valid localization triggers REGROUND.
        Shows that resolution does not dictate decision; evidence uncertainty governs rescue.
        """
        low_res_width = 600
        low_res_height = 800  # 480,000 pixels
        hedging_answer = "The invoice date is unclear and cannot be determined from the image."

        action, meta = self.router.decide_acquisition(
            overview_answer=hedging_answer,
            evidence_region=self.valid_evidence,
            image_width=low_res_width,
            image_height=low_res_height,
        )

        self.assertEqual(action, Action.REGROUND)
        self.assertEqual(meta["decision_path"], "uncertain_answer_with_valid_localization_reground")
        self.assertEqual(meta["factual_status"], "EXPLICIT_HEDGING")
        self.assertTrue(meta["acquisition_valid"])

    def test_invalid_or_missing_localization_safe_path(self):
        """
        4. Invalid or missing localization must follow an explicit safe path:
        - When answer is uncertain and localization is invalid -> ABSTAIN.
        - When answer is fluent and localization is invalid -> STOP.
        - In both cases, REGROUND must NEVER be selected.
        """
        invalid_evidence = EvidenceRegion(
            target_box=(100, 150, 1500, 200),  # ymax=1500 > image_height=800 (out of bounds)
            required_context=(),
            evidence_id="invalid_oob_box"
        )

        # Case A: Uncertain answer + Invalid localization -> ABSTAIN
        action_a, meta_a = self.router.decide_acquisition(
            overview_answer="The signature is blurry and not visible.",
            evidence_region=invalid_evidence,
            image_width=1000,
            image_height=800,
        )
        self.assertEqual(action_a, Action.ABSTAIN)
        self.assertFalse(meta_a["acquisition_valid"])
        self.assertIn("Target box invalid", meta_a["rejection_reason"])

        # Case B: Fluent answer + Invalid localization -> STOP
        action_b, meta_b = self.router.decide_acquisition(
            overview_answer="Form 1040",
            evidence_region=invalid_evidence,
            image_width=1000,
            image_height=800,
        )
        self.assertEqual(action_b, Action.STOP)
        self.assertFalse(meta_b["acquisition_valid"])

        # Case C: None evidence region + Uncertain answer -> ABSTAIN
        action_c, meta_c = self.router.decide_acquisition(
            overview_answer="",
            evidence_region=None,
            image_width=1000,
            image_height=800,
        )
        self.assertEqual(action_c, Action.ABSTAIN)
        self.assertFalse(meta_c["acquisition_valid"])
        self.assertIn("Missing evidence region", meta_c["rejection_reason"])

    def test_consistent_scoring_and_request_accounting(self):
        """
        5. STOP, REGROUND, and ABSTAIN must have consistent scoring and request accounting:
        - STOP: exactly 1 request; overview answer scored.
        - REGROUND: exactly 2 requests; crop refinement answer scored.
        - ABSTAIN: exactly 1 request; prediction is empty string, scored as strict EM = 0.0, F1 = 0.0.
        """
        ground_truth = "March 15, 2021"

        # STOP accounting
        res_stop = self.runner.run_c5_revised_ccea(
            overview_output="March 15, 2021",
            crop_output="March 15, 2021",
            evidence_region=self.valid_evidence,
            width=1000,
            height=1000,
            ground_truth=ground_truth,
        )
        self.assertEqual(res_stop["action"], Action.STOP.value)
        self.assertEqual(res_stop["calls"], 1)
        self.assertEqual(res_stop["strict_em"], 1.0)
        self.assertEqual(res_stop["f1"], 1.0)

        # REGROUND accounting
        res_reground = self.runner.run_c5_revised_ccea(
            overview_output="The date is blurry",
            crop_output="March 15, 2021",
            evidence_region=self.valid_evidence,
            width=1000,
            height=1000,
            ground_truth=ground_truth,
        )
        self.assertEqual(res_reground["action"], Action.REGROUND.value)
        self.assertEqual(res_reground["calls"], 2)
        self.assertEqual(res_reground["prediction"], "March 15, 2021")
        self.assertEqual(res_reground["strict_em"], 1.0)

        # ABSTAIN accounting
        res_abstain = self.runner.run_c5_revised_ccea(
            overview_output="The date is blurry",
            crop_output="March 15, 2021",
            evidence_region=None,  # Missing localization forces abstention
            width=1000,
            height=1000,
            ground_truth=ground_truth,
        )
        self.assertEqual(res_abstain["action"], Action.ABSTAIN.value)
        self.assertEqual(res_abstain["calls"], 1)
        self.assertEqual(res_abstain["prediction"], "")
        self.assertEqual(res_abstain["strict_em"], 0.0)
        self.assertEqual(res_abstain["f1"], 0.0)
        self.assertTrue(res_abstain["is_abstention"])

    def test_c3_always_executes_second_call_and_c5_conditionally(self):
        """
        6. C3 must always execute the planned second call in the mock runner,
        while C5 must execute it only when its rule selects REGROUND.
        """
        gt = "42"

        # Case A: Clean overview where C5 selects STOP
        c3_a = self.runner.run_c3_overview_plus_fixed_crop("42", "42", gt)
        c5_a = self.runner.run_c5_revised_ccea("42", "42", self.valid_evidence, 1000, 1000, gt)
        self.assertEqual(c3_a["calls"], 2)
        self.assertEqual(c5_a["calls"], 1)  # C5 saved 1 call via STOP

        # Case B: Hedging overview where C5 selects REGROUND
        c3_b = self.runner.run_c3_overview_plus_fixed_crop("unclear", "42", gt)
        c5_b = self.runner.run_c5_revised_ccea("unclear", "42", self.valid_evidence, 1000, 1000, gt)
        self.assertEqual(c3_b["calls"], 2)
        self.assertEqual(c5_b["calls"], 2)  # C5 matches C3 calls via REGROUND

        # Case C: Hedging overview where C5 selects ABSTAIN due to failed localization
        c3_c = self.runner.run_c3_overview_plus_fixed_crop("unclear", "42", gt)
        c5_c = self.runner.run_c5_revised_ccea("unclear", "42", None, 1000, 1000, gt)
        self.assertEqual(c3_c["calls"], 2)
        self.assertEqual(c5_c["calls"], 1)  # C5 saved 1 call via ABSTAIN

    def test_all_five_conditions_matched_configuration(self):
        """
        7. All five conditions must use the frozen model configuration and same scoring implementation.
        """
        gt = "Acme Corp"
        w, h = 1200, 1600
        overview_text = "Acme Corp"
        crop_text = "Acme Corp"

        c1 = self.runner.run_c1_overview(overview_text, gt)
        c2 = self.runner.run_c2_fixed_crop(crop_text, gt)
        c3 = self.runner.run_c3_overview_plus_fixed_crop(overview_text, crop_text, gt)
        c4 = self.runner.run_c4_resolution_only(overview_text, crop_text, w, h, gt)
        c5 = self.runner.run_c5_revised_ccea(overview_text, crop_text, self.valid_evidence, w, h, gt)

        for res in [c1, c2, c3, c4, c5]:
            self.assertEqual(res["model"], "mistral/pixtral-12b-2409")
            self.assertEqual(res["strict_em"], 1.0)
            self.assertEqual(res["f1"], 1.0)



if __name__ == "__main__":
    unittest.main()
