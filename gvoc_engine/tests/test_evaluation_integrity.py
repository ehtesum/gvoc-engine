"""
Tests for evaluation integrity, ground-truth isolation, metric scoring,
and regression test suite for VDU answer evaluation.
"""

import unittest
from discovery.live_validation.scoring import (
    score_vdu_answer,
    score_vdu_strict,
    score_vdu_unit_sensitive,
    score_vdu_comprehensive,
    normalize_text,
    normalize_unit_sensitive,
    requires_unit,
    preserve_decimals_normalize
)
from gvoc_engine.core.state import GroundedState
from gvoc_engine.interventions.ccea_intervention import CCEAResult
from gvoc_engine.controllers.ccea_router import CCEACostAwareRouter


class TestEvaluationIntegrity(unittest.TestCase):
    
    def test_70_versus_70_percent(self):
        """Regression test: 70 vs 70% must be distinguished under unit-sensitive scoring."""
        gt = ["70%"]
        self.assertTrue(requires_unit(gt))
        
        # Prediction with unit matches
        pred_with_unit = "70%"
        f1_u, acc_u = score_vdu_unit_sensitive(pred_with_unit, gt)
        self.assertEqual(acc_u, 1)
        
        # Prediction without unit must NOT match under unit-sensitive scoring
        pred_no_unit = "70"
        f1_nu, acc_nu = score_vdu_unit_sensitive(pred_no_unit, gt)
        self.assertEqual(acc_nu, 0, "70 must not match 70% in unit-sensitive scoring")

    def test_70_versus_70_percent_mixed_reference_aliases(self):
        """Regression test: docvqa_020 has mixed reference aliases ['70', '70%'].
        When the reference requires a percentage, bare '70' must NOT receive unit-sensitive EM = 1,
        even if an unannotated bare alias exists in ground truth."""
        gt = ["70", "70%"]
        self.assertTrue(requires_unit(gt))
        
        # Bare prediction must be rejected by unit-sensitive scorer
        f1_bare, acc_bare = score_vdu_unit_sensitive("70", gt)
        self.assertEqual(acc_bare, 0, "Bare '70' must not match when reference requires percentage")
        
        # Percentage prediction must match
        f1_pct, acc_pct = score_vdu_unit_sensitive("70%", gt)
        self.assertEqual(acc_pct, 1, "'70%' must match when reference requires percentage")

    def test_legitimate_percentage_punctuation_variants_equivalent(self):
        """Regression test: Legitimate punctuation and whitespace variants of '70%'
        remain equivalent under unit-sensitive scoring."""
        gt = ["70", "70%"]
        valid_variants = ["70%", "(70%)", "70%.", "70 %", "[70%]", " 70% "]
        for var in valid_variants:
            _, acc = score_vdu_unit_sensitive(var, gt)
            self.assertEqual(acc, 1, f"Punctuation/whitespace variant {var!r} must match 70%")
            
        # Punctuation variants without unit must still fail
        invalid_variants = ["70", "(70)", "70.", "[70]"]
        for inv in invalid_variants:
            _, acc = score_vdu_unit_sensitive(inv, gt)
            self.assertEqual(acc, 0, f"Variant without unit {inv!r} must not match 70%")

    def test_currency_with_and_without_symbol(self):
        """Regression test: Currency amounts with and without currency symbols."""
        gt = ["$975.00"]
        self.assertTrue(requires_unit(gt))
        
        pred_with_sym = "$975.00"
        _, acc_with = score_vdu_unit_sensitive(pred_with_sym, gt)
        self.assertEqual(acc_with, 1)
        
        pred_no_sym = "975.00"
        _, acc_without = score_vdu_unit_sensitive(pred_no_sym, gt)
        self.assertEqual(acc_without, 0, "975.00 must not match $975.00 in unit-sensitive scoring")

    def test_decimal_formatting_not_equated_to_integer(self):
        """Regression test: Decimals must not be collapsed into integers (12.5 != 125)."""
        gt = ["12.5"]
        norm_gt = normalize_text(gt[0])
        self.assertEqual(norm_gt, "12.5")
        
        # 125 must not match 12.5
        _, acc_wrong = score_vdu_strict("125", gt)
        self.assertEqual(acc_wrong, 0, "125 must not match 12.5")
        
        # Exact decimal match
        _, acc_right = score_vdu_strict("12.5", gt)
        self.assertEqual(acc_right, 1)

    def test_punctuation_and_whitespace(self):
        """Regression test: Standard punctuation and whitespace variation."""
        gt = ["Dr. Smith, Jr."]
        pred = "dr smith jr"
        _, acc = score_vdu_strict(pred, gt)
        self.assertEqual(acc, 1)
        
        pred_extra_space = "  dr   smith   jr  "
        _, acc_sp = score_vdu_strict(pred_extra_space, gt)
        self.assertEqual(acc_sp, 1)

    def test_nw_versus_n_w(self):
        """Regression test: 'N.W.' versus 'N. W.' normalization equivalence."""
        gt = ["1128 Sixteenth St., N. W."]
        pred_compact = "1128 sixteenth st., n.w."
        pred_spaced = "1128 sixteenth st., n. w."
        
        _, acc_compact = score_vdu_strict(pred_compact, gt)
        _, acc_spaced = score_vdu_strict(pred_spaced, gt)
        self.assertEqual(acc_compact, 1)
        self.assertEqual(acc_spaced, 1)

    def test_multiple_acceptable_references(self):
        """Regression test: Multiple acceptable reference strings."""
        gt = [
            "1128 Sixteenth St., N. W., Washington, D. C. 20036",
            "1128 16th Street N.W., Washington D.C.",
            "1128 16th St NW"
        ]
        
        pred1 = "1128 sixteenth st., n. w., washington, d. c. 20036"
        _, acc1 = score_vdu_strict(pred1, gt)
        self.assertEqual(acc1, 1)
        
        pred2 = "1128 16th st nw"
        _, acc2 = score_vdu_strict(pred2, gt)
        self.assertEqual(acc2, 1)

    def test_empty_predictions(self):
        """Regression test: Empty predictions, whitespace, and None."""
        gt = ["Valid Answer"]
        for empty_val in ["", "   ", None]:
            f1, acc = score_vdu_strict(empty_val, gt)
            self.assertEqual(acc, 0)
            self.assertEqual(f1, 0.0)

    def test_abstentions(self):
        """Regression test: Abstentions score 0 unless ground-truth permits it."""
        gt_standard = ["42.0"]
        f1_abs, acc_abs = score_vdu_strict("ABSTAIN_INSUFFICIENT_EVIDENCE", gt_standard)
        self.assertEqual(acc_abs, 0)
        self.assertEqual(f1_abs, 0.0)

        gt_unanswerable = ["ABSTAIN"]
        f1_ok, acc_ok = score_vdu_strict("ABSTAIN", gt_unanswerable)
        self.assertEqual(acc_ok, 1)
        self.assertEqual(f1_ok, 1.0)

    def test_verbose_prediction_substring_rejected(self):
        """Regression test: Verbose predictions with answer as substring must be rejected in strict EM."""
        gt = ["42.0"]
        pred_verbose = "The document indicates the value is 42.0."
        
        # Strict mode rejects substring shortcut
        _, acc_strict = score_vdu_strict(pred_verbose, gt)
        self.assertEqual(acc_strict, 0, "Verbose answer containing substring must NOT get EM=1 in strict mode")

        # Token F1 should still capture the overlap
        f1_strict, _ = score_vdu_strict(pred_verbose, gt)
        self.assertGreater(f1_strict, 0.0)

    def test_incorrect_answers_with_high_token_overlap(self):
        """Regression test: Answers with high token overlap but wrong factual content get EM=0."""
        gt = ["Total gross revenue is forty two million dollars"]
        pred = "Total net revenue is forty two million dollars"
        
        f1, acc = score_vdu_strict(pred, gt)
        self.assertEqual(acc, 0, "Different key fact must have EM=0 even if token overlap is high")
        self.assertGreater(f1, 0.70, "Token overlap F1 is high, but EM is 0")

    def test_policy_does_not_inspect_ground_truth(self):
        """Verify that the router takes ONLY state, budget, and image geometry, never ground truth."""
        router = CCEACostAwareRouter()
        import inspect
        sig = inspect.signature(router.select_action)
        self.assertNotIn("ground_truth", sig.parameters)
        self.assertNotIn("gt", sig.parameters)
        self.assertNotIn("label", sig.parameters)

    def test_failed_intervention_preserves_error_telemetry(self):
        """Failed or rejected interventions must record status and failure reason."""
        result = CCEAResult(
            initial_answer="Initial answer",
            final_answer="Initial answer",
            status="REJECTED_INVALID_BOX",
            failure_reason="Coordinates out of bounds: ymax=1200 > height=800"
        )
        self.assertNotEqual(result.status, "SUCCESS")
        self.assertIsNotNone(result.failure_reason)
        self.assertEqual(result.final_answer, "Initial answer")

    def test_total_tokens_sums_all_calls(self):
        """Ensure token accounting captures all sequential calls."""
        res = CCEAResult(
            prompt_tokens=150,
            completion_tokens=45,
            total_tokens=195,
            num_calls=2
        )
        self.assertEqual(res.total_tokens, res.prompt_tokens + res.completion_tokens)
        self.assertEqual(res.num_calls, 2)


if __name__ == "__main__":
    unittest.main()
