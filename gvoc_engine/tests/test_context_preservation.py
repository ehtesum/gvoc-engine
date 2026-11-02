"""Unit tests for context-preserving crop validation."""
import unittest
from gvoc_engine.interventions.context_preservation import (
    ContextKind, EvidenceRegion, assess_crop, expand_box,
    propose_context_preserving_crop,
)


class TestContextPreservation(unittest.TestCase):
    def setUp(self):
        self.evidence = EvidenceRegion(
            target_box=(100, 150, 130, 190),
            required_context=(
                (ContextKind.COLUMN_LABEL, (40, 145, 70, 205)),
                (ContextKind.ROW_LABEL, (100, 80, 130, 140)),
                (ContextKind.UNIT, (10, 80, 35, 205)),
            ),
            evidence_id="table-cell",
        )

    def test_full_context_crop_is_valid(self):
        result = assess_crop((0, 60, 150, 220), self.evidence, 300, 200)
        self.assertTrue(result.valid)
        self.assertEqual(result.missing_context, ())

    def test_target_only_crop_is_rejected(self):
        result = assess_crop((95, 145, 135, 195), self.evidence, 300, 200)
        self.assertFalse(result.valid)
        self.assertTrue(result.target_retained)
        self.assertEqual(set(result.missing_context), {
            ContextKind.COLUMN_LABEL, ContextKind.ROW_LABEL, ContextKind.UNIT
        })

    def test_proposed_crop_covers_target_and_context(self):
        crop = propose_context_preserving_crop(
            self.evidence.target_box,
            [b for _, b in self.evidence.required_context],
            width=300, height=200, margin_fraction=0.0,
        )
        result = assess_crop(crop, self.evidence, 300, 200)
        self.assertTrue(result.valid)

    def test_expansion_clips_to_image(self):
        self.assertEqual(expand_box((0, 0, 20, 30), 100, 80, 0.5), (0, 0, 30, 45))

    def test_invalid_crop_fails_closed(self):
        with self.assertRaises(ValueError):
            assess_crop((10, 10, 10, 30), self.evidence, 300, 200)


if __name__ == "__main__":
    unittest.main()
