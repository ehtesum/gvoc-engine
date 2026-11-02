"""
Unit tests for CCEA geometry, context preservation, and coordinate parsing.
"""

import unittest
from gvoc_engine.interventions.context_preservation import (
    ContextKind,
    EvidenceRegion,
    assess_crop,
    expand_box,
    propose_context_preserving_crop,
    validate_box,
    parse_bounding_box,
    format_box,
)


class TestCCEAGeometryAndContext(unittest.TestCase):
    def setUp(self):
        # 300x200 image space (height=200, width=300)
        self.evidence = EvidenceRegion(
            target_box=(100, 150, 130, 190),  # Table cell
            required_context=(
                (ContextKind.COLUMN_LABEL, (40, 145, 70, 205)),
                (ContextKind.ROW_LABEL, (100, 80, 130, 140)),
                (ContextKind.UNIT, (10, 80, 35, 205)),
                (ContextKind.FOOTNOTE, (170, 80, 190, 250)),
                (ContextKind.AXIS_TICK, (95, 50, 135, 75)),
                (ContextKind.LEGEND, (15, 10, 35, 70)),
            ),
            evidence_id="table_with_multi_context",
        )

    def test_valid_crop_retains_all_required_context(self):
        # Box covering [0, 0, 200, 300] fully covers target and all contexts
        result = assess_crop((0, 0, 200, 300), self.evidence, 300, 200)
        self.assertTrue(result.valid)
        self.assertTrue(result.target_retained)
        self.assertEqual(len(result.retained_context), 6)
        self.assertEqual(result.missing_context, ())
        self.assertAlmostEqual(result.coverage, 1.0)

    def test_target_only_crop_fails_all_contexts(self):
        # Tightly cropped to target only
        target_crop = (100, 150, 130, 190)
        result = assess_crop(target_crop, self.evidence, 300, 200)
        self.assertFalse(result.valid)
        self.assertTrue(result.target_retained)
        self.assertEqual(len(result.missing_context), 6)
        self.assertIn(ContextKind.COLUMN_LABEL, result.missing_context)
        self.assertIn(ContextKind.ROW_LABEL, result.missing_context)
        self.assertIn(ContextKind.UNIT, result.missing_context)

    def test_missing_single_unit_rejects_crop(self):
        # Crop includes target, column label, row label, but cuts off UNIT at top (ymin=10)
        crop_omitting_unit = (38, 70, 195, 255)
        result = assess_crop(crop_omitting_unit, self.evidence, 300, 200)
        self.assertFalse(result.valid)
        self.assertIn(ContextKind.UNIT, result.missing_context)

    def test_crop_touching_image_boundaries(self):
        # Image is width=300, height=200
        boundary_crop = (0, 0, 200, 300)
        valid, err = validate_box(boundary_crop, 300, 200)
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_invalid_reversed_and_empty_boxes_fail_closed(self):
        # ymin >= ymax
        valid, err = validate_box((100, 10, 50, 80), 300, 200)
        self.assertFalse(valid)
        self.assertIn("ymin=100 >= ymax=50", err)

        # xmin >= xmax
        valid, err = validate_box((10, 100, 50, 50), 300, 200)
        self.assertFalse(valid)
        self.assertIn("xmin=100 >= xmax=50", err)

        # Zero area
        valid, err = validate_box((50, 50, 50, 80), 300, 200)
        self.assertFalse(valid)

        # Negative coordinates
        valid, err = validate_box((-10, 10, 50, 80), 300, 200)
        self.assertFalse(valid)

        # Exceeds image dimensions
        valid, err = validate_box((10, 10, 250, 100), 300, 200)  # height is 200
        self.assertFalse(valid)
        self.assertIn("exceeds image height=200", err)

    def test_proposed_context_crop_covers_target_and_context(self):
        crop = propose_context_preserving_crop(
            self.evidence.target_box,
            [b for _, b in self.evidence.required_context],
            width=300,
            height=200,
            margin_fraction=0.02,
        )
        res = assess_crop(crop, self.evidence, 300, 200)
        self.assertTrue(res.valid)
        self.assertEqual(res.missing_context, ())

    def test_missing_context_annotations_evaluates_target_only(self):
        empty_evidence = EvidenceRegion(
            target_box=(20, 30, 80, 90),
            required_context=(),
            evidence_id="no_context",
        )
        res = assess_crop((20, 30, 80, 90), empty_evidence, 300, 200)
        self.assertTrue(res.valid)
        self.assertTrue(res.target_retained)
        self.assertEqual(res.missing_context, ())

    def test_parse_bounding_box_formats(self):
        width, height = 1000, 800

        # Raw list
        box, msg = parse_bounding_box([100, 200, 400, 600], width, height)
        self.assertEqual(box, (100, 200, 400, 600))

        # Dict format
        box, msg = parse_bounding_box({"bbox": [50, 50, 200, 300]}, width, height)
        self.assertEqual(box, (50, 50, 200, 300))

        # JSON markdown code block
        md_text = "Here is the box:\n```json\n[150, 100, 500, 450]\n```"
        box, msg = parse_bounding_box(md_text, width, height)
        self.assertEqual(box, (150, 100, 500, 450))

        # Plain text with brackets
        raw_text = "The target is located at [80, 120, 300, 400]."
        box, msg = parse_bounding_box(raw_text, width, height)
        self.assertEqual(box, (80, 120, 300, 400))

        # Normalized coordinates scaled to pixels
        norm_text = "[0.10, 0.20, 0.50, 0.80]"
        box, msg = parse_bounding_box(norm_text, width, height)
        # y0=80, x0=200, y1=400, x1=800
        self.assertEqual(box, (80, 200, 400, 800))

        # Malformed string fails closed
        box, msg = parse_bounding_box("I could not find the box.", width, height)
        self.assertIsNone(box)
        self.assertIn("Could not extract", msg)


if __name__ == "__main__":
    unittest.main()
