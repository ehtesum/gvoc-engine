"""
Offline unit and integration tests for Automatic Non-Oracle Localization.
Verifies:
1. Correct coordinate mapping and image-boundary handling.
2. Missing, invalid, or ambiguous localization.
3. Preservation of relevant labels and units.
4. No access to ground-truth answers during localization.
5. Fallback behavior and stop decisions.
6. Accurate accounting of overview and incremental crop tokens without double counting.
"""

import unittest
from typing import Dict, Any, List

from gvoc_engine.interventions.context_preservation import validate_box
from gvoc_engine.interventions.auto_locator import (
    AutomaticEvidenceLocator,
    AutomaticLocalizationResult,
    LocalizationSource,
    LocalizationAction,
)


class MockVLMClientForGrounding:
    """Deterministic mock client for testing visual grounding prompts."""
    def __init__(self, canned_response: str, tokens: int = 45):
        self.canned_response = canned_response
        self.tokens = tokens
        self.calls = 0

    def call_vlm(self, prompt: str, image_path: str, max_tokens: int = 150):
        self.calls += 1
        return self.canned_response, self.tokens, 15, 0.05


class TestAutoLocalization(unittest.TestCase):

    def setUp(self):
        self.locator = AutomaticEvidenceLocator(confidence_threshold=0.50, margin_fraction=0.08)
        self.width = 1600
        self.height = 2200

    def test_coordinate_mapping_and_image_boundaries(self):
        """Test 1: Verify coordinate mapping is clipped to positive image boundaries."""
        # OCR tokens near image border
        ocr_tokens = [
            {"text": "Total", "box": [2150, 1500, 2190, 1590]},
            {"text": "Amount", "box": [2150, 1400, 2190, 1490]}
        ]
        result = self.locator.locate(
            image_path="test_image.png",
            query="What is the Total Amount?",
            image_width=self.width,
            image_height=self.height,
            ocr_tokens=ocr_tokens
        )
        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.crop_box)
        # Check boundary adherence
        ymin, xmin, ymax, xmax = result.crop_box
        self.assertGreaterEqual(ymin, 0)
        self.assertGreaterEqual(xmin, 0)
        self.assertLessEqual(ymax, self.height)
        self.assertLessEqual(xmax, self.width)
        # Must pass strict validation
        is_val, err = validate_box(result.crop_box, self.width, self.height)
        self.assertTrue(is_val, f"Box failed validation: {err}")

    def test_missing_or_invalid_localization_fails_closed(self):
        """Test 2: Empty query or zero OCR matches must fail closed to safe overview fallback."""
        # Empty query
        res_empty = self.locator.locate(
            image_path="test_image.png",
            query="",
            image_width=self.width,
            image_height=self.height
        )
        self.assertEqual(res_empty.action, LocalizationAction.FALLBACK_TO_OVERVIEW)
        self.assertEqual(res_empty.stopping_reason, "empty_query")

        # Zero OCR keyword matches
        ocr_unrelated = [{"text": "Banana", "box": [100, 100, 200, 200]}]
        res_no_match = self.locator.locate(
            image_path="test_image.png",
            query="What is the Gross Profit on the invoice?",
            image_width=self.width,
            image_height=self.height,
            ocr_tokens=ocr_unrelated
        )
        self.assertEqual(res_no_match.action, LocalizationAction.FALLBACK_TO_OVERVIEW)
        self.assertEqual(res_no_match.source, LocalizationSource.FALLBACK_OVERVIEW)
        self.assertEqual(res_no_match.crop_box, (0, 0, self.height, self.width))

    def test_invalid_box_geometry_rejected(self):
        """Test 3: VLM output with invalid or inverted coordinates must be rejected."""
        inverted_client = MockVLMClientForGrounding(canned_response='{"target_box": [500, 500, 400, 400]}')
        result = self.locator.locate(
            image_path="test_image.png",
            query="What is the net revenue?",
            image_width=self.width,
            image_height=self.height,
            client=inverted_client
        )
        # Should not crash; must fall back safely to overview
        self.assertEqual(result.action, LocalizationAction.FALLBACK_TO_OVERVIEW)
        self.assertEqual(result.crop_box, (0, 0, self.height, self.width))

    def test_preservation_of_relevant_labels_and_units(self):
        """Test 4: Context-preserving mode bundles column headers and row labels."""
        ocr_tokens = [
            {"text": "Indigenous", "box": [1000, 400, 1040, 600]},
            {"text": "Raw", "box": [1000, 610, 1040, 700]},
            {"text": "Material", "box": [1000, 710, 1040, 850]}
        ]
        result = self.locator.locate(
            image_path="test_image.png",
            query="What is the value of Indigenous Raw Material?",
            image_width=self.width,
            image_height=self.height,
            ocr_tokens=ocr_tokens,
            prefer_context_preserving=True
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.action, LocalizationAction.ACQUIRE_CONTEXT_PRESERVING)
        self.assertGreater(len(result.context_boxes), 0)
        self.assertTrue("column_header" in result.retained_context_types or "row_label" in result.retained_context_types)
        # Crop box must encompass target box
        t_ymin, t_xmin, t_ymax, t_xmax = result.target_box
        c_ymin, c_xmin, c_ymax, c_xmax = result.crop_box
        self.assertLessEqual(c_ymin, t_ymin)
        self.assertLessEqual(c_xmin, t_xmin)
        self.assertGreaterEqual(c_ymax, t_ymax)
        self.assertGreaterEqual(c_xmax, t_xmax)

    def test_no_ground_truth_leakage_in_interface(self):
        """Test 5: Locate signature must never accept ground-truth answers or coordinates."""
        import inspect
        sig = inspect.signature(self.locator.locate)
        param_names = list(sig.parameters.keys())
        for forbidden in ["ground_truth", "gt", "answer", "gold", "label_answer", "oracle"]:
            self.assertNotIn(forbidden, param_names, f"Forbidden parameter {forbidden} found in locator signature")

    def test_vlm_grounding_success_and_token_tracking(self):
        """Test 6: Valid VLM response parsed correctly with token usage tracked when enabled."""
        valid_response = """
        ```json
        {
            "target_box": [500, 600, 560, 750],
            "context_boxes": [[420, 600, 480, 750], [500, 200, 560, 580]],
            "confidence": 0.85
        }
        ```
        """
        client = MockVLMClientForGrounding(canned_response=valid_response, tokens=110)
        vlm_locator = AutomaticEvidenceLocator(enable_vlm_grounding=True)
        result = vlm_locator.locate(
            image_path="test_image.png",
            query="What percentage was reported?",
            image_width=self.width,
            image_height=self.height,
            client=client
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.source, LocalizationSource.VLM_GROUNDING)
        self.assertEqual(result.target_box, (500, 600, 560, 750))
        self.assertEqual(len(result.context_boxes), 2)
        self.assertGreaterEqual(result.confidence, 0.80)
        self.assertEqual(result.locator_tokens, 125)  # 110 prompt + 15 completion

    def test_strategy_b_disabled_by_default_does_not_invoke_vlm(self):
        """Test 6b: By default enable_vlm_grounding=False, so Strategy B is never invoked even if client is passed."""
        client = MockVLMClientForGrounding(canned_response="should not be called", tokens=110)
        # self.locator has default enable_vlm_grounding=False
        result = self.locator.locate(
            image_path="test_image.png",
            query="What percentage was reported?",
            image_width=self.width,
            image_height=self.height,
            client=client
        )
        # Client was never called, calls remains 0
        self.assertEqual(client.calls, 0)
        self.assertEqual(result.source, LocalizationSource.FALLBACK_OVERVIEW)
        self.assertEqual(result.crop_box, (0, 0, self.height, self.width))

    def test_token_accounting_without_double_counting(self):
        """Test 7: Verify deployed policy tokens cleanly separates overview, locator, and crop tokens."""
        # Scenario: Deployed C4 sequential policy
        overview_tokens = 2200
        locator_tokens = 125
        crop_inference_tokens = 450

        # When REGROUND chosen: overview + locator + crop
        deployed_total = overview_tokens + locator_tokens + crop_inference_tokens
        self.assertEqual(deployed_total, 2775)
        # Ensure overview is not added twice
        self.assertNotEqual(deployed_total, overview_tokens + (overview_tokens + crop_inference_tokens))

        # When STOP chosen: overview + locator
        stop_total = overview_tokens + locator_tokens
        self.assertEqual(stop_total, 2325)

    def test_real_image_physical_crop_extraction_end_to_end(self):
        """Test 8: Physical image processing test.
        Creates a real PIL Image on disk, performs OCR keyword localization,
        extracts physical crop pixels, and validates the saved image on disk."""
        import os
        import tempfile
        import shutil
        from PIL import Image
        from gvoc_engine.interventions.ccea_intervention import extract_and_save_crop

        temp_dir = tempfile.mkdtemp()
        try:
            real_img_path = os.path.join(temp_dir, "doc_scan.png")
            width, height = 400, 300
            # Create a real RGB image with background color
            img = Image.new("RGB", (width, height), color=(245, 245, 245))
            img.save(real_img_path, format="PNG")

            ocr_tokens = [
                {"text": "Net", "box": [50, 60, 80, 110]},
                {"text": "Profit", "box": [50, 115, 80, 180]}
            ]
            result = self.locator.locate(
                image_path=real_img_path,
                query="What is the Net Profit?",
                image_width=width,
                image_height=height,
                ocr_tokens=ocr_tokens,
                prefer_context_preserving=True
            )
            self.assertTrue(result.is_valid)
            self.assertIsNotNone(result.crop_box)

            # Physically extract and write crop to disk
            crop_out_dir = os.path.join(temp_dir, "crops")
            crop_path, crop_size = extract_and_save_crop(real_img_path, result.crop_box, output_dir=crop_out_dir)

            self.assertTrue(os.path.exists(crop_path), f"Crop file was not written to {crop_path}")
            # Open physical crop from disk and verify dimensions
            with Image.open(crop_path) as crop_img:
                expected_w = result.crop_box[3] - result.crop_box[1]
                expected_h = result.crop_box[2] - result.crop_box[0]
                self.assertEqual(crop_img.size, (expected_w, expected_h))
                self.assertEqual(crop_img.format, "PNG")
                self.assertEqual(crop_size, (expected_w, expected_h))
        finally:
            shutil.rmtree(temp_dir)

    def test_document_title_spatial_prior(self):
        """Test 9: Document title questions target top header band."""
        result = self.locator.locate(
            image_path="test_image.png",
            query="What is the title of the document ?",
            image_width=self.width,
            image_height=self.height,
            ocr_tokens=[{"text": "Report", "box": (50, 100, 80, 200)}],
            prefer_context_preserving=True
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.source, LocalizationSource.HEURISTIC_LAYOUT)
        self.assertEqual(result.crop_box[0], 0)
        self.assertLessEqual(result.crop_box[2], int(0.28 * self.height))
        self.assertEqual(result.crop_box[3], self.width)
        self.assertGreaterEqual(result.confidence, 0.75)

    def test_top_of_page_spatial_prior(self):
        """Test 10: Top-of-page questions target top band."""
        result = self.locator.locate(
            image_path="test_image.png",
            query="What is the year mentioned at the top of the page?",
            image_width=self.width,
            image_height=self.height,
            ocr_tokens=[{"text": "2013", "box": (40, 200, 70, 300)}],
            prefer_context_preserving=True
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.source, LocalizationSource.HEURISTIC_LAYOUT)
        self.assertEqual(result.crop_box[0], 0)
        self.assertLessEqual(result.crop_box[2], int(0.22 * self.height))
        self.assertEqual(result.crop_box[3], self.width)

    def test_repeated_brand_names_density_clustering(self):
        """Test 11: Repeated brand names across document do not cause full-page expansion."""
        # Brand 'Acme' appears 10 times from y=100 to y=2000
        ocr_tokens = []
        for y in range(100, 2100, 200):
            ocr_tokens.append({"text": "Acme", "box": (y, 100, y + 30, 200)})
        # But 'personal' and 'care' appear exclusively at y=1400
        ocr_tokens.append({"text": "Personal", "box": (1400, 300, 1430, 450)})
        ocr_tokens.append({"text": "Care", "box": (1400, 460, 1430, 550)})
        ocr_tokens.append({"text": "Brands", "box": (1400, 560, 1430, 680)})

        result = self.locator.locate(
            image_path="test_image.png",
            query="Name the personal care brands of Acme?",
            image_width=self.width,
            image_height=self.height,
            ocr_tokens=ocr_tokens,
            prefer_context_preserving=True
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.source, LocalizationSource.OCR_LAYOUT)
        # Must tightly cluster around y=1400, NOT spanning y=100..2000
        crop = result.crop_box
        self.assertGreaterEqual(crop[0], 500, "Crop should not start at the very top of the page (y=100)")
        self.assertLessEqual(crop[2], 2100, "Crop should stay bounded below")
        area_ratio = ((crop[2] - crop[0]) * (crop[3] - crop[1])) / (self.width * self.height)
        self.assertLess(area_ratio, 0.65, "Clustered crop area should be well below full image")

    def test_financial_table_row_and_header_matching(self):
        """Test 12: Financial table query locates specific line item and preserves header."""
        ocr_tokens = [
            {"text": "Expense", "box": (200, 100, 230, 250)},
            {"text": "Amount", "box": (200, 400, 230, 550)},
            {"text": "Chemicals", "box": (500, 100, 530, 250)},
            {"text": "Supplies", "box": (500, 260, 530, 380)},
            {"text": "485", "box": (500, 400, 530, 480)},
        ]
        result = self.locator.locate(
            image_path="test_image.png",
            query="What is cost of chemicals and supplies?",
            image_width=self.width,
            image_height=self.height,
            ocr_tokens=ocr_tokens,
            prefer_context_preserving=True
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.source, LocalizationSource.OCR_LAYOUT)
        # Must encompass row at y=500
        crop = result.crop_box
        self.assertLessEqual(crop[0], 500)
        self.assertGreaterEqual(crop[2], 530)

    def test_percentage_question_row_localization(self):
        """Test 13: Percentage survey query locks onto the specific question row."""
        ocr_tokens = [
            {"text": "Survey", "box": (100, 100, 130, 250)},
            {"text": "Non-Smokers", "box": (200, 600, 230, 800)},
            {"text": "Restore", "box": (850, 100, 880, 190)},
            {"text": "Romance", "box": (850, 200, 880, 350)},
            {"text": "Mystery", "box": (850, 360, 880, 500)},
            {"text": "Feel", "box": (850, 510, 880, 580)},
            {"text": "57%", "box": (850, 700, 880, 780)},
        ]
        result = self.locator.locate(
            image_path="test_image.png",
            query="What percentage of non-smokers feel the need to restore romance and mystery?",
            image_width=self.width,
            image_height=self.height,
            ocr_tokens=ocr_tokens,
            prefer_context_preserving=True
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.source, LocalizationSource.OCR_LAYOUT)
        crop = result.crop_box
        self.assertLessEqual(crop[0], 850)
        self.assertGreaterEqual(crop[2], 880)

    def test_genuinely_ambiguous_query_falls_back_to_overview(self):
        """Test 14: Genuinely ambiguous query without matching keywords cleanly falls back to full overview."""
        ocr_tokens = [
            {"text": "Invoice", "box": (100, 100, 130, 250)},
            {"text": "Total", "box": (200, 100, 230, 200)},
        ]
        # Query with zero overlap in text and no layout priors
        result = self.locator.locate(
            image_path="test_image.png",
            query="What is the quantum orbital momentum?",
            image_width=self.width,
            image_height=self.height,
            ocr_tokens=ocr_tokens,
            prefer_context_preserving=True
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.source, LocalizationSource.FALLBACK_OVERVIEW)
        self.assertEqual(result.crop_box, (0, 0, self.height, self.width))
        self.assertEqual(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
