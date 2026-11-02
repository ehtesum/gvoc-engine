"""
Unit and integration tests for CCEA physical crop extraction and intervention.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from gvoc_engine.interventions.context_preservation import (
    ContextKind,
    EvidenceRegion,
)
from gvoc_engine.interventions.ccea_intervention import (
    CCEAIntervention,
    CCEAResult,
    extract_and_save_crop,
    get_image_dimensions,
)


class DummyVLMClient:
    """Mock client returning deterministic responses for testing."""
    def __init__(self, answer="Verified Value: 42.5 MW", locator_box="[40, 50, 90, 110]"):
        self.answer = answer
        self.locator_box = locator_box
        self.calls = []

    def call_vlm(self, prompt: str, image_path: str, max_tokens: int = 100):
        self.calls.append({"prompt": prompt, "image_path": image_path})
        if "Localization" in prompt:
            return self.locator_box, 45, 12, 0.05
        return self.answer, 80, 25, 0.12


class TestCCEAIntervention(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.test_dir, "test_doc.png")

        # Create a synthetic 300x200 RGB test image
        img = Image.new("RGB", (300, 200), color=(240, 240, 240))
        img.save(self.image_path)

        self.evidence = EvidenceRegion(
            target_box=(40, 50, 90, 110),
            required_context=(
                (ContextKind.COLUMN_LABEL, (10, 45, 35, 115)),
                (ContextKind.ROW_LABEL, (40, 10, 90, 45)),
                (ContextKind.UNIT, (5, 10, 25, 45)),
            ),
            evidence_id="sample_doc_region"
        )
        self.intervention = CCEAIntervention(output_dir=os.path.join(self.test_dir, "crops"))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_get_image_dimensions(self):
        w, h = get_image_dimensions(self.image_path)
        self.assertEqual((w, h), (300, 200))

    def test_extract_and_save_crop(self):
        crop_path, crop_size = extract_and_save_crop(
            self.image_path,
            (40, 50, 90, 110),
            output_dir=os.path.join(self.test_dir, "crops")
        )
        self.assertTrue(os.path.exists(crop_path))
        self.assertEqual(crop_size, (60, 50))  # width = 110 - 50 = 60, height = 90 - 40 = 50

    def test_full_ccea_intervention_preserves_context(self):
        client = DummyVLMClient()
        res = self.intervention.execute(
            image_path=self.image_path,
            query="What is the rated capacity?",
            client=client,
            evidence_region=self.evidence,
            initial_answer="Uncertain initial guess",
            preserve_context=True,
        )

        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.final_answer, "Verified Value: 42.5 MW")
        self.assertTrue(res.preserved_context_flag)
        self.assertIsNotNone(res.assessment)
        self.assertTrue(res.assessment.valid)
        self.assertEqual(res.missing_context, ())
        self.assertEqual(len(res.retained_context), 3)
        self.assertGreater(res.total_tokens, 0)
        self.assertGreater(res.cost_usd, 0.0)
        self.assertTrue(os.path.exists(res.crop_path))

    def test_ablation_mode_without_context_preservation_records_context_loss(self):
        client = DummyVLMClient()
        res = self.intervention.execute(
            image_path=self.image_path,
            query="What is the value?",
            client=client,
            evidence_region=self.evidence,
            initial_answer="Initial guess",
            preserve_context=False,  # ABLATION CONDITION: target-only crop
        )

        self.assertEqual(res.status, "SUCCESS")
        self.assertFalse(res.preserved_context_flag)
        self.assertIsNotNone(res.assessment)
        # Should record that required context regions were missed by the tight crop
        self.assertFalse(res.assessment.valid)
        self.assertIn(ContextKind.COLUMN_LABEL, res.missing_context)
        self.assertIn(ContextKind.ROW_LABEL, res.missing_context)

    def test_malformed_locator_fails_closed(self):
        client = DummyVLMClient(locator_box="INVALID_NO_COORDINATES")
        evidence_no_target = EvidenceRegion(target_box=None)  # forces locator call
        res = self.intervention.execute(
            image_path=self.image_path,
            query="Find metric",
            client=client,
            evidence_region=evidence_no_target,
            initial_answer="Initial",
            preserve_context=True,
        )
        self.assertEqual(res.status, "REJECTED_INVALID_BOX")
        self.assertIn("Locator output unparseable", res.failure_reason)
        # Final answer reverts to initial answer
        self.assertEqual(res.final_answer, "Initial")

    def test_missing_image_file_fails_gracefully(self):
        client = DummyVLMClient()
        res = self.intervention.execute(
            image_path="non_existent_image.png",
            query="Find metric",
            client=client,
            evidence_region=self.evidence,
            initial_answer="Initial",
        )
        self.assertEqual(res.status, "ERROR")
        self.assertIn("Image file not found", res.failure_reason)


if __name__ == "__main__":
    unittest.main()
