"""Unit tests for the Dual-View Optical-Epistemic Verifier."""

import pytest
from typing import Dict, Any, Optional

from gvoc_engine.interventions.dual_view_verifier import (
    OpticalEpistemicGate,
    GateDecision,
    ContextPreservingCropBuilder,
    DualViewPromptFormatter,
    DualViewVerifier,
    VerificationStatus,
)
from gvoc_engine.interventions.context_preservation import ContextKind, validate_box


class MockVLMVerificationClient:
    """Mock client for offline testing."""
    def __init__(self, response_text: str = '{"status": "CONFIRMED", "answer": "$14.2M", "notes": "Clear view"}'):
        self.response_text = response_text
        self.call_count = 0

    def call(self, prompt: str, image_path: str = "", max_tokens: int = 50) -> Dict[str, Any]:
        self.call_count += 1
        return {
            "status": "success",
            "text": self.response_text,
            "prompt_tokens": 85,
            "completion_tokens": 18,
            "total_tokens": 103,
        }


def test_optical_epistemic_gate_clear_text_stops():
    """Clear text with high confidence should STOP immediately."""
    gate = OpticalEpistemicGate(min_char_height_px=14.0, high_confidence_threshold=0.85)
    res = gate.evaluate(
        candidate_answer="$14.2M",
        confidence=0.95,
        estimated_char_height_px=28.0,
    )
    assert res.decision == GateDecision.STOP
    assert res.optical_adequate is True
    assert res.epistemic_confident is True


def test_optical_epistemic_gate_low_confidence_verifies():
    """Clear resolution but uncertain confidence must trigger verification."""
    gate = OpticalEpistemicGate(min_char_height_px=14.0, high_confidence_threshold=0.85)
    res = gate.evaluate(
        candidate_answer="$14.2M",
        confidence=0.60,
        estimated_char_height_px=25.0,
    )
    assert res.decision == GateDecision.VERIFY_CROP
    assert res.optical_adequate is True
    assert res.epistemic_confident is False


def test_optical_epistemic_gate_optical_blur_verifies():
    """High model confidence but physically tiny character height (<14px) must trigger verification."""
    gate = OpticalEpistemicGate(min_char_height_px=14.0, high_confidence_threshold=0.85)
    res = gate.evaluate(
        candidate_answer="0.5 mg",
        confidence=0.92,
        estimated_char_height_px=8.0,
    )
    assert res.decision == GateDecision.VERIFY_CROP
    assert res.optical_adequate is False


def test_optical_epistemic_gate_hedging_verifies():
    """Hedging phrases in candidate answer must force verification."""
    gate = OpticalEpistemicGate(min_char_height_px=14.0, high_confidence_threshold=0.85)
    res = gate.evaluate(
        candidate_answer="Unclear due to blur, possibly 50mg",
        confidence=0.90,
        estimated_char_height_px=30.0,
    )
    assert res.decision == GateDecision.VERIFY_CROP
    assert res.epistemic_confident is False


def test_crop_builder_header_expansion():
    """Crop builder must expand upwards to encompass column headers."""
    builder = ContextPreservingCropBuilder()
    img_w, img_h = 1000, 1000
    target_box = (600, 400, 650, 500)  # y0=600, x0=400, y1=650, x1=500
    
    crop = builder.build_crop(target_box, img_w, img_h)
    
    # Must be valid within image boundaries
    valid, err = validate_box(crop, img_w, img_h)
    assert valid, err
    
    # Upward boundary (crop[0]) must be substantially above target y0 (600) to capture table header
    assert crop[0] < 500
    # Left boundary (crop[1]) must be left of target x0 (400) to capture row label
    assert crop[1] < 400
    # Target must be contained within crop
    assert crop[0] <= target_box[0] and crop[1] <= target_box[1]
    assert crop[2] >= target_box[2] and crop[3] >= target_box[3]


def test_crop_builder_explicit_context_union():
    """Crop builder must compute the union with explicit context boxes."""
    builder = ContextPreservingCropBuilder()
    img_w, img_h = 1000, 1000
    target_cell = (500, 400, 530, 480)
    col_header = (200, 400, 230, 480)
    row_label = (500, 100, 530, 250)
    
    context_boxes = [
        (ContextKind.COLUMN_LABEL, col_header),
        (ContextKind.ROW_LABEL, row_label),
    ]
    
    crop = builder.build_crop(target_cell, img_w, img_h, context_boxes=context_boxes)
    
    assert crop[0] <= 200  # Captures column header at top
    assert crop[1] <= 100  # Captures row label at left
    assert crop[2] >= 530  # Captures target bottom
    assert crop[3] >= 480  # Captures target right


def test_dual_view_prompt_formatter_and_parser():
    """Formatter must produce concise text and parse JSON responses cleanly."""
    crop_box = (100, 100, 300, 300)
    prompt = DualViewPromptFormatter.format_verification_prompt(
        question="What is the dosage?",
        candidate_answer="50 mg",
        crop_box=crop_box,
        doc_width=1000,
        doc_height=1000,
    )
    assert "What is the dosage?" in prompt
    assert '"50 mg"' in prompt
    assert "JSON" in prompt

    # Test JSON with markdown fence
    fenced_resp = "```json\n{\"status\": \"CORRECTED\", \"answer\": \"0.5 mg\", \"notes\": \"Decimal point clear\"}\n```"
    status, ans, notes = DualViewPromptFormatter.parse_verification_response(fenced_resp, "50 mg")
    assert status == VerificationStatus.CORRECTED
    assert ans == "0.5 mg"
    assert "Decimal" in notes


def test_dual_view_verifier_pipeline_stop():
    """When gate stops, base answer is preserved with 0 tokens consumed."""
    client = MockVLMVerificationClient()
    verifier = DualViewVerifier()
    
    res = verifier.verify(
        question="What is the net profit?",
        candidate_answer="$10.5M",
        confidence=0.98,
        image_dims=(1000, 1000),
        vlm_client=client,
        estimated_char_height_px=30.0,
    )
    
    assert res.decision == GateDecision.STOP
    assert res.final_answer == "$10.5M"
    assert res.tokens_consumed == 0
    assert res.intervened is False
    assert client.call_count == 0  # No API call made


def test_dual_view_verifier_pipeline_intervention():
    """When gate triggers, VLM verification updates answer and consumes tokens."""
    client = MockVLMVerificationClient(
        response_text='{"status": "CORRECTED", "answer": "-$14.2M", "notes": "Accounting parentheses observed"}'
    )
    verifier = DualViewVerifier()
    
    res = verifier.verify(
        question="Operating income?",
        candidate_answer="+$14.2M",
        confidence=0.55,  # Low confidence
        image_dims=(1000, 1000),
        vlm_client=client,
        estimated_char_height_px=10.0,  # Tiny text
    )
    
    assert res.decision == GateDecision.VERIFY_CROP
    assert res.final_answer == "-$14.2M"
    assert res.verification_status == VerificationStatus.CORRECTED
    assert res.tokens_consumed > 0
    assert res.intervened is True
    assert client.call_count == 1
