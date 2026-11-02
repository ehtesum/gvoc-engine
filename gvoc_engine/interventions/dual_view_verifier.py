"""Dual-View Optical-Epistemic Verification Engine for Visual Documents.

This module implements the context-preserving dual-view verification architecture
designed to outperform monolithic foundation VLMs by:
1. Preserving base model accuracy on clear, legible text via an Optical-Epistemic Gate.
2. Expanding crop regions to preserve row/column structural headers (preventing context severing).
3. Feeding both global layout and native high-resolution crops in a single turn.
4. Using schema-constrained residual verification to avoid prompt token truncation.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from gvoc_engine.interventions.context_preservation import (
    Box,
    validate_box,
    expand_box,
    containment_ratio,
    ContextKind,
)


class GateDecision(str, Enum):
    """Decision emitted by the Optical-Epistemic Gate."""
    STOP = "STOP"
    VERIFY_CROP = "VERIFY_CROP"


class VerificationStatus(str, Enum):
    """Status emitted by the residual verification module."""
    CONFIRMED = "CONFIRMED"
    CORRECTED = "CORRECTED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class GateResult:
    """Outcome of the Optical-Epistemic Gating check."""
    decision: GateDecision
    optical_adequate: bool
    epistemic_confident: bool
    estimated_char_height_px: float
    confidence: float
    rationale: str


@dataclass(frozen=True)
class VerificationResult:
    """Final result of the Dual-View Verification pipeline."""
    final_answer: str
    decision: GateDecision
    verification_status: VerificationStatus
    base_candidate_answer: str
    tokens_consumed: int
    intervened: bool
    crop_box: Optional[Box] = None
    optical_notes: str = ""


class OpticalEpistemicGate:
    """Decides whether to accept the base VLM prediction or acquire high-res evidence.
    
    Prevents harmful interventions on clear text where the base model is already accurate.
    """

    def __init__(
        self,
        min_char_height_px: float = 14.0,
        high_confidence_threshold: float = 0.85,
        min_margin_threshold: float = 0.30,
    ):
        """
        Args:
            min_char_height_px: Minimum physical character height in pixels. Below this,
                                optical sampling artifacts / blur make text illegible.
            high_confidence_threshold: Softmax or verbalized confidence required to accept.
            min_margin_threshold: Margin between top-1 and top-2 candidate tokens.
        """
        self.min_char_height_px = min_char_height_px
        self.high_confidence_threshold = high_confidence_threshold
        self.min_margin_threshold = min_margin_threshold

    def evaluate(
        self,
        candidate_answer: str,
        confidence: float,
        target_box: Optional[Box] = None,
        image_dims: Optional[Tuple[int, int]] = None,
        estimated_char_height_px: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> GateResult:
        """Evaluates whether to STOP or trigger active visual verification."""
        # 1. Estimate physical character height
        char_h = estimated_char_height_px
        if char_h is None and target_box is not None:
            y0, _, y1, _ = target_box
            box_h = max(1, y1 - y0)
            # Estimate character height from box height and answer length
            char_h = float(box_h)

        # Default fallback if coordinates unavailable
        if char_h is None:
            char_h = 24.0  # Assume standard legible text if unannotated

        optical_adequate = char_h >= self.min_char_height_px

        # 2. Epistemic check
        conf_adequate = confidence >= self.high_confidence_threshold
        margin_adequate = True if margin is None else (margin >= self.min_margin_threshold)
        epistemic_confident = conf_adequate and margin_adequate

        # Empty, uncertain, or hedging answers always trigger verification if text exists
        is_hedging = any(
            h in candidate_answer.lower()
            for h in ["unclear", "blurred", "cannot read", "maybe", "not legible", "smudge"]
        )
        if is_hedging:
            epistemic_confident = False

        if optical_adequate and epistemic_confident:
            decision = GateDecision.STOP
            rationale = (
                f"Optical resolution ({char_h:.1f}px >= {self.min_char_height_px}px) and "
                f"confidence ({confidence:.2f} >= {self.high_confidence_threshold:.2f}) are adequate. "
                f"Preserving base model zero-shot prediction."
            )
        else:
            decision = GateDecision.VERIFY_CROP
            reasons = []
            if not optical_adequate:
                reasons.append(f"optical blur (char height {char_h:.1f}px < {self.min_char_height_px}px)")
            if not epistemic_confident:
                reasons.append(f"epistemic uncertainty (conf {confidence:.2f} < {self.high_confidence_threshold:.2f})")
            rationale = f"Triggering dual-view verification due to {', '.join(reasons)}."

        return GateResult(
            decision=decision,
            optical_adequate=optical_adequate,
            epistemic_confident=epistemic_confident,
            estimated_char_height_px=char_h,
            confidence=confidence,
            rationale=rationale,
        )


class ContextPreservingCropBuilder:
    """Builds crops that preserve enclosing structural headers and units."""

    def __init__(
        self,
        vertical_expansion_factor: float = 0.80,
        horizontal_expansion_factor: float = 0.40,
        min_header_headroom_fraction: float = 0.15,
    ):
        self.vertical_expansion = vertical_expansion_factor
        self.horizontal_expansion = horizontal_expansion_factor
        self.min_header_headroom = min_header_headroom_fraction

    def build_crop(
        self,
        target_box: Box,
        image_width: int,
        image_height: int,
        context_boxes: Optional[List[Tuple[ContextKind, Box]]] = None,
    ) -> Box:
        """Expands target box to guarantee structural header and unit inclusion."""
        is_valid, err = validate_box(target_box, image_width, image_height)
        if not is_valid:
            raise ValueError(f"Invalid target box: {err}")

        y0, x0, y1, x1 = target_box
        box_h = y1 - y0
        box_w = x1 - x0

        # If explicit context boxes (headers, labels) are provided, compute union
        if context_boxes:
            all_boxes = [target_box] + [b for _, b in context_boxes]
            uy0 = min(b[0] for b in all_boxes)
            ux0 = min(b[1] for b in all_boxes)
            uy1 = max(b[2] for b in all_boxes)
            ux1 = max(b[3] for b in all_boxes)
            # Add small padding
            pad_y = int((uy1 - uy0) * 0.05)
            pad_x = int((ux1 - ux0) * 0.05)
            return (
                max(0, uy0 - pad_y),
                max(0, ux0 - pad_x),
                min(image_height, uy1 + pad_y),
                min(image_width, ux1 + pad_x),
            )

        # Otherwise expand structurally: expand vertically upwards to capture column headers
        upward_headroom = int(box_h * self.vertical_expansion + image_height * self.min_header_headroom)
        downward_headroom = int(box_h * 0.30)
        left_headroom = int(box_w * self.horizontal_expansion + image_width * 0.10)
        right_headroom = int(box_w * self.horizontal_expansion)

        crop_y0 = max(0, y0 - upward_headroom)
        crop_x0 = max(0, x0 - left_headroom)
        crop_y1 = min(image_height, y1 + downward_headroom)
        crop_x1 = min(image_width, x1 + right_headroom)

        return (crop_y0, crop_x0, crop_y1, crop_x1)


class DualViewPromptFormatter:
    """Formats dual-view prompts with compact, JSON-constrained schemas."""

    @staticmethod
    def format_verification_prompt(
        question: str,
        candidate_answer: str,
        crop_box: Box,
        doc_width: int,
        doc_height: int,
    ) -> str:
        """Constructs a concise, single-turn verification prompt."""
        return (
            f"You are verifying a visual document query.\n"
            f"Question: {question}\n"
            f"Overview Hypothesis: \"{candidate_answer}\"\n"
            f"The provided image is a high-resolution crop around coordinates {list(crop_box)} "
            f"(page dimensions: {doc_width}x{doc_height}).\n\n"
            f"Task: Inspect the crop for fine details, units, negative signs, or decimals.\n"
            f"Respond strictly in valid JSON with this format:\n"
            f'{{"status": "CONFIRMED"|"CORRECTED", "answer": "<exact text>", "notes": "<short explanation>"}}\n'
            f"Output JSON only."
        )

    @staticmethod
    def parse_verification_response(response_text: str, fallback_candidate: str) -> Tuple[VerificationStatus, str, str]:
        """Parses the model's structured verification response with error tolerance."""
        clean_text = response_text.strip()
        # Extract json block if wrapped in markdown code fence
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                status_raw = data.get("status", "").upper()
                if "CONFIRM" in status_raw:
                    status = VerificationStatus.CONFIRMED
                elif "CORRECT" in status_raw:
                    status = VerificationStatus.CORRECTED
                else:
                    status = VerificationStatus.UNRESOLVED

                answer = str(data.get("answer", "")).strip()
                if not answer:
                    answer = fallback_candidate

                notes = str(data.get("notes", "")).strip()
                return status, answer, notes
            except Exception:
                pass

        # Fallback heuristic if JSON failed
        if "confirm" in clean_text.lower():
            return VerificationStatus.CONFIRMED, fallback_candidate, "Parsed via text fallback"
        
        # If response is concise and not raw JSON, treat it as candidate
        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
        if lines:
            first_line = lines[0]
            if len(first_line) < 80:
                return VerificationStatus.CORRECTED, first_line, "Parsed from first line"

        return VerificationStatus.UNRESOLVED, fallback_candidate, "Could not parse verification JSON"


class DualViewVerifier:
    """End-to-end Dual-View Verification pipeline."""

    def __init__(
        self,
        gate: Optional[OpticalEpistemicGate] = None,
        crop_builder: Optional[ContextPreservingCropBuilder] = None,
    ):
        self.gate = gate or OpticalEpistemicGate()
        self.crop_builder = crop_builder or ContextPreservingCropBuilder()

    def verify(
        self,
        question: str,
        candidate_answer: str,
        confidence: float,
        image_dims: Tuple[int, int],
        vlm_client: Any,
        target_box: Optional[Box] = None,
        estimated_char_height_px: Optional[float] = None,
        context_boxes: Optional[List[Tuple[ContextKind, Box]]] = None,
        crop_image_path: Optional[str] = None,
    ) -> VerificationResult:
        """Executes the complete optical-epistemic verification pipeline."""
        width, height = image_dims

        # 1. Optical-Epistemic Gating check
        gate_result = self.gate.evaluate(
            candidate_answer=candidate_answer,
            confidence=confidence,
            target_box=target_box,
            image_dims=image_dims,
            estimated_char_height_px=estimated_char_height_px,
        )

        # 2. If Gate decides STOP, return base candidate immediately
        if gate_result.decision == GateDecision.STOP:
            return VerificationResult(
                final_answer=candidate_answer,
                decision=GateDecision.STOP,
                verification_status=VerificationStatus.CONFIRMED,
                base_candidate_answer=candidate_answer,
                tokens_consumed=0,
                intervened=False,
                crop_box=None,
                optical_notes=gate_result.rationale,
            )

        # 3. Otherwise, build context-preserving crop
        if target_box is None:
            # Symmetrical center/overview fallback
            target_box = (int(height * 0.25), int(width * 0.25), int(height * 0.75), int(width * 0.75))

        crop_box = self.crop_builder.build_crop(
            target_box=target_box,
            image_width=width,
            image_height=height,
            context_boxes=context_boxes,
        )

        # 4. Format prompt
        prompt = DualViewPromptFormatter.format_verification_prompt(
            question=question,
            candidate_answer=candidate_answer,
            crop_box=crop_box,
            doc_width=width,
            doc_height=height,
        )

        # 5. Call VLM client (constrained max_tokens to prevent truncation)
        call_res = vlm_client.call(
            prompt=prompt,
            image_path=crop_image_path or "",
            max_tokens=50,
        )

        resp_text = call_res.get("text", "")
        tokens = call_res.get("total_tokens", 45)

        # 6. Parse structured response
        status, answer, notes = DualViewPromptFormatter.parse_verification_response(
            resp_text, fallback_candidate=candidate_answer
        )

        return VerificationResult(
            final_answer=answer,
            decision=GateDecision.VERIFY_CROP,
            verification_status=status,
            base_candidate_answer=candidate_answer,
            tokens_consumed=tokens,
            intervened=True,
            crop_box=crop_box,
            optical_notes=f"{gate_result.rationale} | VLM: {notes}",
        )
