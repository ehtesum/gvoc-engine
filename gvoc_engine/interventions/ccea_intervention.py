"""
Context-Preserving, Cost-Aware Visual Evidence Acquisition (CCEA) Intervention.

Performs physical visual evidence localization, context-preserving crop extraction,
validation against required headers/labels/units, and focused VLM question answering.
"""

import os
import time
import uuid
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Sequence, Union

from PIL import Image

from gvoc_engine.interventions.context_preservation import (
    Box,
    ContextKind,
    EvidenceRegion,
    CropAssessment,
    validate_box,
    assess_crop,
    propose_context_preserving_crop,
    expand_box,
    parse_bounding_box,
    format_box,
)
from gvoc_engine.core.budget_guard import BudgetExceededError


@dataclass
class CCEAResult:
    """Telemetry and outcome record of a CCEA visual intervention."""
    initial_answer: str = ""
    final_answer: str = ""
    target_box: Optional[Box] = None
    crop_box: Optional[Box] = None
    source_dimensions: Tuple[int, int] = (0, 0)      # (width, height)
    crop_dimensions: Tuple[int, int] = (0, 0)        # (width, height)
    assessment: Optional[CropAssessment] = None
    retained_context: Tuple[ContextKind, ...] = ()
    missing_context: Tuple[ContextKind, ...] = ()
    preserved_context_flag: bool = False
    intervention_prompt: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency: float = 0.0
    cost_usd: float = 0.0
    num_calls: int = 0
    status: str = "INITIALIZED"                      # SUCCESS, REJECTED_INVALID_BOX, REJECTED_CONTEXT_LOSS, FALLBACK, ERROR
    failure_reason: Optional[str] = None
    crop_path: Optional[str] = None


def get_image_dimensions(image_path: Union[str, Path]) -> Tuple[int, int]:
    """Reads image dimensions (width, height) using PIL without loading full pixel array."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")
    with Image.open(path) as img:
        return img.size  # (width, height)


def extract_and_save_crop(
    image_path: Union[str, Path],
    box: Box,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[str, Tuple[int, int]]:
    """Physically extracts a visual crop [ymin, xmin, ymax, xmax] and writes it to disk.
    PIL coordinates require (left, upper, right, lower) = (xmin, ymin, xmax, ymax).
    Returns: (saved_crop_path, (crop_width, crop_height))
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        width, height = img.size
        is_valid, err = validate_box(box, width, height)
        if not is_valid:
            raise ValueError(f"Cannot crop invalid box {box} on image ({width}x{height}): {err}")

        ymin, xmin, ymax, xmax = box
        # PIL uses (left, upper, right, lower) = (xmin, ymin, xmax, ymax)
        crop_img = img.crop((xmin, ymin, xmax, ymax))
        crop_size = crop_img.size  # (crop_width, crop_height)

        if output_dir:
            out_dir = Path(output_dir)
        else:
            out_dir = Path(tempfile.gettempdir()) / "ccea_crops"
        out_dir.mkdir(parents=True, exist_ok=True)

        crop_filename = f"crop_{uuid.uuid4().hex[:8]}_{ymin}_{xmin}_{ymax}_{xmax}.png"
        crop_path = str(out_dir / crop_filename)
        crop_img.save(crop_path, format="PNG")
        return crop_path, crop_size


class CCEAIntervention:
    """Executes Context-Preserving Visual Evidence Acquisition."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else None

    def execute(
        self,
        image_path: str,
        query: str,
        client: Any,
        evidence_region: Optional[EvidenceRegion] = None,
        initial_answer: str = "",
        preserve_context: bool = True,
        max_tokens: int = 120,
        budget_remaining: int = 2000,
    ) -> CCEAResult:
        """Runs the full visual evidence acquisition pipeline from image to answer.
        
        Steps:
        1. Obtain image dimensions.
        2. Localize or retrieve target bounding box.
        3. Assess or enforce context preservation (table headers, row labels, units, etc.).
        4. Physically crop the source image.
        5. Invoke the vision model on the acquired crop.
        6. Return complete CCEAResult telemetry.
        """
        t0 = time.time()
        result = CCEAResult(
            initial_answer=initial_answer,
            final_answer=initial_answer,
            preserved_context_flag=preserve_context
        )

        # Pre-check image file
        img_path = Path(image_path)
        if not img_path.exists():
            # Mock or unit-test fallback when physical image is absent
            result.status = "ERROR"
            result.failure_reason = f"Image file not found: {image_path}"
            result.latency = time.time() - t0
            return result

        try:
            width, height = get_image_dimensions(img_path)
            result.source_dimensions = (width, height)
        except Exception as e:
            result.status = "ERROR"
            result.failure_reason = f"Failed to read image dimensions: {e}"
            result.latency = time.time() - t0
            return result

        # -------------------------------------------------------------
        # Step 1: Target Evidence Localization
        # -------------------------------------------------------------
        target_box: Optional[Box] = None
        if evidence_region and evidence_region.target_box:
            target_box = evidence_region.target_box
            is_valid, err = validate_box(target_box, width, height)
            if not is_valid:
                result.status = "REJECTED_INVALID_BOX"
                result.failure_reason = f"Supplied target box invalid: {err}"
                result.latency = time.time() - t0
                return result
        else:
            # Model-based localization query
            loc_prompt = (
                f"Visual Evidence Localization Task:\n"
                f"Question: {query}\n"
                f"Identify the precise visual region containing the evidence needed to answer the question.\n"
                f"Image dimensions are width={width}, height={height}.\n"
                f"Output ONLY the bounding box in format [ymin, xmin, ymax, xmax] with no other text."
            )
            try:
                if hasattr(client, "call_vlm"):
                    resp, p_tok, c_tok, lat = client.call_vlm(loc_prompt, str(img_path), max_tokens=60)
                elif hasattr(client, "call"):
                    res_dict = client.call(loc_prompt, image_path=str(img_path), max_tokens=60)
                    resp = res_dict.get("text", "")
                    p_tok = res_dict.get("prompt_tokens", 0)
                    c_tok = res_dict.get("completion_tokens", 0)
                    lat = res_dict.get("latency_sec", 0.0)
                else:
                    resp = "[0, 0, 10, 10]"
                    p_tok, c_tok, lat = 50, 10, 0.1

                result.num_calls += 1
                result.prompt_tokens += p_tok
                result.completion_tokens += c_tok
                result.total_tokens += (p_tok + c_tok)

                parsed_box, parse_msg = parse_bounding_box(resp, width, height)
                if parsed_box is None:
                    result.status = "REJECTED_INVALID_BOX"
                    result.failure_reason = f"Locator output unparseable: {parse_msg}"
                    result.latency = time.time() - t0
                    return result
                target_box = parsed_box
            except Exception as e:
                result.status = "ERROR"
                result.failure_reason = f"Locator call failed: {e}"
                result.latency = time.time() - t0
                return result

        result.target_box = target_box

        # -------------------------------------------------------------
        # Step 2: Context Preservation & Crop Selection
        # -------------------------------------------------------------
        crop_box: Box = target_box
        has_context = bool(evidence_region and evidence_region.required_context)

        if preserve_context and has_context and evidence_region is not None:
            # Propose crop covering target plus all declared context regions
            try:
                crop_box = propose_context_preserving_crop(
                    target_box=target_box,
                    required_context_boxes=[b for _, b in evidence_region.required_context],
                    width=width,
                    height=height,
                    margin_fraction=0.08,
                )
            except Exception as e:
                result.status = "REJECTED_INVALID_BOX"
                result.failure_reason = f"Failed to construct context-preserving crop: {e}"
                result.latency = time.time() - t0
                return result

            # Validate crop assessment
            assessment = assess_crop(crop_box, evidence_region, width, height)
            result.assessment = assessment
            result.retained_context = assessment.retained_context
            result.missing_context = assessment.missing_context

            if not assessment.valid:
                result.status = "REJECTED_CONTEXT_LOSS"
                result.failure_reason = f"Crop rejected: missing context {assessment.missing_context}"
                result.latency = time.time() - t0
                return result
        else:
            # Ablation / Target-only condition: crop only the target (optionally with minimal margin)
            crop_box = expand_box(target_box, width, height, margin_fraction=0.05)
            if has_context and evidence_region is not None:
                # Assess what context is lost for ablation reporting
                assessment = assess_crop(crop_box, evidence_region, width, height)
                result.assessment = assessment
                result.retained_context = assessment.retained_context
                result.missing_context = assessment.missing_context

        result.crop_box = crop_box

        # -------------------------------------------------------------
        # Step 3: Physical Image Crop Extraction
        # -------------------------------------------------------------
        try:
            crop_path, crop_dims = extract_and_save_crop(img_path, crop_box, self.output_dir)
            result.crop_path = crop_path
            result.crop_dimensions = crop_dims
        except Exception as e:
            result.status = "ERROR"
            result.failure_reason = f"Image crop extraction failed: {e}"
            result.latency = time.time() - t0
            return result

        # -------------------------------------------------------------
        # Step 4: Vision Model Call with Acquired Visual Crop
        # -------------------------------------------------------------
        ans_prompt = (
            f"Focused Evidence Visual Question Answering:\n"
            f"Question: {query}\n"
            f"Carefully inspect the provided visual evidence crop (including any table headers, units, row/column labels, and values).\n"
            f"Provide a concise, direct answer to the question."
        )
        result.intervention_prompt = ans_prompt

        try:
            if hasattr(client, "call_vlm"):
                ans_text, p_tok, c_tok, lat = client.call_vlm(ans_prompt, crop_path, max_tokens=max_tokens)
            elif hasattr(client, "call"):
                res_dict = client.call(ans_prompt, image_path=crop_path, max_tokens=max_tokens)
                ans_text = res_dict.get("text", "")
                p_tok = res_dict.get("prompt_tokens") if res_dict.get("prompt_tokens") is not None else 2800
                c_tok = res_dict.get("completion_tokens") if res_dict.get("completion_tokens") is not None else 60
                lat = res_dict.get("latency_sec") or 0.0
            else:
                ans_text = "Acquired evidence answer"
                p_tok, c_tok, lat = 100, 20, 0.2

            result.num_calls += 1
            result.prompt_tokens += p_tok
            result.completion_tokens += c_tok
            result.total_tokens += (p_tok + c_tok)
            result.final_answer = ans_text.strip()
            result.cost_usd = (result.total_tokens / 1000.0) * 0.00015
            result.status = "SUCCESS"
        except BudgetExceededError as be:
            result.status = "BLOCKED_BUDGET_EXCEEDED"
            result.failure_reason = f"Budget guard refused intervention request: {be}"
            result.final_answer = "BLOCKED_BUDGET_EXCEEDED"
            result.latency = time.time() - t0
            raise be
        except Exception as e:
            result.status = "ERROR"
            result.failure_reason = f"Intervention VLM call on crop failed: {e}"
            result.final_answer = initial_answer  # Fall back to initial answer

        result.latency = time.time() - t0
        return result
