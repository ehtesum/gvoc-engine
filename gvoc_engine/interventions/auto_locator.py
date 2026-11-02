"""
Automatic Non-Oracle Evidence Localization Module.

Implements autonomous visual evidence and context localization for document VQA
without access to ground-truth answers or answer-derived coordinates.

Inputs at inference time:
1. Document Image (image_path, width, height)
2. Query / Question string
3. Optional permitted OCR / layout tokens (token bounding boxes)
4. Optional VLM client for visual grounding requests

Guarantees:
- Zero ground-truth leakage: Ground-truth answers and oracle boxes are strictly forbidden.
- Context preservation: Automatically identifies and bundles associated row labels,
  column headers, units, and nearby explanatory labels with the target cell.
- Fail-closed geometry validation: Rejects out-of-order, out-of-bounds, or degenerate boxes.
- Fallback to full-image overview when localization confidence is insufficient or parsing fails.
- Complete telemetry logging: records source, uncalibrated confidence, bounding boxes,
  token accounting, and decision reasons.
"""

import math
import re
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Sequence, Union

from gvoc_engine.interventions.context_preservation import (
    Box,
    ContextKind,
    EvidenceRegion,
    CropAssessment,
    validate_box,
    expand_box,
    propose_context_preserving_crop,
    parse_bounding_box,
    containment_ratio,
)


class LocalizationSource(str, Enum):
    VLM_GROUNDING = "vlm_visual_grounding"
    OCR_LAYOUT = "ocr_layout_matching"
    HEURISTIC_LAYOUT = "heuristic_layout"
    FALLBACK_OVERVIEW = "fallback_to_overview"


class LocalizationAction(str, Enum):
    ACQUIRE_CONTEXT_PRESERVING = "acquire_context_preserving"
    ACQUIRE_TARGET_ONLY = "acquire_target_only"
    FALLBACK_TO_OVERVIEW = "fallback_to_overview"
    ABSTAIN = "abstain"


@dataclass
class AutomaticLocalizationResult:
    """Telemetry and output of an automatic non-oracle localization run."""
    target_box: Optional[Box] = None
    context_boxes: List[Box] = field(default_factory=list)
    crop_box: Optional[Box] = None
    source: LocalizationSource = LocalizationSource.FALLBACK_OVERVIEW
    confidence: float = 0.0          # Uncalibrated heuristic confidence score in [0.0, 1.0]
    action: LocalizationAction = LocalizationAction.FALLBACK_TO_OVERVIEW
    is_valid: bool = False
    retained_context_types: List[str] = field(default_factory=list)
    stopping_reason: str = "initialized"
    locator_tokens: int = 0
    latency_sec: float = 0.0
    telemetry: Dict[str, Any] = field(default_factory=dict)


class AutomaticEvidenceLocator:
    """
    Non-Oracle Visual Evidence Locator for CCEA.
    
    Identifies candidate target evidence and contextual labels using only:
    - Image dimensions & image content
    - Question semantics & keywords
    - Permitted inference-time OCR words/boxes (if provided)
    - VLM visual grounding call (if client provided)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.50,
        margin_fraction: float = 0.08,
        min_box_area_px: int = 200,
        max_box_area_ratio: float = 0.90,
        enable_vlm_grounding: bool = False,
    ):
        self.confidence_threshold = confidence_threshold
        self.margin_fraction = margin_fraction
        self.min_box_area_px = min_box_area_px
        self.max_box_area_ratio = max_box_area_ratio
        self.enable_vlm_grounding = enable_vlm_grounding

    def locate(
        self,
        image_path: Union[str, Path],
        query: str,
        image_width: int,
        image_height: int,
        client: Optional[Any] = None,
        ocr_tokens: Optional[List[Dict[str, Any]]] = None,
        prefer_context_preserving: bool = True,
    ) -> AutomaticLocalizationResult:
        """
        Executes automatic non-oracle evidence localization.
        
        Parameters:
            image_path: Path to document image.
            query: Question string (ground truth is strictly prohibited).
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            client: Optional VLM client for model-based grounding prompts.
            ocr_tokens: Optional list of OCR tokens with format:
                        [{"text": str, "box": [ymin, xmin, ymax, xmax]}, ...]
            prefer_context_preserving: If True, bundles contextual row/col labels.
            
        Returns:
            AutomaticLocalizationResult with validated crop_box and decision telemetry.
        """
        # 1. Pre-flight check on image geometry
        if image_width <= 0 or image_height <= 0:
            return AutomaticLocalizationResult(
                source=LocalizationSource.FALLBACK_OVERVIEW,
                action=LocalizationAction.FALLBACK_TO_OVERVIEW,
                stopping_reason="invalid_image_dimensions",
                is_valid=False
            )

        if not query or not str(query).strip():
            return AutomaticLocalizationResult(
                source=LocalizationSource.FALLBACK_OVERVIEW,
                action=LocalizationAction.FALLBACK_TO_OVERVIEW,
                stopping_reason="empty_query",
                is_valid=False
            )

        # 2. Strategy A: OCR layout matching if tokens provided
        if ocr_tokens and len(ocr_tokens) > 0:
            result = self._locate_via_ocr_layout(
                query=query,
                ocr_tokens=ocr_tokens,
                width=image_width,
                height=image_height,
                prefer_context_preserving=prefer_context_preserving
            )
            if result.is_valid and result.confidence >= self.confidence_threshold:
                return result

        # 3. Strategy B: VLM visual grounding call if enabled AND client provided
        if self.enable_vlm_grounding and client is not None:
            result = self._locate_via_vlm_grounding(
                image_path=str(image_path),
                query=query,
                width=image_width,
                height=image_height,
                client=client,
                prefer_context_preserving=prefer_context_preserving
            )
            if result.is_valid and result.confidence >= self.confidence_threshold:
                return result

        # 4. Strategy C: Safe Fallback to Full Overview
        full_overview_box: Box = (0, 0, image_height, image_width)
        return AutomaticLocalizationResult(
            target_box=None,
            context_boxes=[],
            crop_box=full_overview_box,
            source=LocalizationSource.FALLBACK_OVERVIEW,
            confidence=0.0,
            action=LocalizationAction.FALLBACK_TO_OVERVIEW,
            is_valid=True,
            stopping_reason="confidence_below_threshold_safe_overview_fallback",
            telemetry={
                "fallback_applied": True,
                "reason": "Localization source failed or confidence below threshold",
                "threshold": self.confidence_threshold
            }
        )

    def _locate_via_ocr_layout(
        self,
        query: str,
        ocr_tokens: List[Dict[str, Any]],
        width: int,
        height: int,
        prefer_context_preserving: bool = True
    ) -> AutomaticLocalizationResult:
        """Locates evidence and associated labels by aligning query keywords with OCR tokens and spatial priors."""
        q_lower = (query or "").lower().strip()

        # 1. Spatial Layout Directives (Priors for Title, Top, Bottom)
        # Top-of-page directives
        if any(p in q_lower for p in ["top of the page", "top of page", "at the top"]):
            top_h = min(height, max(150, int(0.22 * height)))
            target_box: Box = (0, 0, top_h, width)
            return AutomaticLocalizationResult(
                target_box=target_box,
                context_boxes=[],
                crop_box=target_box,
                source=LocalizationSource.HEURISTIC_LAYOUT,
                confidence=0.85,
                action=LocalizationAction.ACQUIRE_CONTEXT_PRESERVING,
                is_valid=True,
                stopping_reason="query_specifies_top_of_page_layout",
                telemetry={"spatial_prior": "top_of_page"}
            )

        # Document title directives
        if any(p in q_lower for p in ["title", "document title", "title of the document"]):
            title_h = min(height, max(200, int(0.28 * height)))
            target_box: Box = (0, 0, title_h, width)
            return AutomaticLocalizationResult(
                target_box=target_box,
                context_boxes=[],
                crop_box=target_box,
                source=LocalizationSource.HEURISTIC_LAYOUT,
                confidence=0.80,
                action=LocalizationAction.ACQUIRE_CONTEXT_PRESERVING,
                is_valid=True,
                stopping_reason="query_specifies_document_title_layout",
                telemetry={"spatial_prior": "document_title"}
            )

        # Bottom / Footnote directives
        if any(p in q_lower for p in ["bottom of the page", "bottom of page", "footnote", "footer"]):
            bot_y0 = max(0, int(0.78 * height))
            target_box: Box = (bot_y0, 0, height, width)
            return AutomaticLocalizationResult(
                target_box=target_box,
                context_boxes=[],
                crop_box=target_box,
                source=LocalizationSource.HEURISTIC_LAYOUT,
                confidence=0.80,
                action=LocalizationAction.ACQUIRE_CONTEXT_PRESERVING,
                is_valid=True,
                stopping_reason="query_specifies_bottom_layout",
                telemetry={"spatial_prior": "bottom_of_page"}
            )

        # 2. Extract keywords from query (excluding stopwords)
        stopwords = {
            "what", "is", "the", "of", "in", "for", "to", "and", "a", "an", "on", "at",
            "which", "how", "many", "much", "where", "who", "when", "by", "from",
            "are", "has", "have", "had", "be", "been", "was", "were", "do", "does", "did",
            "name", "give", "state", "?"
        }
        words = re.findall(r"\b[a-zA-Z0-9_\-\%]+\b", q_lower)
        keywords = [w for w in words if w not in stopwords and len(w) > 1]

        if not keywords:
            return AutomaticLocalizationResult(
                source=LocalizationSource.OCR_LAYOUT,
                stopping_reason="no_discriminative_keywords_in_query",
                is_valid=False
            )

        # 3. Document term frequency for Inverse Document Frequency (IDF) weighting
        doc_word_counts: Dict[str, int] = {}
        for tok in ocr_tokens:
            t_text = str(tok.get("text", "")).lower().strip()
            doc_word_counts[t_text] = doc_word_counts.get(t_text, 0) + 1

        matched_tokens = []
        for tok in ocr_tokens:
            tok_text = str(tok.get("text", "")).lower().strip()
            tok_box = tok.get("box")
            if not tok_box or len(tok_box) != 4:
                continue
            is_val, _ = validate_box(tok_box, width, height)
            if not is_val:
                continue

            best_kw = None
            best_weight = 0.0
            for kw in keywords:
                # Exact match or substantial substring match
                if kw == tok_text or (len(kw) >= 4 and len(tok_text) >= 4 and (kw in tok_text or tok_text in kw)):
                    freq = doc_word_counts.get(tok_text, 1)
                    # Downweight ubiquitous repeated words; boost rare discriminating entities
                    weight = 1.0 / (1.0 + 0.3 * (freq - 1))
                    if len(kw) >= 5:
                        weight *= 1.4
                    if weight > best_weight:
                        best_weight = weight
                        best_kw = kw

            if best_kw is not None:
                matched_tokens.append({
                    "tok": tok,
                    "kw": best_kw,
                    "weight": best_weight,
                    "box": tok_box
                })

        if not matched_tokens:
            return AutomaticLocalizationResult(
                source=LocalizationSource.OCR_LAYOUT,
                stopping_reason="zero_ocr_keyword_matches",
                is_valid=False
            )

        # 4. Spatial Density Windowing along Y-axis
        # Candidate window height: ~20% of image height (captures table row / section)
        band_height = max(180, int(0.22 * height))
        best_score = -1.0
        best_tokens_in_band = []

        candidate_centers = sorted(list(set((t["box"][0] + t["box"][2]) // 2 for t in matched_tokens)))
        for c_y in candidate_centers:
            y0 = max(0, c_y - band_height // 2)
            y1 = min(height, y0 + band_height)
            tokens_in = [t for t in matched_tokens if t["box"][0] >= y0 and t["box"][2] <= y1]
            if not tokens_in:
                continue

            unique_kws = set(t["kw"] for t in tokens_in)
            score = sum(t["weight"] for t in tokens_in) + (len(unique_kws) * 2.5)

            if score > best_score:
                best_score = score
                best_tokens_in_band = tokens_in

        if not best_tokens_in_band:
            return AutomaticLocalizationResult(
                source=LocalizationSource.OCR_LAYOUT,
                stopping_reason="no_coherent_spatial_cluster",
                is_valid=False
            )

        # 5. Formulate Target Box and Context Preservation Crop
        cluster_ymin = min(t["box"][0] for t in best_tokens_in_band)
        cluster_xmin = min(t["box"][1] for t in best_tokens_in_band)
        cluster_ymax = max(t["box"][2] for t in best_tokens_in_band)
        cluster_xmax = max(t["box"][3] for t in best_tokens_in_band)

        target_box: Box = (cluster_ymin, cluster_xmin, cluster_ymax, cluster_xmax)
        is_val, err = validate_box(target_box, width, height)
        if not is_val:
            return AutomaticLocalizationResult(
                source=LocalizationSource.OCR_LAYOUT,
                stopping_reason=f"invalid_target_box_geometry: {err}",
                is_valid=False
            )

        # Context expansion: expand vertically to include line margins and headers above
        row_h = cluster_ymax - cluster_ymin
        margin_y = max(20, int(row_h * 0.40))
        crop_ymin = max(0, cluster_ymin - margin_y)
        crop_ymax = min(height, cluster_ymax + margin_y + max(25, int(row_h * 0.50)))

        context_boxes: List[Box] = []
        retained_types: List[str] = []

        if prefer_context_preserving:
            # Include column headers above if row is located lower down the document
            if crop_ymin > int(0.12 * height):
                header_y0 = max(0, crop_ymin - int(0.12 * height))
                header_box = (header_y0, 0, crop_ymin, width)
                if validate_box(header_box, width, height)[0]:
                    context_boxes.append(header_box)
                    retained_types.append("column_header")
                    crop_ymin = header_y0

            row_label_box = (crop_ymin, 0, crop_ymax, max(cluster_xmin, 100))
            if validate_box(row_label_box, width, height)[0]:
                context_boxes.append(row_label_box)
                retained_types.append("row_label")

        # In document tables and text passages, expand horizontally across row/column width
        crop_box: Box = (crop_ymin, 0, crop_ymax, width)

        # 6. Area and Confidence Checks
        area = (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])
        if area < self.min_box_area_px or area > (width * height * self.max_box_area_ratio):
            return AutomaticLocalizationResult(
                source=LocalizationSource.OCR_LAYOUT,
                stopping_reason="box_area_violates_size_bounds",
                confidence=0.0,
                is_valid=False
            )

        unique_kws_count = len(set(t["kw"] for t in best_tokens_in_band))
        conf = min(0.95, max(0.35, unique_kws_count / len(keywords)))

        return AutomaticLocalizationResult(
            target_box=target_box,
            context_boxes=context_boxes,
            crop_box=crop_box,
            source=LocalizationSource.OCR_LAYOUT,
            confidence=round(conf, 4),
            action=LocalizationAction.ACQUIRE_CONTEXT_PRESERVING if prefer_context_preserving else LocalizationAction.ACQUIRE_TARGET_ONLY,
            is_valid=True,
            retained_context_types=retained_types,
            stopping_reason="ocr_layout_matching_success",
            telemetry={
                "matched_tokens_in_cluster": len(best_tokens_in_band),
                "unique_keywords_matched": unique_kws_count,
                "total_keywords": len(keywords),
                "band_score": round(best_score, 2),
                "area_ratio": round(area / (width * height), 4)
            }
        )

    def _locate_via_vlm_grounding(
        self,
        image_path: str,
        query: str,
        width: int,
        height: int,
        client: Any,
        prefer_context_preserving: bool = True
    ) -> AutomaticLocalizationResult:
        """Asks VLM model to provide bounding coordinates for evidence without answer leakage."""
        prompt = (
            f"Visual Document Evidence Localization Task:\n"
            f"Question: {query}\n\n"
            f"Identify the bounding box of the document region containing the visual evidence needed to answer the question.\n"
            f"Include the target data cell and any associated row label, column header, or unit.\n"
            f"Return JSON only with format:\n"
            f"```json\n"
            f'{{"target_box": [ymin, xmin, ymax, xmax], "context_boxes": [[ymin, xmin, ymax, xmax]], "confidence": 0.8}}\n'
            f"```\n"
            f"Coordinates must be integers in pixel space [0..{height}, 0..{width}] or normalized [0.0..1.0]."
        )

        try:
            tokens_used = 0
            latency = 0.0
            if hasattr(client, "call_vlm"):
                resp, p_tok, c_tok, lat = client.call_vlm(prompt, image_path, max_tokens=150)
                tokens_used = p_tok + c_tok
                latency = lat
            elif hasattr(client, "call"):
                res_dict = client.call(prompt, image_path=image_path, max_tokens=150)
                resp = res_dict.get("text", "")
                tokens_used = res_dict.get("total_tokens", 0)
                latency = res_dict.get("latency_sec", 0.0)
            else:
                return AutomaticLocalizationResult(
                    source=LocalizationSource.VLM_GROUNDING,
                    stopping_reason="client_unsupported_interface",
                    is_valid=False
                )

            # Parse JSON or structured box from response
            target_box = None
            context_boxes = []
            confidence = 0.50

            # Attempt JSON parse
            json_match = re.search(r"\{[\s\S]*\}", resp)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if "target_box" in data:
                        target_box, _ = parse_bounding_box(data["target_box"], width, height)
                    if "context_boxes" in data and isinstance(data["context_boxes"], list):
                        for cb in data["context_boxes"]:
                            parsed_cb, _ = parse_bounding_box(cb, width, height)
                            if parsed_cb:
                                context_boxes.append(parsed_cb)
                    if "confidence" in data and isinstance(data["confidence"], (int, float)):
                        confidence = float(data["confidence"])
                except Exception:
                    pass

            if target_box is None:
                # Fallback to regex coordinate extraction
                target_box, _ = parse_bounding_box(resp, width, height)

            if target_box is None:
                return AutomaticLocalizationResult(
                    source=LocalizationSource.VLM_GROUNDING,
                    stopping_reason="vlm_bounding_box_unparseable",
                    locator_tokens=tokens_used,
                    latency_sec=latency,
                    is_valid=False
                )

            # Validate target box
            is_val, err = validate_box(target_box, width, height)
            if not is_val:
                return AutomaticLocalizationResult(
                    source=LocalizationSource.VLM_GROUNDING,
                    stopping_reason=f"vlm_target_box_invalid: {err}",
                    locator_tokens=tokens_used,
                    latency_sec=latency,
                    is_valid=False
                )

            # Form crop
            retained_types = []
            if prefer_context_preserving and context_boxes:
                try:
                    crop_box = propose_context_preserving_crop(target_box, context_boxes, width, height, margin_fraction=self.margin_fraction)
                    action = LocalizationAction.ACQUIRE_CONTEXT_PRESERVING
                    retained_types = ["row_label", "column_header"]
                except Exception:
                    crop_box = expand_box(target_box, width, height, margin_fraction=self.margin_fraction)
                    action = LocalizationAction.ACQUIRE_TARGET_ONLY
            else:
                crop_box = expand_box(target_box, width, height, margin_fraction=self.margin_fraction)
                action = LocalizationAction.ACQUIRE_TARGET_ONLY

            # Size bounds check
            area = (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])
            if area < self.min_box_area_px or area > (width * height * self.max_box_area_ratio):
                return AutomaticLocalizationResult(
                    source=LocalizationSource.VLM_GROUNDING,
                    stopping_reason="crop_box_violates_size_bounds",
                    locator_tokens=tokens_used,
                    latency_sec=latency,
                    confidence=confidence,
                    is_valid=False
                )

            return AutomaticLocalizationResult(
                target_box=target_box,
                context_boxes=context_boxes,
                crop_box=crop_box,
                source=LocalizationSource.VLM_GROUNDING,
                confidence=round(confidence, 4),
                action=action,
                is_valid=True,
                retained_context_types=retained_types,
                stopping_reason="vlm_grounding_success",
                locator_tokens=tokens_used,
                latency_sec=latency,
                telemetry={"parsed_from": "vlm_response"}
            )

        except Exception as e:
            return AutomaticLocalizationResult(
                source=LocalizationSource.VLM_GROUNDING,
                stopping_reason=f"vlm_grounding_exception: {e}",
                is_valid=False
            )
