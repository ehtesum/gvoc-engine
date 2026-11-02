"""Deterministic crop validation for context-preserving visual evidence acquisition.

Boxes use integer pixel coordinates [ymin, xmin, ymax, xmax] with exclusive
maximum bounds. Required context regions must come from annotation or a
grounded locator; geometry alone cannot infer semantic context.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Tuple, Sequence, Any, Optional, Union, Dict

Box = Tuple[int, int, int, int]


class ContextKind(str, Enum):
    TABLE_HEADER = "table_header"
    ROW_LABEL = "row_label"
    COLUMN_LABEL = "column_label"
    UNIT = "unit"
    LEGEND = "legend"
    AXIS_TICK = "axis_tick"
    FOOTNOTE = "footnote"
    NEIGHBORING_CELL = "neighboring_cell"


@dataclass(frozen=True)
class EvidenceRegion:
    target_box: Box
    required_context: Tuple[Tuple[ContextKind, Box], ...] = ()
    evidence_id: str = ""


@dataclass(frozen=True)
class CropAssessment:
    valid: bool
    target_retained: bool
    retained_context: Tuple[ContextKind, ...]
    missing_context: Tuple[ContextKind, ...]
    coverage: float


def validate_box(box: Any, width: int, height: int) -> Tuple[bool, Optional[str]]:
    """Strictly validates a box [ymin, xmin, ymax, xmax] against image dimensions.
    Fails closed if any coordinate is invalid, out of order, or out of bounds.
    """
    if width <= 0 or height <= 0:
        return False, f"Image dimensions must be positive, got width={width}, height={height}"
    if not isinstance(box, (list, tuple)):
        return False, f"Box must be a list or tuple of 4 coordinates, got {type(box).__name__}"
    if len(box) != 4:
        return False, f"Box must contain exactly 4 coordinates, got length {len(box)}"
    for idx, v in enumerate(box):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False, f"Coordinate at index {idx} must be numeric, got {type(v).__name__}: {v}"
        if isinstance(v, float) and not v.is_integer() and (v > 1.0 or v < 0.0):
            return False, f"Coordinate at index {idx} has fractional pixel value {v}"
    
    y0, x0, y1, x1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if y0 < 0 or x0 < 0:
        return False, f"Negative coordinate: y0={y0}, x0={x0}"
    if y0 >= y1:
        return False, f"Invalid vertical order or zero height: ymin={y0} >= ymax={y1}"
    if x0 >= x1:
        return False, f"Invalid horizontal order or zero width: xmin={x0} >= xmax={x1}"
    if y1 > height:
        return False, f"ymax={y1} exceeds image height={height}"
    if x1 > width:
        return False, f"xmax={x1} exceeds image width={width}"
    return True, None


def _valid_box(box: Sequence[int], width: int, height: int) -> bool:
    valid, _ = validate_box(box, width, height)
    return valid


def intersection_area(a: Box, b: Box) -> int:
    y0, x0 = max(a[0], b[0]), max(a[1], b[1])
    y1, x1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, y1 - y0) * max(0, x1 - x0)


def containment_ratio(region: Box, crop: Box) -> float:
    area = max(0, region[2] - region[0]) * max(0, region[3] - region[1])
    return 0.0 if area == 0 else intersection_area(region, crop) / area


def expand_box(box: Box, width: int, height: int, margin_fraction: float = 0.20) -> Box:
    """Expand a valid box and clip it to image boundaries."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    if not 0.0 <= margin_fraction <= 2.0:
        raise ValueError("margin_fraction must be in [0, 2].")
    if not _valid_box(box, width, height):
        raise ValueError("box must be a valid [ymin, xmin, ymax, xmax] image box.")
    y0, x0, y1, x1 = box
    dy, dx = round((y1 - y0) * margin_fraction), round((x1 - x0) * margin_fraction)
    return max(0, y0-dy), max(0, x0-dx), min(height, y1+dy), min(width, x1+dx)


def assess_crop(
    crop_box: Box,
    evidence: EvidenceRegion,
    image_width: int,
    image_height: int,
    min_target_coverage: float = 0.98,
    min_context_coverage: float = 0.95,
) -> CropAssessment:
    """Reject a crop if it loses the target or any required context region."""
    if not _valid_box(crop_box, image_width, image_height):
        raise ValueError("crop_box must be a valid image box.")
    if not 0.0 <= min_target_coverage <= 1.0 or not 0.0 <= min_context_coverage <= 1.0:
        raise ValueError("Coverage thresholds must be in [0, 1].")
    if not _valid_box(evidence.target_box, image_width, image_height):
        raise ValueError("target_box must be a valid image box.")
    for _, box in evidence.required_context:
        if not _valid_box(box, image_width, image_height):
            raise ValueError("Each context box must be valid and inside the image.")

    target_cov = containment_ratio(evidence.target_box, crop_box)
    retained: List[ContextKind] = []
    missing: List[ContextKind] = []
    coverages = [target_cov]
    for kind, box in evidence.required_context:
        ratio = containment_ratio(box, crop_box)
        coverages.append(ratio)
        (retained if ratio >= min_context_coverage else missing).append(kind)
    target_ok = target_cov >= min_target_coverage
    return CropAssessment(
        valid=target_ok and not missing,
        target_retained=target_ok,
        retained_context=tuple(retained),
        missing_context=tuple(missing),
        coverage=sum(coverages) / len(coverages),
    )


def propose_context_preserving_crop(
    target_box: Box,
    required_context_boxes: Iterable[Box],
    width: int,
    height: int,
    margin_fraction: float = 0.05,
) -> Box:
    """Build a box covering target plus declared context, then add a margin."""
    boxes = [target_box, *list(required_context_boxes)]
    if not boxes or any(not _valid_box(b, width, height) for b in boxes):
        raise ValueError("Every target/context box must be valid and inside the image.")
    union = (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )
    return expand_box(union, width, height, margin_fraction)


def format_box(box: Box) -> str:
    """Formats a box as standard [ymin, xmin, ymax, xmax] string."""
    return f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]"


def parse_bounding_box(raw_input: Any, width: int, height: int) -> Tuple[Optional[Box], str]:
    """Parses and validates a bounding box from arbitrary model text or structured output.
    Supports:
    - Direct list/tuple of 4 numeric values
    - JSON string or markdown block (e.g. ```json [10, 20, 100, 200] ``` or {"bbox": [...]})
    - Raw text regex match for [y0, x0, y1, x1]
    - Automatic scaling if coordinates are normalized in [0.0, 1.0]
    Fails closed: returns (None, error_reason) if coordinates cannot be parsed or fail geometry validation.
    """
    import re
    import json

    if raw_input is None:
        return None, "Raw input is None"

    coords: Optional[List[float]] = None

    # Case 1: Structured dict
    if isinstance(raw_input, dict):
        if "bbox" in raw_input and isinstance(raw_input["bbox"], (list, tuple)):
            coords = [float(v) for v in raw_input["bbox"]]
        elif all(k in raw_input for k in ("ymin", "xmin", "ymax", "xmax")):
            coords = [float(raw_input["ymin"]), float(raw_input["xmin"]), float(raw_input["ymax"]), float(raw_input["xmax"])]
        else:
            return None, f"Dict does not contain 'bbox' or ('ymin','xmin','ymax','xmax') keys: {list(raw_input.keys())}"

    # Case 2: Direct sequence of numbers
    elif isinstance(raw_input, (list, tuple)):
        if len(raw_input) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw_input):
            coords = [float(v) for v in raw_input]
        else:
            return None, f"List/tuple must contain 4 numbers, got: {raw_input}"

    # Case 3: String output
    elif isinstance(raw_input, str):
        text = raw_input.strip()
        # Clean markdown code blocks if present
        if "```" in text:
            blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if blocks:
                text = blocks[0].strip()

        # Try parsing as JSON first
        try:
            parsed_json = json.loads(text)
            if isinstance(parsed_json, (list, tuple)) and len(parsed_json) == 4:
                coords = [float(v) for v in parsed_json]
            elif isinstance(parsed_json, dict):
                if "bbox" in parsed_json and isinstance(parsed_json["bbox"], (list, tuple)):
                    coords = [float(v) for v in parsed_json["bbox"]]
                elif all(k in parsed_json for k in ("ymin", "xmin", "ymax", "xmax")):
                    coords = [float(parsed_json["ymin"]), float(parsed_json["xmin"]), float(parsed_json["ymax"]), float(parsed_json["xmax"])]
        except Exception:
            pass

        # If JSON parsing did not succeed, use regex pattern matching for 4 numbers inside brackets or commas
        if coords is None:
            match = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text)
            if match:
                coords = [float(match.group(i)) for i in range(1, 5)]
            else:
                # Try finding any sequence of 4 numbers
                num_matches = re.findall(r"-?\d+(?:\.\d+)?", text)
                if len(num_matches) >= 4:
                    coords = [float(num_matches[0]), float(num_matches[1]), float(num_matches[2]), float(num_matches[3])]

    if coords is None or len(coords) != 4:
        return None, f"Could not extract 4 bounding box coordinates from output: {raw_input!r}"

    # Check if coordinates are normalized in [0.0, 1.0] (and not all zero/integer)
    is_normalized = all(0.0 <= c <= 1.0 for c in coords) and any(0.0 < c < 1.0 for c in coords)
    if is_normalized:
        y0 = round(coords[0] * height)
        x0 = round(coords[1] * width)
        y1 = round(coords[2] * height)
        x1 = round(coords[3] * width)
    else:
        y0, x0, y1, x1 = round(coords[0]), round(coords[1]), round(coords[2]), round(coords[3])

    cand_box: Box = (y0, x0, y1, x1)
    is_valid, reason = validate_box(cand_box, width, height)
    if not is_valid:
        return None, f"Parsed box {cand_box} failed validation: {reason}"

    return cand_box, "Successfully parsed and validated"
