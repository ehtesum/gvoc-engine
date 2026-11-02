"""
Synthetic Multi-Severity Failure Perturbations for Recovery Benchmark.
"""

import random
from typing import List, Dict, Any, Tuple
from gvoc_engine.core.state import GroundedState


class FailurePerturbationEngine:
    """Injects controlled, multi-severity failures into intermediate states."""

    SEVERITY_LEVELS = [10, 20, 30, 50]  # Pixel shifts or percentage noise

    @classmethod
    def inject_spatial_shift(
        cls,
        state: GroundedState,
        severity_px: int = 30
    ) -> Tuple[GroundedState, str]:
        """Shifts visual bounding box by +/- severity_px, degrading layout alignment."""
        if state.current_bbox:
            dy = random.choice([-severity_px, severity_px])
            dx = random.choice([-severity_px, severity_px])
            state.current_bbox = [
                state.current_bbox[0] + dy,
                state.current_bbox[1] + dx,
                state.current_bbox[2] + dy,
                state.current_bbox[3] + dx
            ]
        # Layout loss degrades with perturbation severity
        state.d_layout = min(0.95, 0.15 + (severity_px / 70.0))
        state.d_evidence = min(0.90, 0.20 + (severity_px / 80.0))
        state.c_judg = max(0.20, state.c_judg - 0.35)
        return state, f"spatial_shift_{severity_px}px"

    @classmethod
    def inject_reading_order_inversion(cls, state: GroundedState) -> Tuple[GroundedState, str]:
        """Inverts reading order DAG, causing trajectory drift."""
        state.d_traj = 0.55
        state.c_plan = 0.45
        return state, "reading_order_inversion"

    @classmethod
    def inject_ocr_character_noise(cls, state: GroundedState) -> Tuple[GroundedState, str]:
        """Injects character noise, degrading executor extraction confidence."""
        state.c_exec = 0.40
        state.d_answer = 0.50
        return state, "ocr_noise"

    @classmethod
    def inject_arithmetic_swap(cls, state: GroundedState) -> Tuple[GroundedState, str]:
        """Perturbs calculation logic."""
        state.d_answer = 0.65
        state.c_exec = 0.50
        return state, "arithmetic_swap"


def apply_visual_document_perturbation(failure_type: str) -> Dict[str, float]:
    """
    Returns the grounded divergence signals resulting from a simulated visual document failure.
    """
    if failure_type == "normal_correct_state":
        return {
            "layout_divergence": 0.05,
            "reading_order_divergence": 0.04,
            "evidence_divergence": 0.06,
            "consensus": 0.95
        }
    elif "spatial_shift" in failure_type:
        px = 30
        if "10px" in failure_type:
            px = 10
        elif "20px" in failure_type:
            px = 20
        elif "50px" in failure_type:
            px = 50
        return {
            "layout_divergence": min(0.95, 0.15 + (px / 70.0)),
            "reading_order_divergence": 0.08,
            "evidence_divergence": min(0.90, 0.20 + (px / 80.0)),
            "consensus": max(0.20, 0.90 - (px / 50.0))
        }
    elif failure_type == "reading_order_inversion":
        return {
            "layout_divergence": 0.10,
            "reading_order_divergence": 0.65,
            "evidence_divergence": 0.15,
            "consensus": 0.40
        }
    elif failure_type == "ocr_noise":
        return {
            "layout_divergence": 0.12,
            "reading_order_divergence": 0.10,
            "evidence_divergence": 0.55,
            "consensus": 0.45
        }
    elif failure_type == "arithmetic_swap":
        return {
            "layout_divergence": 0.06,
            "reading_order_divergence": 0.08,
            "evidence_divergence": 0.35,
            "consensus": 0.30
        }
    elif failure_type == "plan_step_deletion":
        return {
            "layout_divergence": 0.08,
            "reading_order_divergence": 0.50,
            "evidence_divergence": 0.40,
            "consensus": 0.35
        }
    elif failure_type == "unrecoverable_noise":
        return {
            "layout_divergence": 0.92,
            "reading_order_divergence": 0.88,
            "evidence_divergence": 0.95,
            "consensus": 0.10
        }
    else:
        return {
            "layout_divergence": 0.25,
            "reading_order_divergence": 0.20,
            "evidence_divergence": 0.25,
            "consensus": 0.50
        }

