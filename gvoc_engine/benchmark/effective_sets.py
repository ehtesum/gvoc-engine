"""
Multi-Label Effective Intervention Sets A* for Diagnostic Benchmark.
Prevents penalizing controllers that select alternative valid repairs.
"""

from typing import Set, Dict
from gvoc_engine.core.action_space import Action


# Ground-truth mapping from failure mode to Effective Intervention Set A*
EFFECTIVE_SETS_MAP: Dict[str, Set[Action]] = {
    "spatial_shift_10px": {Action.REGROUND, Action.VERIFY},
    "spatial_shift_20px": {Action.REGROUND},
    "spatial_shift_30px": {Action.REGROUND},
    "spatial_shift_50px": {Action.REGROUND, Action.REPLAN},
    "reading_order_inversion": {Action.REPLAN, Action.REGROUND},
    "table_cell_drift": {Action.REGROUND, Action.REEXEC},
    "ocr_noise": {Action.REEXEC, Action.REGROUND},
    "arithmetic_swap": {Action.REEXEC, Action.VERIFY},
    "plan_step_deletion": {Action.REPLAN},
    "normal_correct_state": {Action.STOP}
}


def get_effective_intervention_set(failure_type: str) -> Set[Action]:
    """Returns the multi-label set of valid actions that resolve the failure."""
    return EFFECTIVE_SETS_MAP.get(failure_type, {Action.STOP})
