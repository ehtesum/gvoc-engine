"""
Action Space Definition for Visual Document Process Repair.
"""

from enum import Enum
from typing import List, Set


class Action(str, Enum):
    """The 6 fundamental reasoning-repair interventions in GVoC."""
    STOP = "stop"              # Terminate and output current best answer
    REGROUND = "reground"      # Spatial bounding-box refinement & crop re-alignment
    REPLAN = "replan"          # Query sub-goal decomposition & re-planning
    REEXEC = "reexec"          # Deterministic tool execution / table cell OCR re-parsing
    VERIFY = "verify"          # Multi-modal layout geometric consistency check
    ABSTAIN = "abstain"        # Calibrated safe abstention when reliability is irrecoverable


GVoCAction = Action


class ActionSpace:
    """Manages the partition between active process repairs and terminal operators."""

    ALL_ACTIONS: List[Action] = [
        Action.STOP,
        Action.REGROUND,
        Action.REPLAN,
        Action.REEXEC,
        Action.VERIFY,
        Action.ABSTAIN
    ]

    # Active repair interventions (incur non-zero test-time compute)
    ACTIVE_ACTIONS: List[Action] = [
        Action.REGROUND,
        Action.REPLAN,
        Action.REEXEC,
        Action.VERIFY
    ]

    # Terminal operators
    TERMINAL_ACTIONS: List[Action] = [
        Action.STOP,
        Action.ABSTAIN
    ]

    @classmethod
    def is_active(cls, action: Action) -> bool:
        return action in cls.ACTIVE_ACTIONS

    @classmethod
    def is_terminal(cls, action: Action) -> bool:
        return action in cls.TERMINAL_ACTIONS
