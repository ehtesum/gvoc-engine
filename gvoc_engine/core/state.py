"""
Grounded Intermediate State Representation s_t in R^d.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional


@dataclass
class GroundedState:
    """
    Multimodal intermediate reasoning state vector.
    Treated as an initial candidate vector subject to feature ablation.
    """
    # 1. Semantic confidences from agent outputs
    c_plan: float = 0.85        # Planner confidence [0, 1]
    c_exec: float = 0.80        # Executor confidence [0, 1]
    c_judg: float = 0.85        # Judge/Verifier confidence [0, 1]
    c_consensus: float = 0.90   # Multi-agent agreement ratio [0, 1]

    # 2. Grounded geometric and visual alignment signals
    d_layout: float = 0.05      # Geometric layout loss [0, 1] (lower is better aligned)
    d_traj: float = 0.08        # Trajectory drift indicator [0, 1]
    d_evidence: float = 0.10    # Evidence-to-visual crop discrepancy [0, 1]
    d_answer: float = 0.05      # Perturbation instability of answer [0, 1]
    rho_visual: float = 0.50    # Document visual density (elements/area) [0, 1]

    # 3. Dynamic compute budget constraints
    b_remaining: int = 2000     # Remaining token budget B_t
    n_tokens_spent: int = 250   # Accumulated inference tokens

    # Metadata (non-vector)
    current_answer: str = ""
    current_bbox: Optional[List[int]] = None
    step_index: int = 0

    def to_vector(self) -> List[float]:
        """Converts state into a normalized 11-dimensional feature vector."""
        return [
            self.c_plan,
            self.c_exec,
            self.c_judg,
            self.c_consensus,
            self.d_layout,
            self.d_traj,
            self.d_evidence,
            self.d_answer,
            self.rho_visual,
            min(1.0, self.b_remaining / 4000.0),
            min(1.0, self.n_tokens_spent / 4000.0)
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes state for database logging."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundedState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def build_grounded_state_vector(
    C_plan: float = 0.85,
    C_exec: float = 0.80,
    C_judg: float = 0.85,
    C_consensus: float = 0.90,
    D_layout: float = 0.05,
    D_traj: float = 0.08,
    D_evidence: float = 0.10,
    D_answer: float = 0.05,
    rho_visual: float = 0.50,
    B_t: float = 1.0,
    N_tokens: int = 250
) -> GroundedState:
    """Convenience constructor for GroundedState from mathematical parameter names."""
    return GroundedState(
        c_plan=C_plan,
        c_exec=C_exec,
        c_judg=C_judg,
        c_consensus=C_consensus,
        d_layout=D_layout,
        d_traj=D_traj,
        d_evidence=D_evidence,
        d_answer=D_answer,
        rho_visual=rho_visual,
        b_remaining=int(B_t * 2000),
        n_tokens_spent=N_tokens
    )
