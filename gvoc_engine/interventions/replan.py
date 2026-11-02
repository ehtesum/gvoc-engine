"""
Query Decomposition & Re-Planning Intervention.
Breaks down complex compound questions into executable atomic sub-goals.
"""

from typing import Dict, Any, Tuple
from gvoc_engine.core.state import GroundedState


def execute_replan(
    state: GroundedState,
    client: Any,
    image_path: str,
    query: str
) -> Tuple[GroundedState, int, float]:
    """
    Executes re-planning when plan logic or sub-goal sequencing failed.
    Returns: (updated_state, tokens_spent, cost_estimate)
    """
    prompt = (
        f"Query Re-Planning Task:\n"
        f"Query: {query}\n"
        f"Current Answer Attempt: {state.current_answer}\n"
        f"Diagnosed Issue: Flawed sub-goal sequencing, missing condition, or multi-step logic drift.\n"
        f"Task: Provide a revised, step-by-step atomic decomposition plan to answer the query accurately."
    )

    resp, p_tok, c_tok, lat = client.call_vlm(prompt, image_path, max_tokens=150)
    tokens = p_tok + c_tok

    # Update state: re-planning improves plan confidence and trajectory alignment
    state.c_plan = min(0.95, state.c_plan + 0.20)
    state.d_traj = max(0.01, state.d_traj * 0.35)
    state.n_tokens_spent += tokens
    state.b_remaining = max(0, state.b_remaining - tokens)
    state.step_index += 1

    cost = tokens * 0.001 + lat * 0.01
    return state, tokens, cost
