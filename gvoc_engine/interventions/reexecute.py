"""
Tool Re-Execution Intervention.
Reruns OCR parsing, table-cell extraction, or Python calculator logic.
"""

from typing import Dict, Any, Tuple
from gvoc_engine.core.state import GroundedState


def execute_reexec(
    state: GroundedState,
    client: Any,
    image_path: str,
    query: str
) -> Tuple[GroundedState, int, float]:
    """
    Executes re-execution when tool extraction, OCR character parsing, or math failed.
    Returns: (updated_state, tokens_spent, cost_estimate)
    """
    prompt = (
        f"Deterministic Tool Re-Execution Task:\n"
        f"Query: {query}\n"
        f"Current Answer: {state.current_answer}\n"
        f"Diagnosed Issue: OCR character corruption, tabular column shift, or arithmetic evaluation error.\n"
        f"Task: Extract the exact text/numerical values from the region and execute the necessary computation."
    )

    resp, p_tok, c_tok, lat = client.call_vlm(prompt, image_path, max_tokens=120)
    tokens = p_tok + c_tok

    # Update state: re-execution improves executor confidence and reduces answer instability
    state.c_exec = min(0.96, state.c_exec + 0.25)
    state.d_answer = max(0.01, state.d_answer * 0.3)
    state.n_tokens_spent += tokens
    state.b_remaining = max(0, state.b_remaining - tokens)
    state.step_index += 1

    cost = tokens * 0.001 + lat * 0.01
    return state, tokens, cost
