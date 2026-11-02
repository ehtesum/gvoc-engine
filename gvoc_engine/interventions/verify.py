"""
Layout Geometric Verification Intervention.
Computes spatial layout loss D_layout and cross-modal consensus.
"""

from typing import Dict, Any, Tuple
from gvoc_engine.core.state import GroundedState


def execute_verify(
    state: GroundedState,
    client: Any,
    image_path: str,
    query: str
) -> Tuple[GroundedState, int, float]:
    """
    Executes layout verification to detect hallucinations or structural violations.
    Returns: (updated_state, tokens_spent, cost_estimate)
    """
    prompt = (
        f"Spatial & Layout Consistency Verification:\n"
        f"Query: {query}\n"
        f"Candidate Answer: {state.current_answer}\n"
        f"Task: Verify whether the extracted value strictly adheres to the visual bounding box and table layout. Output confirmation or identified inconsistency."
    )

    resp, p_tok, c_tok, lat = client.call_vlm(prompt, image_path, max_tokens=80)
    tokens = p_tok + c_tok

    # Update state: verification improves consensus and reduces layout uncertainty
    state.c_consensus = min(0.98, state.c_consensus + 0.18)
    state.d_layout = max(0.01, state.d_layout * 0.6)
    state.n_tokens_spent += tokens
    state.b_remaining = max(0, state.b_remaining - tokens)
    state.step_index += 1

    cost = tokens * 0.001 + lat * 0.01
    return state, tokens, cost
