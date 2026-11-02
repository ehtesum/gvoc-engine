"""
Mock VLM/LLM Client for Zero-Cost Offline Simulation & Unit Testing.
"""

import time
import random
from typing import Dict, Any, Tuple, Optional


class MockClient:
    """Simulates realistic VLM outputs, token counts, and latencies offline."""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def call_vlm(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        model: str = "mock-vlm",
        max_tokens: int = 150
    ) -> Tuple[str, int, int, float]:
        """
        Simulates an API call.
        Returns: (text_response, prompt_tokens, completion_tokens, latency_sec)
        """
        t0 = time.time()
        prompt_tokens = len(prompt.split()) + 30
        completion_tokens = min(max_tokens, random.randint(25, 80))
        latency = 0.05 + random.uniform(0.01, 0.04)

        # Generate realistic diagnostic answers
        if "re-ground" in prompt.lower() or "bounding" in prompt.lower():
            response = '{"action": "reground", "box": [120, 85, 420, 310], "confidence": 0.88, "rationale": "Re-anchored to primary target table cell."}'
        elif "re-plan" in prompt.lower():
            response = '{"action": "replan", "subgoals": ["Parse header", "Locate row 4", "Sum column B"], "confidence": 0.82}'
        elif "re-execute" in prompt.lower() or "tool" in prompt.lower():
            response = '{"action": "reexecute", "extracted_value": "42.5%", "confidence": 0.91, "tool": "ocr_crop"}'
        elif "verify" in prompt.lower():
            response = '{"action": "verify", "layout_loss": 0.04, "consistency": 0.94, "is_valid": true}'
        elif "abstain" in prompt.lower():
            response = '{"action": "abstain", "confidence": 0.22, "rationale": "High ambiguity, safe abstention triggered."}'
        else:
            response = '{"answer": "42.5%", "confidence": 0.85, "reasoning": "Direct extraction from document visual grounding."}'

        return response, prompt_tokens, completion_tokens, latency
