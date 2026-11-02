"""
Conservative Request-Level Budget Guard for VLM Evaluations.
Enforces hard monetary caps, cumulative token limits, pre-request cost estimation,
strict retry accounting, and granular request state tracking without reliance on optimistic assumptions.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple


class BudgetExceededError(Exception):
    """Raised when an API request is refused because it would breach the budget guard."""
    pass


@dataclass
class BudgetState:
    planned_requests: int = 0
    attempted_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    blocked_requests: int = 0
    cumulative_requests: int = 0  # Mirrors attempted_requests
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0
    cumulative_total_tokens: int = 0
    cumulative_cost_usd: float = 0.0
    refused_requests: int = 0     # Mirrors blocked_requests
    # Distinct token accounting breakdown
    reported_tokens: int = 0
    conservatively_charged_tokens: int = 0
    reported_cost_usd: float = 0.0
    conservatively_charged_cost_usd: float = 0.0
    operational_limit_reached: bool = False
    overage_anomalies: list = field(default_factory=list)


class BudgetGuard:
    """
    Guards API invocations against cost and token overruns.
    
    Hard Limits:
        max_budget_usd: $0.01500 USD
        max_total_tokens: 75,000 tokens
        
    Operational Limits:
        operational_budget_usd: $0.01000 USD
        operational_token_limit: 60,000 tokens
        
    Pricing Model (Mistral Pixtral-12B):
        $0.15 per 1,000,000 tokens ($0.00015 per 1k tokens) for both input and output.
    """

    def __init__(
        self,
        max_budget_usd: float = 0.015,
        max_total_tokens: int = 75000,
        operational_budget_usd: float = 0.010,
        operational_token_limit: int = 60000,
        price_per_million_tokens: float = 0.15,
        default_missing_prompt_tokens: int = 2800,
        default_missing_completion_tokens: int = 60,
        planned_requests: int = 0,
        max_consecutive_missing_metadata: int = 3,
    ):
        self.max_budget_usd = max_budget_usd
        self.max_total_tokens = max_total_tokens
        self.operational_budget_usd = operational_budget_usd
        self.operational_token_limit = operational_token_limit
        self.price_per_million_tokens = price_per_million_tokens
        self.default_missing_prompt_tokens = default_missing_prompt_tokens
        self.default_missing_completion_tokens = default_missing_completion_tokens
        self.max_consecutive_missing_metadata = max_consecutive_missing_metadata
        self.consecutive_missing_metadata = 0

        self.state = BudgetState(planned_requests=planned_requests)

    def estimate_request_cost(
        self,
        has_image: bool = True,
        max_tokens: int = 60,
        prompt_char_len: int = 200
    ) -> Tuple[int, float]:
        """
        Estimates the conservative maximum token count and monetary cost of a pending request.
        Guarantees that the reservation covers the worst-case scenario:
        - Image input tokens: ~2,500 tokens (high-res patch tokenization) when has_image is True.
        - Conservative prompt fallback floor: default_missing_prompt_tokens (2,800) for image calls.
        - Text prompt overhead: ~prompt_char_len / 3.0 tokens.
        - Output tokens: max_tokens.
        """
        if has_image:
            image_tokens = 2500
            prompt_floor = self.default_missing_prompt_tokens
        else:
            image_tokens = 0
            prompt_floor = 50

        text_tokens = max(50, int(prompt_char_len / 3.0))
        prompt_component = image_tokens + text_tokens
        est_prompt_tokens = max(prompt_floor, prompt_component)
        est_completion_tokens = max_tokens
        est_total_tokens = est_prompt_tokens + est_completion_tokens
        est_cost_usd = (est_total_tokens / 1000000.0) * self.price_per_million_tokens
        return est_total_tokens, est_cost_usd

    def check_pre_flight(
        self,
        has_image: bool = True,
        max_tokens: int = 60,
        prompt_char_len: int = 200
    ) -> Tuple[bool, Optional[str]]:
        """
        Pre-flight check: Verifies if the next request can be safely submitted.
        Refuses if the estimated upper-bound reservation would breach operational or hard limits.
        Equality Policy: Exactly reaching the ceiling is permitted; exceeding by >= 1 token or $0.00001 is refused.
        """
        est_tokens, est_cost = self.estimate_request_cost(
            has_image=has_image,
            max_tokens=max_tokens,
            prompt_char_len=prompt_char_len
        )

        projected_tokens = self.state.cumulative_total_tokens + est_tokens
        projected_cost = self.state.cumulative_cost_usd + est_cost

        # 1. Hard ceiling check: strictly fail-closed if projected or cumulative exceeds max
        if projected_cost > self.max_budget_usd or self.state.cumulative_cost_usd > self.max_budget_usd:
            self.state.blocked_requests += 1
            self.state.refused_requests += 1
            return False, (
                f"BudgetGuard Refusal: Projected cost ${projected_cost:.5f} exceeds "
                f"hard limit ${self.max_budget_usd:.5f}"
            )

        if projected_tokens > self.max_total_tokens or self.state.cumulative_total_tokens > self.max_total_tokens:
            self.state.blocked_requests += 1
            self.state.refused_requests += 1
            return False, (
                f"BudgetGuard Refusal: Projected tokens {projected_tokens} exceed "
                f"hard limit {self.max_total_tokens}"
            )

        # 2. Operational ceiling check:
        # Check if operational ceiling was already reached or breached by a previous request
        if (self.state.cumulative_total_tokens >= self.operational_token_limit or
                self.state.cumulative_cost_usd >= self.operational_budget_usd):
            self.state.blocked_requests += 1
            self.state.refused_requests += 1
            return False, (
                f"BudgetGuard Operational Halt: Operational ceiling already reached "
                f"(tokens: {self.state.cumulative_total_tokens} >= {self.operational_token_limit}, "
                f"cost: ${self.state.cumulative_cost_usd:.5f} >= ${self.operational_budget_usd:.5f})"
            )

        # Allow equality, strictly refuse > ceiling
        if projected_cost > self.operational_budget_usd:
            self.state.blocked_requests += 1
            self.state.refused_requests += 1
            return False, (
                f"BudgetGuard Operational Halt: Projected cost ${projected_cost:.5f} exceeds "
                f"operational threshold ${self.operational_budget_usd:.5f} (reserve active)"
            )

        if projected_tokens > self.operational_token_limit:
            self.state.blocked_requests += 1
            self.state.refused_requests += 1
            return False, (
                f"BudgetGuard Operational Halt: Projected tokens {projected_tokens} exceed "
                f"operational threshold {self.operational_token_limit} (reserve active)"
            )

        return True, None

    def record_attempt(
        self,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        is_error: bool = False,
        is_retry: bool = False
    ) -> Dict[str, Any]:
        """
        Records an actual HTTP attempt (successful or failed) and updates cumulative state.
        Conservatively handles missing provider metadata and tracks reported vs charged breakdown.
        """
        self.state.attempted_requests += 1
        self.state.cumulative_requests += 1

        if is_retry:
            self.state.retried_requests += 1

        if is_error:
            self.state.failed_requests += 1
        else:
            self.state.successful_requests += 1

        # Fallback for missing usage metadata
        if prompt_tokens is None and not is_error:
            self.consecutive_missing_metadata += 1
            if self.consecutive_missing_metadata >= self.max_consecutive_missing_metadata:
                raise BudgetExceededError(
                    f"BudgetGuard Refusal: Usage metadata missing repeatedly "
                    f"({self.consecutive_missing_metadata} consecutive requests). "
                    f"Aborting run cleanly for provider accounting safety."
                )
        elif not is_error:
            self.consecutive_missing_metadata = 0

        # Ledger breakdown: reported vs conservatively charged
        if prompt_tokens is not None and completion_tokens is not None and not is_error:
            p_tok = prompt_tokens
            c_tok = completion_tokens
            t_tok = total_tokens if total_tokens is not None else (p_tok + c_tok)
            cost_usd = (t_tok / 1000000.0) * self.price_per_million_tokens
            self.state.reported_tokens += t_tok
            self.state.reported_cost_usd += cost_usd
        else:
            p_tok = prompt_tokens if prompt_tokens is not None else self.default_missing_prompt_tokens
            c_tok = completion_tokens if completion_tokens is not None else (
                self.default_missing_completion_tokens if not is_error else 0
            )
            t_tok = total_tokens if total_tokens is not None else (p_tok + c_tok)
            cost_usd = (t_tok / 1000000.0) * self.price_per_million_tokens
            self.state.conservatively_charged_tokens += t_tok
            self.state.conservatively_charged_cost_usd += cost_usd

        self.state.cumulative_prompt_tokens += p_tok
        self.state.cumulative_completion_tokens += c_tok
        self.state.cumulative_total_tokens += t_tok
        self.state.cumulative_cost_usd += cost_usd

        # Reconcile operational limit reached flag
        if (self.state.cumulative_total_tokens >= self.operational_token_limit or
                self.state.cumulative_cost_usd >= self.operational_budget_usd):
            self.state.operational_limit_reached = True

        # Check for unexpected overage anomaly
        max_expected_single = self.default_missing_prompt_tokens + self.default_missing_completion_tokens
        if t_tok > max_expected_single:
            self.state.overage_anomalies.append({
                "request_index": self.state.attempted_requests,
                "actual_tokens": t_tok,
                "cost_usd": cost_usd,
                "message": f"Actual tokens {t_tok} exceeded conservative ceiling ({max_expected_single})."
            })

        return {
            "attempted_requests": self.state.attempted_requests,
            "cumulative_requests": self.state.cumulative_requests,
            "successful_requests": self.state.successful_requests,
            "failed_requests": self.state.failed_requests,
            "retried_requests": self.state.retried_requests,
            "blocked_requests": self.state.blocked_requests,
            "cumulative_total_tokens": self.state.cumulative_total_tokens,
            "cumulative_cost_usd": round(self.state.cumulative_cost_usd, 6),
            "reported_tokens": self.state.reported_tokens,
            "conservatively_charged_tokens": self.state.conservatively_charged_tokens,
            "operational_limit_reached": self.state.operational_limit_reached,
            "remaining_budget_usd": round(max(0.0, self.max_budget_usd - self.state.cumulative_cost_usd), 6),
            "remaining_tokens": max(0, self.max_total_tokens - self.state.cumulative_total_tokens),
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "planned_requests": self.state.planned_requests,
            "attempted_requests": self.state.attempted_requests,
            "successful_requests": self.state.successful_requests,
            "failed_requests": self.state.failed_requests,
            "retried_requests": self.state.retried_requests,
            "blocked_requests": self.state.blocked_requests,
            "cumulative_requests": self.state.cumulative_requests,
            "cumulative_prompt_tokens": self.state.cumulative_prompt_tokens,
            "cumulative_completion_tokens": self.state.cumulative_completion_tokens,
            "cumulative_total_tokens": self.state.cumulative_total_tokens,
            "cumulative_cost_usd": round(self.state.cumulative_cost_usd, 6),
            "refused_requests": self.state.refused_requests,
            "hard_limit_usd": self.max_budget_usd,
            "hard_token_limit": self.max_total_tokens,
            "operational_limit_usd": self.operational_budget_usd,
            "operational_token_limit": self.operational_token_limit,
            "reported_tokens": self.state.reported_tokens,
            "conservatively_charged_tokens": self.state.conservatively_charged_tokens,
            "reported_cost_usd": round(self.state.reported_cost_usd, 6),
            "conservatively_charged_cost_usd": round(self.state.conservatively_charged_cost_usd, 6),
            "operational_limit_reached": self.state.operational_limit_reached,
            "overage_anomalies": self.state.overage_anomalies,
        }

