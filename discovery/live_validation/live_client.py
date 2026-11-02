"""
Dedicated Live VLM Client for Empirical Validation.
CRITICAL RESEARCH INTEGRITY RULE: MockClient fallback is STRICTLY PROHIBITED in live mode.
Integrates conservative request-level BudgetGuard with pre-flight estimation,
strict retry accounting, and hard expenditure caps.
"""

import time
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from gvoc_engine.config import MODELS
from gvoc_engine.core.budget_guard import BudgetGuard, BudgetExceededError


class LiveAPIError(Exception):
    """Raised when live API call fails completely across all keys and retries."""
    pass


class LiveVLMClient:
    """
    Manages live API calls with key rotation, exponential backoff,
    pre-request budget enforcement, and conservative usage accounting.
    """

    def __init__(
        self,
        provider: str = "mistral",
        budget_guard: Optional[BudgetGuard] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        client_factory: Optional[Callable[[], Any]] = None
    ):
        self.provider = provider.lower()
        if self.provider == "mistral":
            self.model_name = "pixtral-12b-2409"
            self.base_url = "https://api.mistral.ai/v1"
            self.api_keys = MODELS.get("mistral", {}).get("api_keys", [])
        elif self.provider in ["openrouter", "qwen"]:
            self.model_name = "qwen/qwen2.5-vl-72b-instruct"
            self.base_url = "https://openrouter.ai/api/v1"
            self.api_keys = MODELS.get("openrouter", {}).get("api_keys", [])
        else:
            raise ValueError(f"Unsupported live provider: {provider}")

        # Allow tests to inject client_factory without real keys
        self._client_factory = client_factory
        if not self._client_factory and not self.api_keys:
            raise LiveAPIError(f"No API keys available for provider {self.provider}")

        self.current_key_idx = 0
        self.timeout = timeout
        self.max_retries = max_retries
        self.budget_guard = budget_guard if budget_guard is not None else BudgetGuard()

    def _rotate_key(self, reason: str = "Rate limit / 429"):
        if self.api_keys and len(self.api_keys) > 1:
            prev = self.current_key_idx
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            print(f"[Key Rotation] {self.provider.upper()}: Key #{prev+1} -> Key #{self.current_key_idx+1} ({reason})")

    def _get_client(self) -> Any:
        if self._client_factory:
            return self._client_factory()
        if OpenAI is None:
            raise LiveAPIError("openai package is required for live API calls. Install with: pip install openai")
        key = self.api_keys[self.current_key_idx % len(self.api_keys)]
        return OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout)

    def call(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_tokens: int = 150,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes a multimodal VLM API call under strict budget guardrails.
        Returns detailed provenance dictionary including token counts, latency, and status.
        Raises BudgetExceededError if budget guard refuses the request.
        Raises LiveAPIError if the call cannot be completed.
        """
        # Enforce conservative per-request output limit
        if max_tokens > 250:
            raise ValueError(f"Requested max_tokens ({max_tokens}) exceeds conservative safety limit (250)")

        # Enforce conservative prompt length limit
        if len(prompt) > 4000:
            raise ValueError(f"Requested prompt length ({len(prompt)}) exceeds conservative safety limit (4000)")

        content_items = [{"type": "text", "text": prompt}]
        if image_path:
            p = Path(image_path)
            if not p.exists():
                raise FileNotFoundError(f"Image not found at {image_path}")
            if p.stat().st_size > 10 * 1024 * 1024:
                raise ValueError(f"Image file size ({p.stat().st_size} bytes) exceeds 10MB safety limit")
            with open(p, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            ext = p.suffix.lower().replace(".", "")
            mime = "image/png" if ext == "png" else "image/jpeg"
            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64_data}"}
            })

        messages = [{"role": "user", "content": content_items}]
        backoff = 1.0
        last_error = None

        for attempt in range(self.max_retries):
            # Pre-flight check before EVERY attempt (including retries)
            allowed, refusal_reason = self.budget_guard.check_pre_flight(
                has_image=(image_path is not None),
                max_tokens=max_tokens,
                prompt_char_len=len(prompt)
            )
            if not allowed:
                raise BudgetExceededError(refusal_reason)

            t0 = time.time()
            client = self._get_client()
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                latency = time.time() - t0
                raw_text = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)

                # Record actual usage in budget guard
                p_tok = getattr(usage, "prompt_tokens", None) if usage else None
                c_tok = getattr(usage, "completion_tokens", None) if usage else None
                t_tok = getattr(usage, "total_tokens", None) if usage else None

                self.budget_guard.record_attempt(
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=t_tok,
                    is_error=False,
                    is_retry=(attempt > 0)
                )

                return {
                    "status": "success",
                    "text": raw_text.strip(),
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "total_tokens": t_tok,
                    "latency_sec": latency,
                    "model_id": self.model_name,
                    "provider": self.provider,
                    "attempt": attempt + 1,
                    "is_live_call": True,
                    "budget_summary": self.budget_guard.get_summary()
                }
            except Exception as e:
                # Record failed attempt conservatively against the budget!
                self.budget_guard.record_attempt(is_error=True, is_retry=(attempt > 0))

                err_str = str(e).lower()
                last_error = e
                is_rate_limit = any(t in err_str for t in ["429", "rate limit", "quota", "exhausted"])
                self._rotate_key(reason=type(e).__name__)
                if attempt < self.max_retries - 1:
                    if is_rate_limit:
                        time.sleep(backoff)
                        backoff = min(backoff * 2.0, 8.0)
                    else:
                        time.sleep(0.5)

        raise LiveAPIError(f"Failed to execute live call for {self.model_name} after {self.max_retries} attempts: {last_error}")

