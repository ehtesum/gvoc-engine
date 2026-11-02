"""
Unified Multi-Provider API Client with Key Rotation and Rate Limiting.
"""

import time
import json
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from gvoc_engine.config import MODELS
from gvoc_engine.providers.mock_client import MockClient


class UnifiedClientManager:
    """Manages API calls across Google, Groq, and OpenRouter with key rotation."""

    def __init__(self, provider: str = "groq", use_mock: bool = False):
        self.provider = provider.lower()
        self.use_mock = use_mock
        self.mock_client = MockClient()
        
        self.cfg = MODELS.get(self.provider, MODELS["groq"])
        self.model_name = self.cfg["model_name"]
        self.api_keys = self.cfg["api_keys"]
        self.current_key_idx = 0

        # Endpoint URLs
        self.base_urls = {
            "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "groq": "https://api.groq.com/openai/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "mistral": "https://api.mistral.ai/v1"
        }

    def _get_active_client(self) -> Optional[Any]:
        """Returns an OpenAI client initialized with the current rotated key."""
        if not self.api_keys or OpenAI is None:
            return None
        key = self.api_keys[self.current_key_idx % len(self.api_keys)]
        engine = self.cfg.get("provider", self.provider)
        base_url = self.base_urls.get(engine, self.base_urls["groq"])
        return OpenAI(api_key=key, base_url=base_url)

    def _rotate_key(self, reason: str = "Rate limit / 429"):
        """Rotates to the next available API key in the pool and logs notification."""
        if len(self.api_keys) > 1:
            prev_idx = self.current_key_idx
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            print(f"\n[Revolving API] {self.provider.upper()} Key #{prev_idx + 1} -> Rotated to Key #{self.current_key_idx + 1}/{len(self.api_keys)} | Cause: {reason}")

    def call_vlm(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0
    ) -> Tuple[str, int, int, float]:
        """
        Executes a multimodal VLM call with key rotation and exponential backoff.
        Returns: (text_response, prompt_tokens, completion_tokens, latency_sec)
        """
        if self.use_mock or not self.api_keys or OpenAI is None:
            return self.mock_client.call_vlm(prompt, image_path, model=self.model_name, max_tokens=max_tokens)

        # Prepare messages
        content_items = [{"type": "text", "text": prompt}]
        if image_path and Path(image_path).exists():
            try:
                with open(image_path, "rb") as img_file:
                    b64_data = base64.b64encode(img_file.read()).decode("utf-8")
                    content_items.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
                    })
            except Exception as e:
                print(f"[Client Warning] Image encoding failed: {e}")

        messages = [{"role": "user", "content": content_items}]

        # Retry loop with key rotation
        max_retries = max(4, len(self.api_keys) * 3)
        backoff = 1.0

        for attempt in range(max_retries):
            t0 = time.time()
            client = self._get_active_client()
            if not client:
                return self.mock_client.call_vlm(prompt, image_path, model=self.model_name, max_tokens=max_tokens)

            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                latency = time.time() - t0
                text = response.choices[0].message.content or ""
                p_tok = response.usage.prompt_tokens if response.usage else len(prompt.split()) + 50
                c_tok = response.usage.completion_tokens if response.usage else len(text.split())
                return text, p_tok, c_tok, latency

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(term in err_str for term in ["429", "rate limit", "quota", "exhausted", "resource", "limit", "capacity", "503", "timeout"])
                
                # Immediately revolve to next key
                self._rotate_key(reason=type(e).__name__)

                if is_rate_limit:
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 6.0)
                else:
                    time.sleep(0.5)

        # Fallback to mock if API exhaustion across all rotated keys
        return self.mock_client.call_vlm(prompt, image_path, model=self.model_name, max_tokens=max_tokens)
