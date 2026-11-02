"""
Global Configuration & Secure Credential Resolver for GVoC Engine.

Resolves API keys securely from system environment or external secured path
without duplicating or exposing keys in the repository.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = os.getenv("GVOC_DB_PATH", str(BASE_DIR / "gvoc_evaluation.db"))

# External secure credential path (configurable via environment variable)
SECURE_TOKENS_PATH = Path(os.getenv("GVOC_ENV_PATH", ".env"))


def load_secure_credentials():
    """
    Dynamically loads API keys into os.environ from external secure directory
    if they are not already set in the current process environment.
    """
    try:
        if SECURE_TOKENS_PATH.is_file():
            with open(SECURE_TOKENS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        if k and not os.environ.get(k):
                            os.environ[k] = v
    except Exception:
        pass


# Initialize secure credentials at import
load_secure_credentials()


def get_api_keys(prefix: str) -> List[str]:
    """Retrieves all rotated keys for a provider (e.g. GROQ_API_KEY, GROQ_API_KEY_2)."""
    keys = []
    base_key = os.environ.get(prefix)
    if base_key:
        keys.append(base_key)
    
    # Check suffixes _2, _3, etc.
    for i in range(2, 10):
        rotated = os.environ.get(f"{prefix}_{i}")
        if rotated:
            keys.append(rotated)
    return keys


# Active Model Configurations
MODELS = {
    "gemini": {
        "provider": "google",
        "model_name": "gemini-3.6-flash",
        "api_keys": get_api_keys("GOOGLE_API_KEY") or get_api_keys("GEMINI_API_KEY")
    },
    "google": {
        "provider": "google",
        "model_name": "gemini-3.6-flash",
        "api_keys": get_api_keys("GOOGLE_API_KEY") or get_api_keys("GEMINI_API_KEY")
    },
    "groq": {
        "provider": "groq",
        "model_name": "llama-3.2-11b-vision-preview",
        "api_keys": get_api_keys("GROQ_API_KEY")
    },
    "openrouter": {
        "provider": "openrouter",
        "model_name": "qwen/qwen-2-vl-7b-instruct:free",
        "api_keys": get_api_keys("OPENROUTER_API_KEY")
    },
    "mistral": {
        "provider": "mistral",
        "model_name": "pixtral-12b-2409",
        "api_keys": get_api_keys("MISTRAL_API_KEY")
    },
    "qwen": {
        "provider": "groq",
        "model_name": "qwen/qwen3.8-27b",
        "api_keys": get_api_keys("GROQ_API_KEY")
    },
    "gpt_oss": {
        "provider": "groq",
        "model_name": "openai/gpt-oss-120b",
        "api_keys": get_api_keys("GROQ_API_KEY")
    }
}

# Value-of-Compute & Safety Utility Parameters
DEFAULT_UTILITY_WEIGHTS = {
    "lambda_R": 1.0,   # Marginal reliability gain weight
    "lambda_C": 0.05,  # Compute cost penalty
    "lambda_S": 0.30,  # Safe abstention bonus
    "tau_safe": 0.35,  # Correctness threshold below which safe abstention triggers
    "epsilon": 1e-4    # Denominator stabilizer for Value-of-Compute ratio
}

# Standard test-time compute budgets (in tokens)
COMPUTE_BUDGET_BANDS = [500, 1000, 2000, 4000]
