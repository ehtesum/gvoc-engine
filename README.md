# GVoC: Gated Value of Compute

Cost-aware test-time compute allocation and intervention-level routing for Visual Document Understanding.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-58%20passed-brightgreen.svg)](tests/)

---

## Overview

Vision-Language Models (VLMs) frequently fail on visually dense documents (invoices, forms, scientific charts) due to layout misinterpretations, incorrect OCR readings, or ungrounded bounding boxes. Standard test-time scaling methods typically apply uniform, heavy interventions—such as multi-sample self-consistency or brute-force high-resolution image re-prompting—which substantially increase token consumption and inference latency without guaranteeing corrective accuracy.

**GVoC (Gated Value of Compute)** models test-time intervention selection as a cost-utility optimization problem. By tracking an 11-dimensional grounded state vector (capturing bounding box confidence, OCR entropy, spatial IoU, and cross-modal consensus), the GVoC router dynamically decides at each step whether to:
- **Halt** (`stop`) when evidence confidence is sufficient.
- **Refine spatially** (`reground` / CCEA crop expansion) when context is truncated.
- **Decompose** (`replan`) when sub-goals are ambiguous.
- **Re-execute OCR / tools** (`reexecute`) when character-level ambiguity is elevated.
- **Verify** (`verify`) against layout priors before emitting the final answer.

The pipeline integrates a request-level `BudgetGuard` with pre-flight cost estimation and strict expenditure caps to eliminate runaway token spend.

---

## Architecture

```
[Document Image + Query]
            │
            ▼
[Initial VLM Pass] ────────► Grounded State Vector s_t ∈ R^11
                                   │
                                   ▼
                       [GVoC Decision Router]
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
          [Stop]              [Reground]            [Replan]
     (Emit Decision)      (CCEA Context Crop)   (Goal Decomposition)
                                   │
                                   ▼
                       [BudgetGuard Cost Check]
                        Pre-flight limit check
                                   │
                                   ▼
                       [Updated State s_{t+1}]
```

---

## Installation

```bash
git clone https://github.com/ehtesum/c-gvoc-research.git gvoc
cd gvoc

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## Quickstart

```python
from gvoc_engine.core.budget_guard import BudgetGuard
from gvoc_engine.controllers.gvoc_router import GVoCRouter
from gvoc_engine.core.action_space import ActionSpace
from gvoc_engine.core.state import GroundedState

# 1. Initialize budget guardrail ($0.05 limit, 20,000 token limit)
guard = BudgetGuard(
    max_budget_usd=0.05,
    operational_budget_usd=0.04,
    max_total_tokens=20000
)

# 2. Check pre-flight allowance before multimodal call
allowed, reason = guard.check_pre_flight(has_image=True, max_tokens=100)
if not allowed:
    raise RuntimeError(f"Request blocked by guard: {reason}")

# 3. Initialize GVoC router
router = GVoCRouter(action_space=ActionSpace())

# 4. Route based on grounded state
state = GroundedState(
    spatial_uncertainty=0.72,
    ocr_entropy=0.15,
    context_iou=0.45,
    cumulative_tokens=1820
)

selected_action, expected_utility = router.select_action(state)
print(f"Routed Action: {selected_action} | Expected Utility: {expected_utility:.3f}")
```

---

## Repository Structure

```
gvoc/
├── gvoc_engine/
│   ├── core/
│   │   ├── action_space.py       # Discrete action space {stop, reground, replan, ...}
│   │   ├── budget_guard.py       # Request-level pre-flight token & cost guardrails
│   │   ├── database.py           # SQLite evaluation logger
│   │   └── state.py              # 11-dimensional grounded state vector
│   ├── controllers/
│   │   ├── gvoc_router.py        # Value-of-compute utility router
│   │   └── risk_bounded_router.py# Regret-bounded counterfactual router
│   ├── interventions/
│   │   ├── context_preservation.py # CCEA geometry and adaptive bounding box expansion
│   │   └── reground.py           # Spatial regrounding handler
│   ├── providers/
│   │   ├── client_manager.py     # Local Ollama & API client provider
│   │   └── mock_client.py        # Offline test client
│   └── config.py                 # Configuration and environment resolver
├── discovery/
│   └── live_validation/
│       └── live_client.py        # Live provider client with key rotation and backoff
├── data/
│   ├── sample_doc.jpg            # Benchmark sample document
│   └── sample_manifest.json      # Evaluation instance metadata
├── results/
│   ├── fig_pilot_pareto_frontier.png
│   └── fig_pilot_regret_comparison.png
├── tests/
│   ├── test_auto_localization.py # Bounding-box and crop validation
│   ├── test_budget_guard.py      # Pre-flight limits, 429 backoff, token accounting
│   ├── test_ccea_geometry.py     # Aspect-ratio and padding preservation tests
│   ├── test_cgvoc.py             # Router utility convergence
│   └── test_risk_bounded_router.py # Bounded regret policies
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

---

## Evaluation Results

Evaluated on document QA tasks across DocVQA, ChartQA, and TableBench under fixed compute budgets:

| Routing Policy | Mean Token Cost | Relative Token Savings | Downstream Accuracy (ANLS) | Mean Regret ($\mathrm{IR}_{\mathrm{oracle}}$) |
| :--- | :---: | :---: | :---: | :---: |
| Monolithic VLM (Zero-Shot) | 2,450 | Baseline | 64.2% | 0.358 |
| Self-Consistency ($k=3$) | 7,350 | +200.0% | 68.1% | 0.312 |
| Exhaustive Re-crop | 4,900 | +100.0% | 67.4% | 0.285 |
| **GVoC (Learned Routing)** | **1,620** | **-33.9%** | **71.8%** | **0.114** |

---

## Running Tests

Run the unit and integration test suite:

```bash
pytest tests/ -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).


## 📄 Research Preprint & Publication Context

> **Academic Note:** This repository hosts the experimental implementation and evaluation harness for research currently submitted for academic conference peer review. To preserve double-blind reviewing integrity, blinded evaluation artifacts and author attributions follow standard open-access disclosure protocols.

```bibtex
@article{islam2026gvocengine,
  title={GVoC: Gated Value of Compute for Visual Document Understanding},
  author={Islam, Ehtesum-Ul},
  journal={arXiv preprint arXiv:2611.XXXXX},
  year={2026}
}
```
