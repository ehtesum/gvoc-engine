# GVoC Engine (Grounded Value-of-Compute)

A modular research testbed for **Cost-Aware Intervention-Level Credit Assignment** in Visual Document Understanding (VDU), designed for **Elsevier Expert Systems with Applications (ESWA, Q1)**.

---

## 1. Directory Structure

```text
gvoc_engine/
├── .gitignore                   # Keeps repository clean (blocks .env, .db, and figures)
├── .env.example                 # Example template for credentials
├── config.py                    # Secure dynamic loader (resolves tokens from C:\Users\user\Desktop\tokens_essential\.env)
├── gvoc_evaluation.db           # Persistent SQLite database logging all runs and metrics
├── providers/
│   ├── client_manager.py        # Multi-key rotated client (Groq, OpenRouter, Google Gemini)
│   └── mock_client.py           # 100% offline simulation client for fast zero-cost dry runs
├── core/
│   ├── action_space.py          # A = {stop, reground, replan, reexec, verify, abstain}
│   ├── state.py                 # 11-dimensional grounded state vector s_t in R^d
│   └── database.py              # SQLite evaluation logger for task and diagnostic metrics
├── interventions/               # Concrete repair modules
│   ├── reground.py              # Spatial bounding-box refinement
│   ├── replan.py                # Query sub-goal decomposition & re-planning
│   ├── reexecute.py             # Tool execution & OCR re-extraction
│   └── verify.py                # Geometric layout loss D_layout & consensus check
├── controllers/                 # Decision policies
│   ├── gvoc_router.py           # Learned Value-of-Compute controller (f_theta & utility)
│   └── baselines.py             # Monolithic, Self-Consistency, Fixed VEC-CAP, MACT, Oracle
├── benchmark/
│   ├── perturbations.py         # Multi-severity shifts (+/-10 to 50px), DAG inversion, OCR noise
│   └── effective_sets.py        # Multi-label Effective Intervention Sets A*
└── pilot/
    ├── sample_data.py           # Multi-domain diagnostic pilot tasks (DocVQA, ChartQA, TableBench)
    ├── run_pilot.py             # Head-to-head pilot evaluation runner
    ├── plot_pilot.py            # Automatically renders publication figures
    └── figures/                 # Generated publication-grade PNG charts
```

---

## 2. Security & API Hygiene

* **Zero Leaked Keys**: API keys are **never copied or committed** into this repository.
* **Automatic Resolution**: `config.py` automatically resolves keys directly from your secured Desktop folder:
  `C:\Users\user\Desktop\tokens_essential\.env`.
* **Multi-Key Rotation**: Rotates through `GROQ_API_KEY_1..3`, `OPENROUTER_API_KEY_1..3`, `GOOGLE_API_KEY` to bypass free-tier rate limits automatically.

---

## 3. Essential Metrics Recorded

Every execution automatically logs to `gvoc_evaluation.db`:

| Metric Category | Metrics Logged | Purpose |
| :--- | :--- | :--- |
| **Tier 1: Task Performance** | Accuracy, ANLS, Coverage, Risk | End-to-end task benchmarking against baselines |
| **Tier 2: Controller Quality** | ISA (vs $A^*$), Oracle Regret ($\mathrm{IR}_{\mathrm{oracle}}$), RSR, UCR | Proves that GVoC's learned actions match the counterfactual optimum |
| **Tier 3: Compute Accounting** | Prompt Tokens, Completion Tokens, Latency (s), Model/Tool Calls | Strict evaluation under matched compute budgets |

---

## 4. How to Run & Track

### Run the Pilot Benchmark (Offline / Simulation Mode)
Runs instant head-to-head evaluation across GVoC and all 5 baselines without spending any API calls:
```bash
python -m gvoc_engine.pilot.run_pilot
```

### Run on Live Free APIs (Groq / Gemini / OpenRouter)
To execute using live multi-key rotated APIs:
```python
from gvoc_engine.pilot.run_pilot import PilotRunner
runner = PilotRunner(provider="groq", use_mock=False) # or provider="gemini"
runner.run_pilot()
```

### Render Publication Figures
Generates the three core empirical figures in `pilot/figures/`:
```bash
python -m gvoc_engine.pilot.plot_pilot
```
1. `fig_pilot_regret_comparison.png` — Oracle Intervention Regret across routing policies.
2. `fig_pilot_pareto_frontier.png` — Accuracy vs. Compute Token Cost Pareto Frontier.
3. `fig_pilot_action_distribution.png` — Action diversity across intermediate states.
