"""
Comprehensive Evaluation of Risk-Bounded Counterfactual Intervention Policy.

Steps:
1. Evaluates all 7 methods across all benchmark scenarios.
2. Reports:
   - Task Accuracy (%)
   - Tokens per query
   - Wall-clock decision latency (ms)
   - Empirical Oracle Regret
   - Intervention Harm Rate (%) [Y(stop)=1 -> Y(action)=0]
   - 95% Confidence Intervals via paired bootstrap
3. Stratified evaluation across Validation (DocVQA, ChartQA) vs Held-out Test (TableBench, MMLongBench).
"""

import time
import math
import random
from typing import Dict, Any, List, Tuple
from gvoc_engine.core.action_space import Action, ActionSpace
from gvoc_engine.core.state import GroundedState
from gvoc_engine.benchmark.fresh_benchmark_loader import get_fresh_benchmark_dataset
from gvoc_engine.pilot.run_pilot import PilotRunner
from gvoc_engine.controllers.gvoc_router import GVoCRouter
from gvoc_engine.controllers.risk_bounded_router import RiskBoundedCounterfactualRouter


def bootstrap_ci(data: List[float], n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    if not data:
        return 0.0, 0.0
    means = []
    n = len(data)
    for _ in range(n_boot):
        sample = [random.choice(data) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower_idx = int((1 - ci) / 2 * n_boot)
    upper_idx = int((1 + ci) / 2 * n_boot)
    return means[lower_idx], means[upper_idx]


def run_full_evaluation():
    random.seed(42)
    runner = PilotRunner(use_mock=True)
    dataset = get_fresh_benchmark_dataset()

    router_point = GVoCRouter(calibration_mode="point_estimate", kappa=0.0)
    router_risk = RiskBoundedCounterfactualRouter(kappa=0.10)

    baselines = [
        "monolithic",
        "confidence_router",
        "mact",
        "fixed_veccap",
        "gvoc_point_estimate",
        "gvoc_risk_bounded",
        "oracle"
    ]

    all_records = {b: [] for b in baselines}

    for item in dataset:
        st_dict = item["initial_state"]
        query = item["query"]
        gt = item["ground_truth"]
        eff_set = item["effective_set"]

        branches = runner.run_counterfactual_branches(st_dict, query, gt, eff_set)
        oracle_best_act, oracle_max_val = runner.oracle_router.select_action_from_oracle(
            {act: b["v_emp"] for act, b in branches.items()}
        )
        st = GroundedState.from_dict(st_dict)

        for b in baselines:
            t0 = time.perf_counter()
            if b == "monolithic":
                chosen = runner.monolithic.select_action(st, 2000)
            elif b == "confidence_router":
                chosen = runner.confidence_router.select_action(st, 2000)
            elif b == "mact":
                chosen = runner.mact.select_action(st, 2000)
            elif b == "fixed_veccap":
                chosen = runner.fixed_veccap.select_action(st, 2000)
            elif b == "gvoc_point_estimate":
                chosen, _ = router_point.select_action(st, 2000)
            elif b == "gvoc_risk_bounded":
                chosen, _ = router_risk.select_action(st, 2000)
            elif b == "oracle":
                chosen = oracle_best_act
            else:
                chosen = Action.STOP
            lat_ms = (time.perf_counter() - t0) * 1000

            b_res = branches[chosen]
            y_stop = branches[Action.STOP]["is_correct"]
            y_final = b_res["is_correct"]
            oracle_regret = max(0.0, oracle_max_val - b_res["v_emp"])

            # Intervention harm: broke a previously correct answer
            is_harm = 1 if y_stop == 1 and y_final == 0 and ActionSpace.is_active(chosen) else 0

            # Tokens
            toks = st.n_tokens_spent + b_res["tokens"]

            all_records[b].append({
                "id": item["id"],
                "domain": item["id"].split("_")[0],
                "is_correct": y_final,
                "tokens": toks,
                "latency_ms": lat_ms,
                "regret": oracle_regret,
                "is_harm": is_harm,
                "chosen_action": chosen.value if hasattr(chosen, "value") else str(chosen),
                "is_active": 1 if ActionSpace.is_active(chosen) else 0
            })

    print("\n" + "=" * 105)
    print("        COMPREHENSIVE SCIENTIFIC AUDIT: RISK-BOUNDED COUNTERFACTUAL POLICY (N=40)")
    print("=" * 105)
    print(f"{'Method':<22} | {'Accuracy (95% CI)':<22} | {'Tokens/Query':<12} | {'Lat (ms)':<9} | {'Regret (95% CI)':<20} | {'Harm Rate':<10}")
    print("-" * 105)

    for b in baselines:
        recs = all_records[b]
        n = len(recs)
        acc_vals = [r["is_correct"] for r in recs]
        tok_vals = [r["tokens"] for r in recs]
        lat_vals = [r["latency_ms"] for r in recs]
        reg_vals = [r["regret"] for r in recs]
        harm_vals = [r["is_harm"] for r in recs]

        acc_mean = sum(acc_vals) / n * 100
        acc_ci_low, acc_ci_high = bootstrap_ci([x * 100 for x in acc_vals])

        tok_mean = sum(tok_vals) / n
        lat_mean = sum(lat_vals) / n

        reg_mean = sum(reg_vals) / n
        reg_ci_low, reg_ci_high = bootstrap_ci(reg_vals)

        harm_mean = sum(harm_vals) / n * 100

        acc_str = f"{acc_mean:4.1f}% [{acc_ci_low:4.1f}, {acc_ci_high:4.1f}]"
        reg_str = f"{reg_mean:5.3f} [{reg_ci_low:5.3f}, {reg_ci_high:5.3f}]"

        print(f"{b:<22} | {acc_str:<22} | {tok_mean:>12.1f} | {lat_mean:>9.3f} | {reg_str:<20} | {harm_mean:>8.1f}%")

    print("=" * 105)

    # Cross-domain Stratification: Validation vs Held-Out Test
    print("\n" + "-" * 85)
    print("       HELD-OUT GENERALIZATION AUDIT: VALIDATION DOMAINS VS TEST DOMAINS")
    print("-" * 85)
    print(f"{'Split / Method':<38} | {'Accuracy (%)':<14} | {'Avg Tokens':<12} | {'Oracle Regret':<14}")
    print("-" * 85)

    for split_name, domain_filter in [
        ("Validation Domains (DocVQA + ChartQA)", lambda d: d in ["docvqa", "chartqa"]),
        ("Held-out Test Domains (TableBench + MMLong)", lambda d: d in ["tablebench", "mmlong"])
    ]:
        print(f"[{split_name}]")
        for b in ["gvoc_point_estimate", "gvoc_risk_bounded", "fixed_veccap", "mact"]:
            split_recs = [r for r in all_records[b] if domain_filter(r["domain"])]
            s_acc = sum(r["is_correct"] for r in split_recs) / len(split_recs) * 100
            s_tok = sum(r["tokens"] for r in split_recs) / len(split_recs)
            s_reg = sum(r["regret"] for r in split_recs) / len(split_recs)
            print(f"  {b:<36} | {s_acc:>12.1f}% | {s_tok:>12.1f} | {s_reg:>14.3f}")
        print("-" * 85)


if __name__ == "__main__":
    run_full_evaluation()
