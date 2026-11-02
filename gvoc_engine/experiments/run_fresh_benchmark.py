"""
Comprehensive Multi-Domain Benchmark Runner for GVoC Engine.

Runs head-to-head evaluations across:
1. Monolithic VLM
2. Fixed VEC-CAP (Prior Deterministic Rule Baseline)
3. MACT (Adaptive Agent-Wise Scaling)
4. Confidence-Only Router
5. GVoC (Learned Value-of-Compute Router)
6. Oracle GVoC (Empirical Upper Bound)

Spanning: DocVQA, ChartQA, TableBench, and MMLongBench-Doc (40 instances).
"""

import sys
import argparse
import time
from typing import Dict, Any, List
from collections import Counter

from gvoc_engine.core.action_space import Action
from gvoc_engine.benchmark.fresh_benchmark_loader import get_fresh_benchmark_dataset
from gvoc_engine.pilot.run_pilot import PilotRunner
from gvoc_engine.pilot.plot_pilot import plot_pilot_results


def run_fresh_benchmark(
    mode: str = "sim",
    provider: str = "groq",
    target_dataset: str = "all"
):
    """Executes the full fresh diagnostic benchmark suite."""
    use_mock = (mode.lower() == "sim")
    runner = PilotRunner(provider=provider, use_mock=use_mock)

    # 1. Load fresh benchmark instances
    all_instances = get_fresh_benchmark_dataset()
    if target_dataset.lower() != "all":
        instances = [x for x in all_instances if x["dataset"].lower() == target_dataset.lower()]
    else:
        instances = all_instances

    baselines = ["monolithic", "fixed_veccap", "mact", "confidence_router", "gvoc", "oracle"]
    
    print("\n" + "="*85)
    print(f"       GVoC FRESH BENCHMARK RUNNER (ESWA EVALUATION MATRIX)")
    print(f"       Mode: {'OFFLINE SIMULATION (Zero Cost)' if use_mock else f'LIVE API ({provider.upper()})'}")
    print(f"       Total Benchmark Instances: {len(instances)} (Domains: DocVQA, ChartQA, TableBench, MMLongBench)")
    print("="*85 + "\n")

    t_start = time.time()
    results_by_baseline: Dict[str, List[Dict[str, Any]]] = {b: [] for b in baselines}

    for idx, item in enumerate(instances, 1):
        domain = item["dataset"].upper()
        inst_id = item["id"]
        failure = item["failure_type"]
        sys.stdout.write(f"\r[{idx:02d}/{len(instances):02d}] Evaluating [{domain}] {inst_id:<15} (Defect: {failure:<24})...")
        sys.stdout.flush()

        for b in baselines:
            res = runner.evaluate_instance(item, b)
            results_by_baseline[b].append(res)

    print("\n\n" + "="*85)
    print("                 OVERALL HEAD-TO-HEAD BENCHMARK RESULTS")
    print("="*85)
    print(f"{'Method':<20} | {'Accuracy':<10} | {'ISA':<8} | {'Oracle Regret':<14} | {'RSR':<8} | {'UCR':<8} | {'Avg Tokens':<10}")
    print("-" * 85)

    for b in baselines:
        items = results_by_baseline[b]
        n = len(items)
        acc = sum(x["is_correct"] for x in items) / n
        isa = sum(x["is_in_effective_set"] for x in items) / n
        regret = sum(x["oracle_regret"] for x in items) / n
        recoverable = sum(1 for x in items if x["chosen_action"] != "stop")
        rsr = sum(x["recovery_success"] for x in items) / max(1, recoverable)
        ucr = sum(x["unnecessary_compute"] for x in items) / n
        avg_tok = sum(x["total_tokens"] for x in items) / n

        print(f"{b:<20} | {acc*100:>8.1f}% | {isa*100:>6.1f}% | {regret:>14.3f} | {rsr*100:>6.1f}% | {ucr*100:>6.1f}% | {avg_tok:>10.0f}")
    print("="*85)

    # 2. Breakdown by Dataset Domain
    datasets = sorted(list(set(x["dataset"] for x in instances)))
    print("\n" + "-"*85)
    print("              DOMAIN-SPECIFIC BREAKDOWN (ACCURACY % / ORACLE REGRET)")
    print("-"*85)
    print(f"{'Domain':<16} | {'Monolithic':<12} | {'Fixed VEC-CAP':<14} | {'MACT':<12} | {'GVoC (Ours)':<14} | {'Oracle':<10}")
    print("-" * 85)

    for d in datasets:
        d_items = [i for i, x in enumerate(instances) if x["dataset"] == d]
        row_str = f"{d.upper():<16} | "
        for b in ["monolithic", "fixed_veccap", "mact", "gvoc", "oracle"]:
            b_d_items = [results_by_baseline[b][i] for i in d_items]
            d_acc = sum(x["is_correct"] for x in b_d_items) / len(b_d_items) * 100
            d_reg = sum(x["oracle_regret"] for x in b_d_items) / len(b_d_items)
            row_str += f"{d_acc:>4.0f}% ({d_reg:.1f}) | "
        print(row_str)
    print("-" * 85)

    # 3. Action Distribution Entropy for GVoC
    gvoc_actions = [x["chosen_action"] for x in results_by_baseline["gvoc"]]
    action_dist = Counter(gvoc_actions)
    print("\n[GVoC Action Diversity & Diagnostic Entropy]")
    for act, count in action_dist.items():
        print(f"  P(a* = {act:<10}) = {count/len(gvoc_actions)*100:>5.1f}% ({count:02d}/{len(gvoc_actions)})")

    # 4. Update Publication Figures
    print("\n[Generating Publication Figures] Updating Pareto & Regret charts...")
    plot_pilot_results(runner.db.db_path)
    print(f"[Done] Benchmark completed in {time.time() - t_start:.2f}s. Results committed to: {runner.db.db_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fresh GVoC benchmark.")
    parser.add_argument("--mode", type=str, default="sim", choices=["sim", "live"], help="Execution mode (sim or live)")
    parser.add_argument("--provider", type=str, default="mistral", choices=["mistral", "gemini", "google", "groq", "openrouter"], help="Free API provider")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset filter (docvqa, chartqa, tablebench, mmlongbench, all)")
    args = parser.parse_args()

    run_fresh_benchmark(mode=args.mode, provider=args.provider, target_dataset=args.dataset)
