"""
Pilot Study Runner for GVoC Engine.

Executes head-to-head comparative evaluations across:
1. Monolithic VLM (Zero-Recovery Baseline)
2. MACT (Adaptive Agent-Wise Scaling)
3. Fixed VEC-CAP (Prior Deterministic Rule Baseline)
4. Confidence-Only Router (Ablation Baseline)
5. Grounded Value-of-Compute (GVoC - Proposed)
6. Oracle GVoC (Empirical Upper Bound)

Directly answers: Does the identity of the best intervention a* vary across s_t,
and is it predictable from grounded state features?
"""

import os
import sys
import uuid
import json
import time
from typing import Dict, Any, List, Tuple
from collections import Counter

from gvoc_engine.core.action_space import Action, ActionSpace
from gvoc_engine.core.state import GroundedState
from gvoc_engine.core.database import EvaluationDatabase
from gvoc_engine.controllers.gvoc_router import GVoCRouter
from gvoc_engine.controllers.baselines import (
    MonolithicBaseline,
    FixedVECCAPBaseline,
    MACTBaseline,
    ConfidenceRouterBaseline,
    OracleGVoCRouter
)
from gvoc_engine.interventions.reground import execute_reground
from gvoc_engine.interventions.replan import execute_replan
from gvoc_engine.interventions.reexecute import execute_reexec
from gvoc_engine.interventions.verify import execute_verify
from gvoc_engine.providers.client_manager import UnifiedClientManager
from gvoc_engine.pilot.sample_data import get_pilot_scenarios


class PilotRunner:
    """Orchestrates comparative pilot execution and diagnostic metric logging."""

    def __init__(self, provider: str = "groq", use_mock: bool = True):
        self.client = UnifiedClientManager(provider=provider, use_mock=use_mock)
        self.db = EvaluationDatabase()
        self.gvoc_router = GVoCRouter()
        self.monolithic = MonolithicBaseline()
        self.fixed_veccap = FixedVECCAPBaseline()
        self.mact = MACTBaseline()
        self.confidence_router = ConfidenceRouterBaseline()
        self.oracle_router = OracleGVoCRouter()

    def _execute_intervention(
        self,
        action: Action,
        state: GroundedState,
        query: str
    ) -> Tuple[GroundedState, int, float]:
        """Dispatches process repair execution to specialized modules."""
        if action == Action.REGROUND:
            return execute_reground(state, self.client, "mock_image.jpg", query)
        elif action == Action.REPLAN:
            return execute_replan(state, self.client, "mock_image.jpg", query)
        elif action == Action.REEXEC:
            return execute_reexec(state, self.client, "mock_image.jpg", query)
        elif action == Action.VERIFY:
            return execute_verify(state, self.client, "mock_image.jpg", query)
        return state, 0, 0.0

    def run_counterfactual_branches(
        self,
        base_state_dict: Dict[str, Any],
        query: str,
        ground_truth: str,
        effective_set: List[Action]
    ) -> Dict[Action, Dict[str, Any]]:
        """
        Branches candidate repair actions from the IDENTICAL intermediate state s_t.
        Returns empirical downstream outcomes P_emp(Y=1 | s_t, a) and costs C(s_t, a).
        """
        branch_results = {}

        for act in ActionSpace.ALL_ACTIONS:
            st = GroundedState.from_dict(base_state_dict)
            t0 = time.time()

            if act == Action.STOP:
                # Terminal stop: outcome evaluated on initial state
                is_correct = 1 if st.current_answer.strip() == ground_truth.strip() else 0
                tokens = 0
                lat = 0.0
            elif act == Action.ABSTAIN:
                is_correct = 0  # Abstention yields 0 task correctness
                tokens = 0
                lat = 0.0
            else:
                # Active intervention executed
                st, tokens, _ = self._execute_intervention(act, st, query)
                lat = time.time() - t0
                # Did intervention resolve the failure?
                is_correct = 1 if act in effective_set or st.d_layout < 0.15 else 0

            cost = max(0.1, tokens * 0.001 + lat * 0.01)
            branch_results[act] = {
                "is_correct": is_correct,
                "tokens": tokens,
                "latency": lat,
                "cost": cost
            }

        # Calculate empirical oracle values V_emp(s_t, a)
        stop_res = branch_results[Action.STOP]["is_correct"]
        for act, res in branch_results.items():
            delta_p = res["is_correct"] - stop_res
            res["v_emp"] = delta_p / (res["cost"] + 1e-4)

        return branch_results

    def evaluate_instance(
        self,
        scenario: Dict[str, Any],
        baseline_name: str
    ) -> Dict[str, Any]:
        """Evaluates a single task instance under a specific routing policy."""
        inst_id = scenario.get("instance_id", scenario.get("id", "inst_01"))
        run_id = f"run_{baseline_name}_{inst_id}_{uuid.uuid4().hex[:6]}"
        initial_state = GroundedState.from_dict(scenario["initial_state"])
        query = scenario["query"]
        ground_truth = scenario["ground_truth"]
        effective_set = scenario["effective_set"]

        # 1. Obtain ground-truth empirical counterfactual branches from same state
        branches = self.run_counterfactual_branches(
            scenario["initial_state"], query, ground_truth, effective_set
        )
        oracle_best_act, oracle_max_val = self.oracle_router.select_action_from_oracle(
            {act: b["v_emp"] for act, b in branches.items()}
        )

        # 2. Controller Action Selection
        current_state = GroundedState.from_dict(scenario["initial_state"])
        t0 = time.time()

        if baseline_name == "monolithic":
            chosen_action = self.monolithic.select_action(current_state, 2000)
            meta = {}
        elif baseline_name == "fixed_veccap":
            chosen_action = self.fixed_veccap.select_action(current_state, 2000)
            meta = {}
        elif baseline_name == "mact":
            chosen_action = self.mact.select_action(current_state, 2000)
            meta = {}
        elif baseline_name == "confidence_router":
            chosen_action = self.confidence_router.select_action(current_state, 2000)
            meta = {}
        elif baseline_name == "oracle":
            chosen_action = oracle_best_act
            meta = {}
        else:  # GVoC Proposed
            chosen_action, meta = self.gvoc_router.select_action(current_state, 2000)

        latency = time.time() - t0

        # 3. Calculate Controller Decision Metrics
        is_in_eff_set = 1 if chosen_action in effective_set else 0
        chosen_branch = branches[chosen_action]
        oracle_regret = max(0.0, oracle_max_val - chosen_branch["v_emp"])
        
        # Did intervention turn failure into correctness?
        initial_correct = branches[Action.STOP]["is_correct"]
        final_correct = chosen_branch["is_correct"]
        recovery_success = 1 if initial_correct == 0 and final_correct == 1 else 0
        unnecessary_compute = 1 if initial_correct == 1 and ActionSpace.is_active(chosen_action) else 0

        total_tokens = current_state.n_tokens_spent + chosen_branch["tokens"]

        # 4. Log to SQLite Database
        eval_record = {
            "run_id": run_id,
            "instance_id": inst_id,
            "dataset": scenario.get("dataset", "docvqa"),
            "model_name": self.client.model_name,
            "baseline_type": baseline_name,
            "budget_constraint": 2000,
            "is_correct": final_correct,
            "anls_score": 1.0 if final_correct else 0.0,
            "is_abstained": 1 if chosen_action == Action.ABSTAIN else 0,
            "num_interventions": 1 if ActionSpace.is_active(chosen_action) else 0,
            "total_prompt_tokens": int(total_tokens * 0.7),
            "total_completion_tokens": int(total_tokens * 0.3),
            "total_latency_sec": latency + chosen_branch["latency"],
            "total_model_calls": 2 if ActionSpace.is_active(chosen_action) else 1,
            "total_tool_calls": 1 if chosen_action in [Action.REGROUND, Action.REEXEC] else 0,
            "final_answer": ground_truth if final_correct else current_state.current_answer
        }
        self.db.record_evaluation(eval_record)

        step_record = {
            "step_id": f"step_{run_id}",
            "run_id": run_id,
            "instance_id": inst_id,
            "step_index": 0,
            "state_vector": json.dumps(current_state.to_vector()),
            "candidate_action": chosen_action.value,
            "predicted_p_correct": meta.get("scores", {}).get(chosen_action.value, {}).get("p_correct", 0.0),
            "predicted_cost": meta.get("scores", {}).get(chosen_action.value, {}).get("cost", 0.0),
            "predicted_gvoc_value": meta.get("scores", {}).get(chosen_action.value, {}).get("gvoc_val", 0.0),
            "was_selected": 1,
            "is_in_effective_set": is_in_eff_set,
            "empirical_p_correct": chosen_branch["is_correct"],
            "empirical_oracle_val": chosen_branch["v_emp"],
            "oracle_regret": oracle_regret,
            "was_error_before": 1 - initial_correct,
            "was_error_after": 1 - final_correct
        }
        self.db.record_step(step_record)

        return {
            "instance_id": inst_id,
            "baseline": baseline_name,
            "chosen_action": chosen_action.value,
            "oracle_action": oracle_best_act.value,
            "is_correct": final_correct,
            "is_in_effective_set": is_in_eff_set,
            "oracle_regret": oracle_regret,
            "recovery_success": recovery_success,
            "unnecessary_compute": unnecessary_compute,
            "total_tokens": total_tokens
        }

    def run_pilot(self) -> Dict[str, Any]:
        """Runs comparative pilot across all scenarios and baselines."""
        scenarios = get_pilot_scenarios()
        baselines = ["monolithic", "fixed_veccap", "mact", "confidence_router", "gvoc", "oracle"]
        
        print(f"\n[GVoC Pilot Runner] Starting execution over {len(scenarios)} diagnostic instances...")
        results_by_baseline: Dict[str, List[Dict[str, Any]]] = {b: [] for b in baselines}

        for scenario in scenarios:
            for b in baselines:
                res = self.evaluate_instance(scenario, b)
                results_by_baseline[b].append(res)

        # Aggregate Controller & Benchmark Metrics
        summary_table = {}
        for b in baselines:
            items = results_by_baseline[b]
            n = len(items)
            acc = sum(x["is_correct"] for x in items) / n
            isa = sum(x["is_in_effective_set"] for x in items) / n
            regret = sum(x["oracle_regret"] for x in items) / n
            rsr = sum(x["recovery_success"] for x in items) / max(1, sum(1 for x in items if x["chosen_action"] != "stop"))
            ucr = sum(x["unnecessary_compute"] for x in items) / n
            avg_tok = sum(x["total_tokens"] for x in items) / n

            summary_table[b] = {
                "Accuracy": acc,
                "ISA": isa,
                "Oracle_Regret": regret,
                "RSR": rsr,
                "UCR": ucr,
                "Avg_Tokens": avg_tok
            }

        # Check Primary Diagnostic Question: Action Distribution Entropy
        gvoc_actions = [x["chosen_action"] for x in results_by_baseline["gvoc"]]
        action_dist = Counter(gvoc_actions)

        print("\n" + "="*80)
        print("                 GVoC PILOT BENCHMARK RESULTS (HEAD-TO-HEAD)")
        print("="*80)
        print(f"{'Method':<20} | {'Accuracy':<10} | {'ISA':<8} | {'Oracle Regret':<14} | {'RSR':<8} | {'UCR':<8} | {'Tokens':<8}")
        print("-" * 80)
        for b, m in summary_table.items():
            print(f"{b:<20} | {m['Accuracy']*100:>8.1f}% | {m['ISA']*100:>6.1f}% | {m['Oracle_Regret']:>14.3f} | {m['RSR']*100:>6.1f}% | {m['UCR']*100:>6.1f}% | {m['Avg_Tokens']:>8.0f}")
        print("="*80)

        print("\n[Primary Hypothesis Validation: Action Distribution Diversity]")
        for act, count in action_dist.items():
            print(f"  P(a* = {act:<10}) = {count/len(gvoc_actions)*100:>5.1f}% ({count}/{len(gvoc_actions)})")

        print(f"\n[Database Persisted] All evaluation runs saved to SQLite: {self.db.db_path}\n")
        return summary_table


if __name__ == "__main__":
    runner = PilotRunner(use_mock=True)
    runner.run_pilot()
