"""
Diagnostic Multi-Domain Pilot Scenarios for GVoC Engine.
Covers DocVQA, ChartQA, and TableBench instances with controlled failure modes.
"""

from typing import List, Dict, Any
from gvoc_engine.core.action_space import Action


def get_pilot_scenarios() -> List[Dict[str, Any]]:
    """Returns diverse diagnostic VDU tasks with varying failure profiles."""
    return [
        {
            "instance_id": "docvqa_inv_001",
            "dataset": "docvqa",
            "query": "What is the total balance due indicated on the invoice header?",
            "ground_truth": "$1,450.00",
            "initial_state": {
                "c_plan": 0.88, "c_exec": 0.82, "c_judg": 0.35, "c_consensus": 0.80,
                "d_layout": 0.45, "d_traj": 0.10, "d_evidence": 0.40, "d_answer": 0.12,
                "rho_visual": 0.65, "current_bbox": [80, 50, 160, 240], "current_answer": "$145.00"
            },
            "failure_type": "spatial_shift_30px",
            "effective_set": [Action.REGROUND]
        },
        {
            "instance_id": "tablebench_fin_002",
            "dataset": "tablebench",
            "query": "Calculate the YoY net operating income growth between 2024 and 2025.",
            "ground_truth": "14.2%",
            "initial_state": {
                "c_plan": 0.40, "c_exec": 0.85, "c_judg": 0.60, "c_consensus": 0.50,
                "d_layout": 0.08, "d_traj": 0.52, "d_evidence": 0.12, "d_answer": 0.25,
                "rho_visual": 0.70, "current_bbox": [200, 100, 350, 480], "current_answer": "8.5%"
            },
            "failure_type": "reading_order_inversion",
            "effective_set": [Action.REPLAN, Action.REGROUND]
        },
        {
            "instance_id": "chartqa_bar_003",
            "dataset": "chartqa",
            "query": "What is the value of the third blue bar representing North America?",
            "ground_truth": "78.4",
            "initial_state": {
                "c_plan": 0.90, "c_exec": 0.35, "c_judg": 0.55, "c_consensus": 0.70,
                "d_layout": 0.06, "d_traj": 0.08, "d_evidence": 0.15, "d_answer": 0.55,
                "rho_visual": 0.40, "current_bbox": [150, 200, 400, 350], "current_answer": "18.4"
            },
            "failure_type": "ocr_noise",
            "effective_set": [Action.REEXEC, Action.REGROUND]
        },
        {
            "instance_id": "docvqa_tax_004",
            "dataset": "docvqa",
            "query": "Is the signature date legally valid under section 4B?",
            "ground_truth": "Yes, certified on 12/04/2024",
            "initial_state": {
                "c_plan": 0.80, "c_exec": 0.82, "c_judg": 0.75, "c_consensus": 0.55,
                "d_layout": 0.18, "d_traj": 0.12, "d_evidence": 0.20, "d_answer": 0.10,
                "rho_visual": 0.55, "current_bbox": [450, 120, 520, 380], "current_answer": "Yes, certified"
            },
            "failure_type": "spatial_shift_10px",
            "effective_set": [Action.VERIFY, Action.REGROUND]
        },
        {
            "instance_id": "tablebench_tax_005",
            "dataset": "tablebench",
            "query": "What is the effective statutory tax rate for foreign subsidiaries in tier 2?",
            "ground_truth": "21.0%",
            "initial_state": {
                "c_plan": 0.92, "c_exec": 0.90, "c_judg": 0.94, "c_consensus": 0.95,
                "d_layout": 0.03, "d_traj": 0.04, "d_evidence": 0.05, "d_answer": 0.02,
                "rho_visual": 0.60, "current_bbox": [100, 150, 180, 320], "current_answer": "21.0%"
            },
            "failure_type": "normal_correct_state",
            "effective_set": [Action.STOP]
        },
        {
            "instance_id": "docvqa_blur_006",
            "dataset": "docvqa",
            "query": "Transcribe the water-damaged microprint code at the bottom right corner.",
            "ground_truth": "[UNINTELLIGIBLE]",
            "initial_state": {
                "c_plan": 0.25, "c_exec": 0.20, "c_judg": 0.15, "c_consensus": 0.10,
                "d_layout": 0.85, "d_traj": 0.70, "d_evidence": 0.90, "d_answer": 0.80,
                "rho_visual": 0.85, "current_bbox": [550, 480, 600, 550], "current_answer": "X9F-??"
            },
            "failure_type": "unrecoverable_noise",
            "effective_set": [Action.ABSTAIN]
        },
        {
            "instance_id": "chartqa_trend_007",
            "dataset": "chartqa",
            "query": "Does the 2026 forecast surpass the historical peak reached in 2021?",
            "ground_truth": "No, it remains 3.2% lower",
            "initial_state": {
                "c_plan": 0.85, "c_exec": 0.45, "c_judg": 0.60, "c_consensus": 0.65,
                "d_layout": 0.07, "d_traj": 0.15, "d_evidence": 0.10, "d_answer": 0.60,
                "rho_visual": 0.45, "current_bbox": [180, 220, 380, 500], "current_answer": "Yes, it surpasses"
            },
            "failure_type": "arithmetic_swap",
            "effective_set": [Action.REEXEC, Action.VERIFY]
        },
        {
            "instance_id": "tablebench_pivot_008",
            "dataset": "tablebench",
            "query": "Identify the department with the lowest Q3 equipment depreciation.",
            "ground_truth": "Logistics ($4,120)",
            "initial_state": {
                "c_plan": 0.45, "c_exec": 0.78, "c_judg": 0.50, "c_consensus": 0.48,
                "d_layout": 0.12, "d_traj": 0.48, "d_evidence": 0.18, "d_answer": 0.30,
                "rho_visual": 0.75, "current_bbox": [250, 100, 420, 520], "current_answer": "Human Resources"
            },
            "failure_type": "plan_step_deletion",
            "effective_set": [Action.REPLAN]
        }
    ]
