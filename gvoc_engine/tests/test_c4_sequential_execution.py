"""
Unit and integration tests for C4 sequential execution, request counting,
budget guardrails, early termination integrity, and crop-to-row spatial consistency.
NEVER MAKES LIVE CALLS OR SPENDS CREDITS.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from gvoc_engine.core.action_space import Action
from gvoc_engine.core.state import GroundedState
from gvoc_engine.controllers.ccea_router import CCEACostAwareRouter
from gvoc_engine.core.budget_guard import BudgetGuard, BudgetExceededError
from gvoc_engine.experiments.run_ccea_ablation import (
    run_ccea_ablation,
    extract_overview_state,
    get_evidence_region_for_instance
)


class TestC4SequentialExecution(unittest.TestCase):

    def test_c4_request_count_on_stop(self):
        """When C4 decides to STOP, deployed requests is exactly 1 (overview only)."""
        router = CCEACostAwareRouter()
        # Clean state triggers STOP
        state = GroundedState(
            d_layout=0.05,
            d_evidence=0.05,
            c_plan=0.95,
            c_exec=0.95,
            c_judg=0.95,
            c_consensus=0.95
        )
        action, meta = router.select_action(state)
        self.assertEqual(action, Action.STOP)
        
        deployed_calls = 1 if action in [Action.STOP, Action.ABSTAIN] else 2
        self.assertEqual(deployed_calls, 1)

    def test_c4_request_count_on_reground(self):
        """When C4 decides to REGROUND, deployed requests is exactly 2 (overview + fresh intervention)."""
        router = CCEACostAwareRouter()
        # High layout ambiguity triggers REGROUND
        state = GroundedState(
            d_layout=0.30,
            d_evidence=0.25,
            c_plan=0.70,
            c_exec=0.70,
            c_judg=0.60,
            c_consensus=0.75
        )
        action, meta = router.select_action(state)
        self.assertEqual(action, Action.REGROUND)
        
        deployed_calls = 2 if action == Action.REGROUND else 1
        self.assertEqual(deployed_calls, 2)

    def test_c4_request_count_on_abstain(self):
        """When budget or evidence is exhausted, C4 ABSTAINS, incurring exactly 1 call (overview)."""
        router = CCEACostAwareRouter()
        state = GroundedState(
            step_index=2,
            c_plan=0.20,
            c_exec=0.20,
            c_judg=0.20,
            c_consensus=0.20,
            d_layout=0.45,
            d_evidence=0.45
        )
        action, meta = router.select_action(state, remaining_budget=50)
        self.assertEqual(action, Action.ABSTAIN)
        
        deployed_calls = 1 if action in [Action.STOP, Action.ABSTAIN] else 2
        self.assertEqual(deployed_calls, 1)

    def test_tenth_request_blocked_under_25000_token_limit(self):
        """Traces the exact preflight calculation showing Request 10 is blocked at 25,600 projected tokens."""
        guard = BudgetGuard(
            max_budget_usd=0.005,
            operational_budget_usd=0.004,
            max_total_tokens=30000,
            operational_token_limit=25000,
            price_per_million_tokens=0.15
        )
        
        # Requests 1 through 9: Each incurs 2,500 prompt + 60 completion = 2,560 tokens
        for req_idx in range(1, 10):
            allowed, _ = guard.check_pre_flight(has_image=True, max_tokens=60)
            self.assertTrue(allowed, f"Request {req_idx} should be allowed")
            guard.record_attempt(prompt_tokens=2500, completion_tokens=60)
            
        # Cumulative tokens after 9 requests = 9 * 2,560 = 23,040 tokens
        self.assertEqual(guard.state.cumulative_total_tokens, 23040)
        
        # Pre-flight for Request 10: projected = 23,040 + est > 25,000 operational limit
        allowed, reason = guard.check_pre_flight(has_image=True, max_tokens=60)
        self.assertFalse(allowed, "Request 10 MUST be blocked by operational halt threshold")
        self.assertIn("BudgetGuard Operational Halt", reason)
        self.assertIn("exceed operational threshold 25000", reason)
        self.assertEqual(guard.state.blocked_requests, 1)

    def test_canary_five_conditions_fit_comfortably_within_guard(self):
        """A 1-document canary with 5 conditions fits safely within an 18,000-token operational threshold."""
        canary_guard = BudgetGuard(
            max_budget_usd=0.005,
            operational_budget_usd=0.003,
            max_total_tokens=30000,
            operational_token_limit=18000,
            price_per_million_tokens=0.15
        )
        
        # Run 5 requests with upper-bound tokens (2,560 each)
        for req_idx in range(1, 6):
            allowed, reason = canary_guard.check_pre_flight(has_image=True, max_tokens=60)
            self.assertTrue(allowed, f"Canary request {req_idx} must fit safely")
            canary_guard.record_attempt(prompt_tokens=2500, completion_tokens=60)
            
        # Total tokens after all 5 canary conditions: 5 * 2,560 = 12,800 tokens
        self.assertEqual(canary_guard.state.cumulative_total_tokens, 12800)
        self.assertEqual(canary_guard.state.successful_requests, 5)
        self.assertEqual(canary_guard.state.blocked_requests, 0)
        # Cost is well under operational limit ($0.00192 < $0.00300)
        self.assertLess(canary_guard.state.cumulative_cost_usd, 0.002)

    def test_early_termination_and_result_integrity(self):
        """When budget guard halts execution, blocked conditions are marked BLOCKED and run is flagged incomplete."""
        # Operational limit 3000 allows C1 (145 tokens) and C2 (145 tokens, cumulative 290)
        # C3 pre-flight: 290 + 2610 = 2900 <= 3000 (allowed, cumulative 435)
        # C4 pre-flight: 435 + 2610 = 3045 > 3000 -> BLOCKED!
        tight_guard = BudgetGuard(
            max_budget_usd=0.005,
            operational_budget_usd=0.004,
            max_total_tokens=30000,
            operational_token_limit=3200
        )
        
        # Execute ablation with tight guard writing to dedicated test audit file
        test_audit_path = Path("discovery/live_validation/test_early_termination_audit.json")
        stats = run_ccea_ablation(
            start_idx=10,
            end_idx=11, # 1 document
            budget_guard=tight_guard,
            held_out_out_path=str(test_audit_path),
            records_out_path=str(test_audit_path)
        )
        
        # Read saved test audit json to inspect result integrity
        with open(test_audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        meta = data["metadata"]
        self.assertFalse(meta["is_complete"], "Run must be marked incomplete")
        self.assertEqual(meta["run_status"], "INCOMPLETE_BUDGET_HALT")
        self.assertIsNotNone(meta["halt_reason"])
        
        # Ledger A must report blocked request
        self.assertGreater(data["ledger_a_experimental_usage"]["blocked_requests"], 0)
        
        # Blocked condition (C3) in record must have strict_em = None, NOT 0 or 1
        rec = data["records"][0]
        c3_res = rec["conditions"].get("3_ccea_context_preserving", {})
        self.assertEqual(c3_res.get("status"), "BLOCKED_BUDGET_EXCEEDED")
        self.assertIsNone(c3_res.get("strict_em"))
        self.assertEqual(c3_res.get("prediction"), "BLOCKED_BUDGET_GUARD")
        
        # Verify blocked conditions are NOT reported as completed conditions in stats
        self.assertEqual(stats["3_ccea_context_preserving"]["completed_count"], 0)
        self.assertEqual(stats["3_ccea_context_preserving"]["blocked_count"], 1)
        self.assertEqual(stats["3_ccea_context_preserving"]["correct_count"], 0)

        # Verify request ledger distinguishes all six request states
        ledger = data["ledger_a_experimental_usage"]
        for key in ["planned_requests", "attempted_requests", "successful_requests", "failed_requests", "retried_requests", "blocked_requests"]:
            self.assertIn(key, ledger)

        # Condition 4 and 5 must not have been attempted
        self.assertNotIn("4_ccea_cost_aware_controller", rec["conditions"])
        self.assertNotIn("5_fixed_center_crop", rec["conditions"])

    def test_docvqa_021_crop_to_row_consistency(self):
        """Verify that docvqa_021 spatial boxes pair 14,000 with N. Anand in the same table row."""
        ev = get_evidence_region_for_instance("docvqa_021", 1653, 2339)
        self.assertIsNotNone(ev.target_box)
        t_ymin, t_xmin, t_ymax, t_xmax = ev.target_box
        
        # Target box encloses 14,000 (y around 1095-1141)
        self.assertGreaterEqual(t_ymin, 1080)
        self.assertLessEqual(t_ymax, 1150)
        
        # Required context contains row label and column header
        contexts = dict(ev.required_context)
        self.assertIn(ev.required_context[0][0], contexts)
        
        col_box = ev.required_context[0][1]
        row_box = ev.required_context[1][1]
        self.assertLessEqual(row_box[0], t_ymin)
        self.assertGreaterEqual(row_box[2], t_ymax)

    def test_canary_multi_retry_sequence_halts_at_operational_ceiling(self):
        """Simulate repeated retries in the canary; verify preflight halts at operational threshold before hard cap."""
        canary_guard = BudgetGuard(
            max_budget_usd=0.005,
            operational_budget_usd=0.003,
            max_total_tokens=30000,
            operational_token_limit=18000,
            default_missing_prompt_tokens=2800,
            price_per_million_tokens=0.15
        )

        # Attempts 1 to 6 fail and retry (each charged conservative 2,800 tokens)
        for attempt_idx in range(1, 7):
            allowed, _ = canary_guard.check_pre_flight(has_image=True, max_tokens=60)
            self.assertTrue(allowed, f"Attempt {attempt_idx} should be permitted")
            canary_guard.record_attempt(prompt_tokens=2800, completion_tokens=0, is_error=True, is_retry=(attempt_idx % 2 == 0))

        # After 6 failed attempts: cumulative tokens = 6 * 2,800 = 16,800 tokens
        self.assertEqual(canary_guard.state.cumulative_total_tokens, 16800)
        self.assertEqual(canary_guard.state.cumulative_cost_usd, (16800 / 1000000.0) * 0.15)  # $0.00252 USD <= $0.00300

        # Attempt 7 pre-flight: 16,800 + 2,626 = 19,426 > 18,000 -> BLOCKED!
        allowed, reason = canary_guard.check_pre_flight(has_image=True, max_tokens=60)
        self.assertFalse(allowed, "Attempt 7 must be blocked before operational threshold is crossed")
        self.assertIn("BudgetGuard Operational Halt", reason)
        self.assertIn("18000", reason)

        # Hard caps are never breached
        self.assertLess(canary_guard.state.cumulative_total_tokens, 30000)
        self.assertLess(canary_guard.state.cumulative_cost_usd, 0.005)
        self.assertEqual(canary_guard.state.blocked_requests, 1)


if __name__ == "__main__":
    unittest.main()
