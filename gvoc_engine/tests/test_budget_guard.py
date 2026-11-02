"""
Unit and integration tests for Conservative Request-Level BudgetGuard
and LiveVLMClient budget enforcement under mocked API conditions.
NEVER MAKES LIVE API CALLS OR SPENDS CREDITS.
"""

import unittest
from unittest.mock import MagicMock
from gvoc_engine.core.budget_guard import BudgetGuard, BudgetExceededError, BudgetState
from discovery.live_validation.live_client import LiveVLMClient, LiveAPIError


class TestBudgetGuard(unittest.TestCase):

    def test_pre_flight_refusal_when_cost_exceeds_budget(self):
        """Refuses request if estimated cost exceeds remaining budget."""
        # Setup guard with tiny budget ($0.0001, less than 1 image call)
        guard = BudgetGuard(max_budget_usd=0.0001, operational_budget_usd=0.00008)
        allowed, reason = guard.check_pre_flight(has_image=True, max_tokens=60, prompt_char_len=200)
        self.assertFalse(allowed)
        self.assertTrue("BudgetGuard Refusal" in reason or "BudgetGuard Operational Halt" in reason)

    def test_pre_flight_refusal_when_token_threshold_crossed(self):
        """Refuses request if estimated tokens exceed remaining token limit."""
        guard = BudgetGuard(max_total_tokens=1000, operational_token_limit=800)
        # Image call estimates ~2500 prompt tokens + 60 completion = 2560 > 800
        allowed, reason = guard.check_pre_flight(has_image=True, max_tokens=60)
        self.assertFalse(allowed)
        self.assertTrue("BudgetGuard Refusal" in reason or "BudgetGuard Operational Halt" in reason)

    def test_missing_usage_metadata_conservatively_defaulted(self):
        """Missing usage metadata must never be treated as zero."""
        guard = BudgetGuard(
            default_missing_prompt_tokens=2500,
            default_missing_completion_tokens=60
        )
        # Simulate response with missing usage (None)
        status = guard.record_attempt(prompt_tokens=None, completion_tokens=None, total_tokens=None)
        self.assertEqual(status["cumulative_requests"], 1)
        self.assertEqual(status["cumulative_total_tokens"], 2560)
        self.assertGreater(status["cumulative_cost_usd"], 0.0)

    def test_api_error_recorded_against_budget(self):
        """Failed requests must be accounted for in cumulative requests and charged conservatively."""
        guard = BudgetGuard()
        status = guard.record_attempt(is_error=True)
        self.assertEqual(status["cumulative_requests"], 1)
        # Conservative prompt tokens charged, 0 completion tokens
        self.assertEqual(guard.state.cumulative_completion_tokens, 0)
        self.assertEqual(guard.state.cumulative_prompt_tokens, 2800)
        self.assertGreater(guard.state.cumulative_cost_usd, 0.0)

    def test_budget_exhaustion_immediately_before_request(self):
        """When budget is fully exhausted, pre-flight immediately rejects new requests."""
        guard = BudgetGuard(max_budget_usd=0.05, operational_budget_usd=0.04)
        # Simulate spending up to limit
        guard.state.cumulative_cost_usd = 0.045
        allowed, reason = guard.check_pre_flight(has_image=False, max_tokens=10)
        self.assertFalse(allowed)
        self.assertTrue("BudgetGuard Operational Halt" in reason or "BudgetGuard Refusal" in reason)

    def test_client_enforces_output_token_limit(self):
        """LiveVLMClient must reject requests with max_tokens > conservative safety limit (250)."""
        mock_factory = MagicMock()
        client = LiveVLMClient(provider="mistral", client_factory=mock_factory)
        with self.assertRaises(ValueError):
            client.call("Test prompt", max_tokens=500)

    def test_retry_accounting_in_client(self):
        """Client retries must be recorded against the budget guard for every attempt."""
        guard = BudgetGuard()
        mock_client = MagicMock()
        # First attempt fails with 500, second succeeds
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Mocked answer"))]
        mock_usage = MagicMock(prompt_tokens=300, completion_tokens=20, total_tokens=320)
        mock_resp.usage = mock_usage
        
        mock_client.chat.completions.create.side_effect = [
            Exception("Internal Server Error 500"),
            mock_resp
        ]
        
        client = LiveVLMClient(
            provider="mistral",
            budget_guard=guard,
            max_retries=2,
            client_factory=lambda: mock_client
        )
        
        res = client.call("Test prompt", max_tokens=50)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["text"], "Mocked answer")
        self.assertEqual(res["attempt"], 2)
        # Total attempts recorded should be 2 (1 failed + 1 successful)
        self.assertEqual(guard.state.cumulative_requests, 2)
        # Prompt tokens should include failed attempt fallback (2800) + success (300) = 3100
        self.assertEqual(guard.state.cumulative_prompt_tokens, 3100)

    def test_retry_loop_blocked_when_budget_exhausts_mid_retry(self):
        """If first attempt causes budget to breach threshold, retry is blocked by pre-flight check."""
        guard = BudgetGuard(max_budget_usd=0.001, operational_budget_usd=0.0008)
        # Set close to limit
        guard.state.cumulative_cost_usd = 0.0005
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Temporary error")
        
        client = LiveVLMClient(
            provider="mistral",
            budget_guard=guard,
            max_retries=2,
            client_factory=lambda: mock_client
        )
        
        # First attempt proceeds, fails, charges conservative 2500 tokens -> breaches 0.0008 operational limit
        # Second attempt should be blocked by pre-flight check raising BudgetExceededError
        with self.assertRaises(BudgetExceededError):
            client.call("Test prompt", max_tokens=50)
        
        self.assertEqual(guard.state.cumulative_requests, 1)

    def test_pre_flight_refusal_when_hard_budget_breached(self):
        """Hard monetary ceiling ($0.05) immediately refuses new requests."""
        guard = BudgetGuard(max_budget_usd=0.001, operational_budget_usd=0.0009)
        guard.state.cumulative_cost_usd = 0.00095
        allowed, reason = guard.check_pre_flight(has_image=True, max_tokens=60)
        self.assertFalse(allowed)
        self.assertIn("BudgetGuard Refusal", reason)
        self.assertIn("hard limit", reason)

    def test_pre_flight_refusal_when_hard_tokens_breached(self):
        """Hard token ceiling immediately refuses new requests."""
        guard = BudgetGuard(max_total_tokens=5000, operational_token_limit=4800)
        guard.state.cumulative_total_tokens = 4900
        allowed, reason = guard.check_pre_flight(has_image=True, max_tokens=60)
        self.assertFalse(allowed)
        self.assertIn("BudgetGuard Refusal", reason)
        self.assertIn("hard limit", reason)

    def test_ledger_tracks_all_six_states(self):
        """BudgetGuard state tracks planned, attempted, successful, failed, retried, and blocked requests."""
        guard = BudgetGuard(planned_requests=10)
        self.assertEqual(guard.state.planned_requests, 10)
        
        # 1. Success
        guard.record_attempt(prompt_tokens=100, completion_tokens=20, is_error=False, is_retry=False)
        self.assertEqual(guard.state.attempted_requests, 1)
        self.assertEqual(guard.state.successful_requests, 1)
        self.assertEqual(guard.state.failed_requests, 0)
        self.assertEqual(guard.state.retried_requests, 0)

        # 2. Failure on first attempt
        guard.record_attempt(prompt_tokens=None, completion_tokens=None, is_error=True, is_retry=False)
        self.assertEqual(guard.state.attempted_requests, 2)
        self.assertEqual(guard.state.successful_requests, 1)
        self.assertEqual(guard.state.failed_requests, 1)
        self.assertEqual(guard.state.retried_requests, 0)

        # 3. Success on retry
        guard.record_attempt(prompt_tokens=100, completion_tokens=20, is_error=False, is_retry=True)
        self.assertEqual(guard.state.attempted_requests, 3)
        self.assertEqual(guard.state.successful_requests, 2)
        self.assertEqual(guard.state.failed_requests, 1)
        self.assertEqual(guard.state.retried_requests, 1)

        # 4. Blocked pre-flight
        guard.operational_token_limit = 100
        allowed, reason = guard.check_pre_flight(has_image=True, max_tokens=60)
        self.assertFalse(allowed)
        self.assertEqual(guard.state.blocked_requests, 1)

    def test_maximum_retry_sequence_exhaustion(self):
        """When all attempts fail up to max_retries, each attempt is charged conservative prompt fallback and LiveAPIError is raised."""
        guard = BudgetGuard(
            max_budget_usd=0.05,
            operational_budget_usd=0.04,
            default_missing_prompt_tokens=2800
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Persistent API Outage 500")

        client = LiveVLMClient(
            provider="mistral",
            budget_guard=guard,
            max_retries=2,  # 2 total attempts (1 initial + 1 retry)
            client_factory=lambda: mock_client
        )

        with self.assertRaises(LiveAPIError):
            client.call("Test prompt", max_tokens=60)

        # Both attempts recorded
        self.assertEqual(guard.state.attempted_requests, 2)
        self.assertEqual(guard.state.failed_requests, 2)
        self.assertEqual(guard.state.successful_requests, 0)
        self.assertEqual(guard.state.retried_requests, 1)
        # Conservative prompt fallback (2,800) charged for both attempts
        self.assertEqual(guard.state.cumulative_prompt_tokens, 2 * 2800)
        self.assertEqual(guard.state.cumulative_completion_tokens, 0)
        self.assertEqual(guard.state.cumulative_total_tokens, 5600)
        self.assertAlmostEqual(guard.state.cumulative_cost_usd, (5600 / 1000000.0) * 0.15, places=7)

    def test_client_rejects_oversized_prompt(self):
        """LiveVLMClient must reject prompts exceeding 4000 characters."""
        client = LiveVLMClient(provider="mistral", client_factory=MagicMock())
        oversized_prompt = "A" * 4001
        with self.assertRaises(ValueError) as ctx:
            client.call(oversized_prompt, max_tokens=60)
        self.assertIn("exceeds conservative safety limit", str(ctx.exception))

    def test_repeated_missing_usage_metadata_aborts_cleanly(self):
        """BudgetGuard must abort cleanly if usage metadata is missing repeatedly."""
        guard = BudgetGuard(max_consecutive_missing_metadata=3)
        # 1st missing metadata
        guard.record_attempt(prompt_tokens=None, completion_tokens=None, is_error=False)
        self.assertEqual(guard.consecutive_missing_metadata, 1)

        # 2nd missing metadata
        guard.record_attempt(prompt_tokens=None, completion_tokens=None, is_error=False)
        self.assertEqual(guard.consecutive_missing_metadata, 2)

        # 3rd missing metadata must raise BudgetExceededError
        with self.assertRaises(BudgetExceededError) as ctx:
            guard.record_attempt(prompt_tokens=None, completion_tokens=None, is_error=False)
        self.assertIn("Usage metadata missing repeatedly", str(ctx.exception))

    # =========================================================================
    # DETERMINISTIC NINE-SCENARIO OFFLINE STRESS SUITE (A through I)
    # ZERO NETWORK ACCESS: All tests use mock client / synthetic states
    # =========================================================================

    def test_scenario_a_expected_nominal_usage(self):
        """Scenario A: 24 requests under nominal measured usage complete within 60,000 tokens / $0.01000."""
        guard = BudgetGuard(
            max_budget_usd=0.015,
            operational_budget_usd=0.010,
            max_total_tokens=75000,
            operational_token_limit=60000,
            planned_requests=24
        )
        # Simulate 12 C1 overview (2380 tokens each) + 12 C4 reground crops (mean 1545 tokens each)
        c1_tokens = [2380] * 12
        c4_tokens = [1327, 2380, 1834, 928, 1880, 1594, 2380, 1057, 1417, 1127, 1310, 1312]
        
        for c1, c4 in zip(c1_tokens, c4_tokens):
            ok1, r1 = guard.check_pre_flight(has_image=True, max_tokens=60)
            self.assertTrue(ok1, f"C1 check failed: {r1}")
            guard.record_attempt(prompt_tokens=c1-20, completion_tokens=20, is_error=False)

            ok4, r4 = guard.check_pre_flight(has_image=True, max_tokens=60)
            self.assertTrue(ok4, f"C4 check failed: {r4}")
            guard.record_attempt(prompt_tokens=c4-20, completion_tokens=20, is_error=False)

        summary = guard.get_summary()
        self.assertEqual(summary["attempted_requests"], 24)
        self.assertEqual(summary["blocked_requests"], 0)
        self.assertLessEqual(summary["cumulative_total_tokens"], 60000)
        self.assertLessEqual(summary["cumulative_cost_usd"], 0.01000)
        self.assertEqual(summary["cumulative_total_tokens"], 47106)
        self.assertAlmostEqual(summary["cumulative_cost_usd"], 0.0070659, places=6)
        self.assertFalse(summary["operational_limit_reached"])

    def test_scenario_b_maximum_permitted_retry_sequence(self):
        """Scenario B: Persistent retries fail closed before crossing the 60,000 operational ceiling."""
        guard = BudgetGuard(
            max_budget_usd=0.015,
            operational_budget_usd=0.010,
            max_total_tokens=75000,
            operational_token_limit=60000,
            default_missing_prompt_tokens=2800,
            default_missing_completion_tokens=60
        )
        # Each failed attempt charges 2800 prompt tokens.
        # Preflight reservation is 2860 tokens.
        # Run attempts until guard blocks.
        attempts_allowed = 0
        while True:
            allowed, reason = guard.check_pre_flight(has_image=True, max_tokens=60)
            if not allowed:
                self.assertIn("BudgetGuard Operational Halt", reason)
                break
            # Record failed attempt (charges 2800 prompt tokens)
            guard.record_attempt(is_error=True)
            attempts_allowed += 1

        # Must have halted strictly <= 60,000 tokens
        self.assertLessEqual(guard.state.cumulative_total_tokens, 60000)
        # Exactly 21 attempts allowed (21 x 2800 = 58,800 <= 60,000; next projected is 58,800 + 2860 = 61,660 > 60,000)
        self.assertEqual(attempts_allowed, 21)
        self.assertEqual(guard.state.cumulative_total_tokens, 21 * 2800)
        self.assertLessEqual(guard.state.cumulative_total_tokens, 60000)
        self.assertLessEqual(guard.state.cumulative_cost_usd, 0.01000)

    def test_scenario_c_repeated_rate_limit_failures(self):
        """Scenario C: Rate limit 429 errors trigger backoff and conservative accounting in client, halting cleanly."""
        guard = BudgetGuard(
            max_budget_usd=0.015,
            operational_budget_usd=0.010,
            max_total_tokens=75000,
            operational_token_limit=60000
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Rate limit reached 429: quota exhausted")

        client = LiveVLMClient(
            provider="mistral",
            budget_guard=guard,
            max_retries=2,
            client_factory=lambda: mock_client
        )

        calls_attempted = 0
        img_p = "discovery/live_validation/pilot_dataset/images/docvqa_034.jpg"
        while True:
            try:
                client.call("Test prompt", image_path=img_p, max_tokens=60)
            except LiveAPIError:
                calls_attempted += 1
            except BudgetExceededError as be:
                # Successfully halted by budget guard pre-flight check
                self.assertIn("BudgetGuard Operational Halt", str(be))
                break

        # Verification: Halted strictly <= 60,000 tokens
        self.assertLessEqual(guard.state.cumulative_total_tokens, 60000)
        self.assertLessEqual(guard.state.cumulative_cost_usd, 0.01000)
        self.assertGreater(guard.state.failed_requests, 0)

    def test_scenario_d_missing_usage_metadata(self):
        """Scenario D: Missing usage metadata charges conservative fallback and halts after 3 consecutive occurrences."""
        guard = BudgetGuard(
            operational_token_limit=60000,
            default_missing_prompt_tokens=2800,
            default_missing_completion_tokens=60,
            max_consecutive_missing_metadata=3
        )
        # Attempt 1: missing metadata -> charged 2860
        guard.record_attempt(prompt_tokens=None, completion_tokens=None, is_error=False)
        self.assertEqual(guard.state.conservatively_charged_tokens, 2860)
        self.assertEqual(guard.state.reported_tokens, 0)

        # Attempt 2: missing metadata -> charged another 2860
        guard.record_attempt(prompt_tokens=None, completion_tokens=None, is_error=False)
        self.assertEqual(guard.state.conservatively_charged_tokens, 5720)

        # Attempt 3: must abort cleanly with BudgetExceededError
        with self.assertRaises(BudgetExceededError) as ctx:
            guard.record_attempt(prompt_tokens=None, completion_tokens=None, is_error=False)
        self.assertIn("Usage metadata missing repeatedly", str(ctx.exception))

    def test_scenario_e_actual_usage_greater_than_reservation(self):
        """Scenario E: If actual tokens exceed reservation, ledger reconciles immediately, logs anomaly, and halts if limit reached."""
        guard = BudgetGuard(
            operational_token_limit=5000,
            default_missing_prompt_tokens=2800,
            default_missing_completion_tokens=60
        )
        # Pre-flight passes because estimate is 2860 <= 5000
        ok, _ = guard.check_pre_flight(has_image=True, max_tokens=60)
        self.assertTrue(ok)

        # Actual tokens returned from provider are unexpectedly large (e.g. 5200 tokens)
        guard.record_attempt(prompt_tokens=5100, completion_tokens=100, is_error=False)
        self.assertEqual(guard.state.cumulative_total_tokens, 5200)
        self.assertTrue(guard.state.operational_limit_reached)
        self.assertEqual(len(guard.state.overage_anomalies), 1)

        # Immediate next pre-flight must FAIL CLOSED
        ok2, reason2 = guard.check_pre_flight(has_image=True, max_tokens=60)
        self.assertFalse(ok2)
        self.assertIn("Operational ceiling already reached", reason2)

    def test_scenario_f_reservation_exactly_reaches_ceiling(self):
        """Scenario F: Equality policy allows projected usage that exactly equals the operational ceiling."""
        guard = BudgetGuard(operational_token_limit=60000)
        # Reservation for max_tokens=60 with default missing 2800 is 2860
        est_tokens, _ = guard.estimate_request_cost(has_image=True, max_tokens=60)
        # Set cumulative tokens so that cumulative + est_tokens == 60,000 exactly
        guard.state.cumulative_total_tokens = 60000 - est_tokens
        
        ok, reason = guard.check_pre_flight(has_image=True, max_tokens=60)
        self.assertTrue(ok, f"Equality case must be permitted, but was refused: {reason}")

    def test_scenario_g_reservation_exceeds_ceiling_by_one_token(self):
        """Scenario G: Projected usage that exceeds operational ceiling by even 1 token is strictly refused."""
        guard = BudgetGuard(operational_token_limit=60000)
        est_tokens, _ = guard.estimate_request_cost(has_image=True, max_tokens=60)
        # Set cumulative tokens so that cumulative + est_tokens == 60,001 (1 token over ceiling)
        guard.state.cumulative_total_tokens = 60000 - est_tokens + 1
        
        ok, reason = guard.check_pre_flight(has_image=True, max_tokens=60)
        self.assertFalse(ok)
        self.assertIn("BudgetGuard Operational Halt", reason)
        self.assertIn("60001", reason)

    def test_scenario_h_fits_token_limit_exceeds_cost_limit(self):
        """Scenario H: Request fits token limit but exceeds monetary cost limit is blocked on cost."""
        # Generous token limit (100,000) but tight cost limit ($0.00040)
        guard = BudgetGuard(
            max_budget_usd=0.001,
            operational_budget_usd=0.00040,
            operational_token_limit=100000,
            price_per_million_tokens=0.15
        )
        # Reservation of 2860 tokens costs $0.000429 > $0.00040
        ok, reason = guard.check_pre_flight(has_image=True, max_tokens=60)
        self.assertFalse(ok)
        self.assertIn("Projected cost", reason)
        self.assertIn("exceeds operational threshold", reason)

    def test_scenario_i_fits_cost_limit_exceeds_token_limit(self):
        """Scenario I: Request fits monetary cost limit but exceeds token limit is blocked on tokens."""
        # Generous cost limit ($0.05) but tight token limit (2500)
        guard = BudgetGuard(
            max_budget_usd=0.05,
            operational_budget_usd=0.04,
            operational_token_limit=2500
        )
        # Reservation of 2860 tokens > 2500 tokens
        ok, reason = guard.check_pre_flight(has_image=True, max_tokens=60)
        self.assertFalse(ok)
        self.assertIn("Projected tokens", reason)
        self.assertIn("exceed operational threshold", reason)


if __name__ == "__main__":
    unittest.main()

