"""
SQLite Evaluation Database for GVoC Engine.

Records Tier 1 (Task Performance), Tier 2 (Controller Diagnostics), and
Tier 3 (Compute Accounting) metrics with zero data loss and instant resumability.
"""

import sqlite3
from typing import Dict, Any, List, Optional
from gvoc_engine.config import DATABASE_PATH


class EvaluationDatabase:
    """Manages SQLite storage for all comparative benchmarks and pilot studies."""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
        """Initializes relational tables if they do not exist."""
        conn = self._get_connection()
        cur = conn.cursor()

        # Table 1: Instance-Level End-to-End Evaluations
        cur.execute("""
        CREATE TABLE IF NOT EXISTS task_evaluations (
            run_id TEXT PRIMARY KEY,
            instance_id TEXT,
            dataset TEXT,
            model_name TEXT,
            baseline_type TEXT,
            budget_constraint INTEGER,
            is_correct INTEGER,
            anls_score REAL,
            is_abstained INTEGER,
            num_interventions INTEGER,
            total_prompt_tokens INTEGER,
            total_completion_tokens INTEGER,
            total_latency_sec REAL,
            total_model_calls INTEGER,
            total_tool_calls INTEGER,
            final_answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Table 2: Intermediate Intervention Diagnostics & Regret
        cur.execute("""
        CREATE TABLE IF NOT EXISTS intervention_steps (
            step_id TEXT PRIMARY KEY,
            run_id TEXT,
            instance_id TEXT,
            step_index INTEGER,
            state_vector TEXT,
            candidate_action TEXT,
            predicted_p_correct REAL,
            predicted_cost REAL,
            predicted_gvoc_value REAL,
            was_selected INTEGER,
            is_in_effective_set INTEGER,
            empirical_p_correct REAL,
            empirical_oracle_val REAL,
            oracle_regret REAL,
            was_error_before INTEGER,
            was_error_after INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()
        conn.close()

    def record_evaluation(self, eval_data: Dict[str, Any]):
        """Inserts an instance-level evaluation record."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("""
        INSERT OR REPLACE INTO task_evaluations (
            run_id, instance_id, dataset, model_name, baseline_type,
            budget_constraint, is_correct, anls_score, is_abstained,
            num_interventions, total_prompt_tokens, total_completion_tokens,
            total_latency_sec, total_model_calls, total_tool_calls,
            final_answer
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            eval_data["run_id"],
            eval_data["instance_id"],
            eval_data.get("dataset", "docvqa"),
            eval_data.get("model_name", "gemini-1.5-flash"),
            eval_data.get("baseline_type", "gvoc"),
            eval_data.get("budget_constraint", 2000),
            eval_data.get("is_correct", 0),
            eval_data.get("anls_score", 0.0),
            eval_data.get("is_abstained", 0),
            eval_data.get("num_interventions", 0),
            eval_data.get("total_prompt_tokens", 0),
            eval_data.get("total_completion_tokens", 0),
            eval_data.get("total_latency_sec", 0.0),
            eval_data.get("total_model_calls", 1),
            eval_data.get("total_tool_calls", 0),
            eval_data.get("final_answer", "")
        ))
        conn.commit()
        conn.close()

    def record_step(self, step_data: Dict[str, Any]):
        """Inserts a step-level controller decision record."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("""
        INSERT OR REPLACE INTO intervention_steps (
            step_id, run_id, instance_id, step_index, state_vector,
            candidate_action, predicted_p_correct, predicted_cost,
            predicted_gvoc_value, was_selected, is_in_effective_set,
            empirical_p_correct, empirical_oracle_val, oracle_regret,
            was_error_before, was_error_after
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            step_data["step_id"],
            step_data["run_id"],
            step_data["instance_id"],
            step_data.get("step_index", 0),
            step_data.get("state_vector", "[]"),
            step_data.get("candidate_action", "stop"),
            step_data.get("predicted_p_correct", 0.0),
            step_data.get("predicted_cost", 0.0),
            step_data.get("predicted_gvoc_value", 0.0),
            step_data.get("was_selected", 0),
            step_data.get("is_in_effective_set", 0),
            step_data.get("empirical_p_correct", 0.0),
            step_data.get("empirical_oracle_val", 0.0),
            step_data.get("oracle_regret", 0.0),
            step_data.get("was_error_before", 0),
            step_data.get("was_error_after", 0)
        ))
        conn.commit()
        conn.close()

    def get_summary_statistics(self, baseline_type: Optional[str] = None) -> Dict[str, Any]:
        """Calculates aggregate metrics across logged runs."""
        conn = self._get_connection()
        cur = conn.cursor()

        query = "SELECT is_correct, anls_score, is_abstained, total_prompt_tokens, total_completion_tokens, total_latency_sec FROM task_evaluations"
        params = ()
        if baseline_type:
            query += " WHERE baseline_type = ?"
            params = (baseline_type,)

        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return {"count": 0}

        count = len(rows)
        correct = sum(r[0] for r in rows)
        avg_anls = sum(r[1] for r in rows) / count
        abstained = sum(r[2] for r in rows)
        avg_tokens = sum(r[3] + r[4] for r in rows) / count
        avg_lat = sum(r[5] for r in rows) / count

        return {
            "count": count,
            "accuracy": correct / count,
            "avg_anls": avg_anls,
            "abstention_rate": abstained / count,
            "avg_tokens": avg_tokens,
            "avg_latency_sec": avg_lat
        }
