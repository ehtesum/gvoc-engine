"""
Publication-Grade Figure Generator for GVoC Pilot Evaluation.
Outputs:
1. fig_pilot_regret_comparison.png (Oracle Regret comparison)
2. fig_pilot_pareto_frontier.png (Accuracy vs. Token Cost Frontier)
3. fig_pilot_action_distribution.png (Distribution of chosen repair interventions)
"""

import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from gvoc_engine.config import DATABASE_PATH

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def plot_pilot_results(db_path: str = DATABASE_PATH):
    """Queries SQLite evaluation database and generates comparative figures."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Query baseline summary metrics
    cur.execute("""
    SELECT 
        baseline_type,
        AVG(is_correct) as acc,
        AVG(total_prompt_tokens + total_completion_tokens) as avg_tokens
    FROM task_evaluations
    GROUP BY baseline_type
    """)
    rows = cur.fetchall()

    # 2. Query oracle regret per baseline
    cur.execute("""
    SELECT 
        t.baseline_type,
        AVG(s.oracle_regret) as avg_regret,
        AVG(s.is_in_effective_set) as avg_isa
    FROM intervention_steps s
    JOIN task_evaluations t ON s.run_id = t.run_id
    GROUP BY t.baseline_type
    """)
    regret_rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    # 3. Query GVoC Action Distribution
    cur.execute("""
    SELECT candidate_action, COUNT(*) 
    FROM intervention_steps s
    JOIN task_evaluations t ON s.run_id = t.run_id
    WHERE t.baseline_type = 'gvoc' AND s.was_selected = 1
    GROUP BY candidate_action
    """)
    action_counts = dict(cur.fetchall())
    conn.close()

    if not rows:
        print("[Plot Warning] No evaluation rows found in database.")
        return

    # Data structuring
    baselines = [r[0] for r in rows]
    accs = [r[1] * 100 for r in rows]
    tokens = [r[2] for r in rows]
    regrets = [regret_rows.get(b, (0.0, 0.0))[0] for b in baselines]

    # Clean styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    colors = {
        'monolithic': '#94a3b8',
        'self_consistency': '#64748b',
        'mact': '#3b82f6',
        'fixed_veccap': '#f59e0b',
        'confidence_router': '#a855f7',
        'gvoc': '#10b981',
        'oracle': '#059669'
    }

    # =========================================================================
    # Figure 1: Oracle Intervention Regret Comparison (Lower is Better)
    # =========================================================================
    plt.figure(figsize=(8, 4.5), dpi=300)
    bar_colors = [colors.get(b, '#475569') for b in baselines]
    bars = plt.bar(baselines, regrets, color=bar_colors, edgecolor='black', width=0.55, linewidth=0.8)
    plt.title("Oracle Intervention Regret (IR_oracle) across Routing Policies\n(Lower indicates closer alignment with empirical counterfactual optimum)", fontsize=11, fontweight='bold', pad=12)
    plt.ylabel("Empirical Value Regret (IR_oracle)", fontsize=10)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.1, f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    fig1_path = OUTPUT_DIR / "fig_pilot_regret_comparison.png"
    plt.savefig(fig1_path)
    plt.close()
    print(f"[Plot Generated] {fig1_path}")

    # =========================================================================
    # Figure 2: Accuracy vs. Compute Token Cost Frontier
    # =========================================================================
    plt.figure(figsize=(7, 4.8), dpi=300)
    for b, acc, tok in zip(baselines, accs, tokens):
        c = colors.get(b, '#475569')
        s = 140 if b in ['gvoc', 'oracle'] else 80
        marker = '*' if b == 'gvoc' else ('s' if b == 'oracle' else 'o')
        plt.scatter(tok, acc, color=c, s=s, marker=marker, edgecolors='black', label=b.upper(), zorder=4)
        plt.annotate(
            b.replace('_', ' ').title(),
            (tok, acc),
            textcoords="offset points",
            xytext=(0, 8),
            ha='center',
            fontsize=8,
            fontweight='bold' if b == 'gvoc' else 'normal'
        )

    plt.title("Empirical Accuracy vs. Test-Time Compute Frontier", fontsize=11, fontweight='bold', pad=12)
    plt.xlabel("Average Total Inference Tokens Spent", fontsize=10)
    plt.ylabel("Task Accuracy (%)", fontsize=10)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig2_path = OUTPUT_DIR / "fig_pilot_pareto_frontier.png"
    plt.savefig(fig2_path)
    plt.close()
    print(f"[Plot Generated] {fig2_path}")

    # =========================================================================
    # Figure 3: Action Distribution of GVoC Chosen Interventions
    # =========================================================================
    if action_counts:
        plt.figure(figsize=(6, 4), dpi=300)
        acts = list(action_counts.keys())
        counts = list(action_counts.values())
        palette = ['#10b981', '#3b82f6', '#f59e0b', '#ec4899', '#8b5cf6']
        plt.pie(counts, labels=acts, autopct='%1.1f%%', startangle=140, colors=palette[:len(acts)], wedgeprops={'edgecolor': 'black', 'linewidth': 0.8})
        plt.title("Distribution of GVoC Selected Interventions a*\n(Testing Action Diversity across Intermediate States)", fontsize=10, fontweight='bold')
        plt.tight_layout()
        fig3_path = OUTPUT_DIR / "fig_pilot_action_distribution.png"
        plt.savefig(fig3_path)
        plt.close()
        print(f"[Plot Generated] {fig3_path}")


if __name__ == "__main__":
    plot_pilot_results()
