"""
Fresh Comprehensive Diagnostic Benchmark Dataset for GVoC Evaluation.

Contains 40 realistic, diverse multimodal reasoning scenarios across:
1. DocVQA (10 instances) - Invoices, tax filings, legal agreements, forms
2. ChartQA (10 instances) - Bar charts, line graphs, scatter plots, trend lines
3. TableBench (10 instances) - Financial tables, multi-column spreadsheets, cross-cell operations
4. MMLongBench-Doc (10 instances) - Multi-page documents, cross-page referencing
"""

from typing import List, Dict, Any
from gvoc_engine.core.action_space import Action


def get_fresh_benchmark_dataset() -> List[Dict[str, Any]]:
    """Returns 40 diagnostic VDU benchmark instances with ground truth and A*."""
    dataset: List[Dict[str, Any]] = []

    # =========================================================================
    # 1. DocVQA (10 Instances)
    # =========================================================================
    docvqa_items = [
        {
            "id": "docvqa_01",
            "query": "What is the net invoice balance due before the 2% early payment discount?",
            "ground_truth": "$4,820.50",
            "initial_state": {"c_plan": 0.85, "c_exec": 0.80, "c_judg": 0.30, "c_consensus": 0.75, "d_layout": 0.42, "d_traj": 0.08, "d_evidence": 0.45, "d_answer": 0.10, "rho_visual": 0.60, "current_bbox": [110, 60, 190, 260], "current_answer": "$482.05"},
            "failure_type": "spatial_shift_30px",
            "effective_set": [Action.REGROUND]
        },
        {
            "id": "docvqa_02",
            "query": "State the governing jurisdiction cited in Section 14.2 of the indemnification clause.",
            "ground_truth": "State of Delaware",
            "initial_state": {"c_plan": 0.40, "c_exec": 0.85, "c_judg": 0.55, "c_consensus": 0.50, "d_layout": 0.08, "d_traj": 0.58, "d_evidence": 0.15, "d_answer": 0.20, "rho_visual": 0.75, "current_bbox": [320, 80, 420, 500], "current_answer": "State of New York"},
            "failure_type": "reading_order_inversion",
            "effective_set": [Action.REPLAN, Action.REGROUND]
        },
        {
            "id": "docvqa_03",
            "query": "What is the employer identification number (EIN) listed on Box B of Form W-2?",
            "ground_truth": "12-3456789",
            "initial_state": {"c_plan": 0.90, "c_exec": 0.35, "c_judg": 0.50, "c_consensus": 0.65, "d_layout": 0.05, "d_traj": 0.05, "d_evidence": 0.10, "d_answer": 0.60, "rho_visual": 0.50, "current_bbox": [90, 120, 140, 280], "current_answer": "12-3456788"},
            "failure_type": "ocr_noise",
            "effective_set": [Action.REEXEC, Action.REGROUND]
        },
        {
            "id": "docvqa_04",
            "query": "Confirm if the notary stamp carries an embossed seal date of October 2024.",
            "ground_truth": "Yes, certified October 14, 2024",
            "initial_state": {"c_plan": 0.80, "c_exec": 0.85, "c_judg": 0.72, "c_consensus": 0.58, "d_layout": 0.16, "d_traj": 0.10, "d_evidence": 0.18, "d_answer": 0.08, "rho_visual": 0.45, "current_bbox": [480, 300, 560, 480], "current_answer": "Yes, certified"},
            "failure_type": "spatial_shift_10px",
            "effective_set": [Action.VERIFY, Action.REGROUND]
        },
        {
            "id": "docvqa_05",
            "query": "What is the purchase order authorization number in the top right box?",
            "ground_truth": "PO-994182",
            "initial_state": {"c_plan": 0.94, "c_exec": 0.92, "c_judg": 0.95, "c_consensus": 0.96, "d_layout": 0.02, "d_traj": 0.03, "d_evidence": 0.04, "d_answer": 0.01, "rho_visual": 0.55, "current_bbox": [50, 400, 100, 520], "current_answer": "PO-994182"},
            "failure_type": "normal_correct_state",
            "effective_set": [Action.STOP]
        },
        {
            "id": "docvqa_06",
            "query": "Transcribe the smudged tracking barcode alphanumeric sequence.",
            "ground_truth": "[UNINTELLIGIBLE]",
            "initial_state": {"c_plan": 0.20, "c_exec": 0.15, "c_judg": 0.10, "c_consensus": 0.05, "d_layout": 0.88, "d_traj": 0.75, "d_evidence": 0.92, "d_answer": 0.85, "rho_visual": 0.90, "current_bbox": [520, 50, 580, 200], "current_answer": "TRK-??-99"},
            "failure_type": "unrecoverable_noise",
            "effective_set": [Action.ABSTAIN]
        },
        {
            "id": "docvqa_07",
            "query": "What is the freight charge itemized under bill of lading line 3?",
            "ground_truth": "$310.00",
            "initial_state": {"c_plan": 0.82, "c_exec": 0.40, "c_judg": 0.60, "c_consensus": 0.65, "d_layout": 0.08, "d_traj": 0.12, "d_evidence": 0.12, "d_answer": 0.55, "rho_visual": 0.60, "current_bbox": [250, 150, 320, 420], "current_answer": "$31.00"},
            "failure_type": "ocr_noise",
            "effective_set": [Action.REEXEC, Action.REGROUND]
        },
        {
            "id": "docvqa_08",
            "query": "Which entity is designated as the primary consignee in the header?",
            "ground_truth": "Apex Logistics International",
            "initial_state": {"c_plan": 0.50, "c_exec": 0.80, "c_judg": 0.52, "c_consensus": 0.50, "d_layout": 0.10, "d_traj": 0.45, "d_evidence": 0.15, "d_answer": 0.22, "rho_visual": 0.65, "current_bbox": [80, 100, 160, 350], "current_answer": "Apex Freight Hub"},
            "failure_type": "plan_step_deletion",
            "effective_set": [Action.REPLAN]
        },
        {
            "id": "docvqa_09",
            "query": "What is the total sales tax percentage applied to the subtotal?",
            "ground_truth": "8.25%",
            "initial_state": {"c_plan": 0.88, "c_exec": 0.82, "c_judg": 0.40, "c_consensus": 0.70, "d_layout": 0.35, "d_traj": 0.10, "d_evidence": 0.30, "d_answer": 0.12, "rho_visual": 0.55, "current_bbox": [380, 200, 440, 360], "current_answer": "8.5%"},
            "failure_type": "spatial_shift_20px",
            "effective_set": [Action.REGROUND]
        },
        {
            "id": "docvqa_10",
            "query": "Is the document countersigned by both managing partners?",
            "ground_truth": "Yes, signed by J. Doe and M. Smith",
            "initial_state": {"c_plan": 0.85, "c_exec": 0.80, "c_judg": 0.78, "c_consensus": 0.60, "d_layout": 0.14, "d_traj": 0.08, "d_evidence": 0.16, "d_answer": 0.08, "rho_visual": 0.50, "current_bbox": [510, 100, 580, 450], "current_answer": "Yes, signed"},
            "failure_type": "spatial_shift_10px",
            "effective_set": [Action.VERIFY, Action.REGROUND]
        }
    ]
    for x in docvqa_items:
        x["dataset"] = "docvqa"
        dataset.append(x)

    # =========================================================================
    # 2. ChartQA (10 Instances)
    # =========================================================================
    chartqa_items = [
        {
            "id": "chartqa_01",
            "query": "What is the value of the green bar representing APAC in Q3?",
            "ground_truth": "64.2M",
            "initial_state": {"c_plan": 0.90, "c_exec": 0.40, "c_judg": 0.55, "c_consensus": 0.70, "d_layout": 0.05, "d_traj": 0.08, "d_evidence": 0.12, "d_answer": 0.60, "rho_visual": 0.40, "current_bbox": [180, 250, 420, 380], "current_answer": "44.2M"},
            "failure_type": "ocr_noise",
            "effective_set": [Action.REEXEC, Action.REGROUND]
        },
        {
            "id": "chartqa_02",
            "query": "Does the 2026 forecast surpass the historical maximum reached in 2020?",
            "ground_truth": "No, it remains 4.8% lower",
            "initial_state": {"c_plan": 0.82, "c_exec": 0.45, "c_judg": 0.60, "c_consensus": 0.62, "d_layout": 0.08, "d_traj": 0.15, "d_evidence": 0.12, "d_answer": 0.65, "rho_visual": 0.45, "current_bbox": [150, 180, 360, 480], "current_answer": "Yes, it surpasses"},
            "failure_type": "arithmetic_swap",
            "effective_set": [Action.REEXEC, Action.VERIFY]
        },
        {
            "id": "chartqa_03",
            "query": "What is the category with the highest market share in the pie chart?",
            "ground_truth": "Consumer Electronics (38.5%)",
            "initial_state": {"c_plan": 0.92, "c_exec": 0.94, "c_judg": 0.95, "c_consensus": 0.97, "d_layout": 0.02, "d_traj": 0.02, "d_evidence": 0.03, "d_answer": 0.01, "rho_visual": 0.35, "current_bbox": [100, 150, 350, 400], "current_answer": "Consumer Electronics (38.5%)"},
            "failure_type": "normal_correct_state",
            "effective_set": [Action.STOP]
        },
        {
            "id": "chartqa_04",
            "query": "At what year does the dashed red line cross above the solid blue line?",
            "ground_truth": "2018",
            "initial_state": {"c_plan": 0.45, "c_exec": 0.80, "c_judg": 0.50, "c_consensus": 0.48, "d_layout": 0.10, "d_traj": 0.55, "d_evidence": 0.15, "d_answer": 0.25, "rho_visual": 0.50, "current_bbox": [160, 120, 380, 460], "current_answer": "2016"},
            "failure_type": "reading_order_inversion",
            "effective_set": [Action.REPLAN, Action.REGROUND]
        },
        {
            "id": "chartqa_05",
            "query": "What is the label of the outlier data point in the top right quadrant of the scatter plot?",
            "ground_truth": "Cluster 9A",
            "initial_state": {"c_plan": 0.85, "c_exec": 0.78, "c_judg": 0.32, "c_consensus": 0.70, "d_layout": 0.48, "d_traj": 0.10, "d_evidence": 0.42, "d_answer": 0.15, "rho_visual": 0.60, "current_bbox": [80, 380, 160, 480], "current_answer": "Cluster 8A"},
            "failure_type": "spatial_shift_30px",
            "effective_set": [Action.REGROUND]
        },
        {
            "id": "chartqa_06",
            "query": "Transcribe the legend key text occluded by the corrupted watermark.",
            "ground_truth": "[UNINTELLIGIBLE]",
            "initial_state": {"c_plan": 0.20, "c_exec": 0.18, "c_judg": 0.12, "c_consensus": 0.08, "d_layout": 0.85, "d_traj": 0.70, "d_evidence": 0.90, "d_answer": 0.82, "rho_visual": 0.85, "current_bbox": [420, 400, 480, 520], "current_answer": "Series ???"},
            "failure_type": "unrecoverable_noise",
            "effective_set": [Action.ABSTAIN]
        },
        {
            "id": "chartqa_07",
            "query": "Calculate the difference between the 2021 and 2023 export volumes.",
            "ground_truth": "18.5 thousand metric tons",
            "initial_state": {"c_plan": 0.88, "c_exec": 0.42, "c_judg": 0.58, "c_consensus": 0.60, "d_layout": 0.07, "d_traj": 0.10, "d_evidence": 0.10, "d_answer": 0.58, "rho_visual": 0.45, "current_bbox": [180, 150, 380, 440], "current_answer": "8.5 thousand metric tons"},
            "failure_type": "arithmetic_swap",
            "effective_set": [Action.REEXEC, Action.VERIFY]
        },
        {
            "id": "chartqa_08",
            "query": "Identify the trend trajectory for renewable energy adoption over 2015-2025.",
            "ground_truth": "Strictly monotonic growth",
            "initial_state": {"c_plan": 0.45, "c_exec": 0.85, "c_judg": 0.55, "c_consensus": 0.52, "d_layout": 0.09, "d_traj": 0.50, "d_evidence": 0.14, "d_answer": 0.20, "rho_visual": 0.40, "current_bbox": [150, 100, 350, 450], "current_answer": "Cyclical fluctuations"},
            "failure_type": "plan_step_deletion",
            "effective_set": [Action.REPLAN]
        },
        {
            "id": "chartqa_09",
            "query": "What is the y-axis percentage label at the upper gridline bound?",
            "ground_truth": "100%",
            "initial_state": {"c_plan": 0.90, "c_exec": 0.85, "c_judg": 0.76, "c_consensus": 0.65, "d_layout": 0.15, "d_traj": 0.08, "d_evidence": 0.18, "d_answer": 0.08, "rho_visual": 0.40, "current_bbox": [70, 50, 110, 100], "current_answer": "100%"},
            "failure_type": "spatial_shift_10px",
            "effective_set": [Action.VERIFY, Action.REGROUND]
        },
        {
            "id": "chartqa_10",
            "query": "What is the color encoding for the North American benchmark index?",
            "ground_truth": "Deep Amber",
            "initial_state": {"c_plan": 0.86, "c_exec": 0.80, "c_judg": 0.35, "c_consensus": 0.72, "d_layout": 0.38, "d_traj": 0.09, "d_evidence": 0.35, "d_answer": 0.12, "rho_visual": 0.50, "current_bbox": [400, 80, 450, 220], "current_answer": "Yellow"},
            "failure_type": "spatial_shift_20px",
            "effective_set": [Action.REGROUND]
        }
    ]
    for x in chartqa_items:
        x["dataset"] = "chartqa"
        dataset.append(x)

    # =========================================================================
    # 3. TableBench (10 Instances)
    # =========================================================================
    tablebench_items = [
        {
            "id": "tablebench_01",
            "query": "Compute the YoY operating profit margin change for the Automotive division.",
            "ground_truth": "+3.4 percentage points",
            "initial_state": {"c_plan": 0.42, "c_exec": 0.85, "c_judg": 0.58, "c_consensus": 0.52, "d_layout": 0.09, "d_traj": 0.54, "d_evidence": 0.14, "d_answer": 0.28, "rho_visual": 0.70, "current_bbox": [180, 80, 320, 520], "current_answer": "+1.2 percentage points"},
            "failure_type": "reading_order_inversion",
            "effective_set": [Action.REPLAN, Action.REGROUND]
        },
        {
            "id": "tablebench_02",
            "query": "What is the entry in row 5, column 4 of the consolidated balance sheet?",
            "ground_truth": "$184,200",
            "initial_state": {"c_plan": 0.85, "c_exec": 0.82, "c_judg": 0.32, "c_consensus": 0.75, "d_layout": 0.44, "d_traj": 0.08, "d_evidence": 0.40, "d_answer": 0.10, "rho_visual": 0.80, "current_bbox": [220, 280, 260, 360], "current_answer": "$148,200"},
            "failure_type": "spatial_shift_30px",
            "effective_set": [Action.REGROUND]
        },
        {
            "id": "tablebench_03",
            "query": "Sum the Q1 and Q2 research and development expenditures in row 3.",
            "ground_truth": "$52.8M",
            "initial_state": {"c_plan": 0.88, "c_exec": 0.40, "c_judg": 0.56, "c_consensus": 0.62, "d_layout": 0.06, "d_traj": 0.10, "d_evidence": 0.12, "d_answer": 0.62, "rho_visual": 0.65, "current_bbox": [140, 160, 200, 380], "current_answer": "$42.8M"},
            "failure_type": "arithmetic_swap",
            "effective_set": [Action.REEXEC, Action.VERIFY]
        },
        {
            "id": "tablebench_04",
            "query": "What is the reported effective tax rate for FY2023 in the footnote summary table?",
            "ground_truth": "19.5%",
            "initial_state": {"c_plan": 0.94, "c_exec": 0.93, "c_judg": 0.96, "c_consensus": 0.96, "d_layout": 0.02, "d_traj": 0.02, "d_evidence": 0.03, "d_answer": 0.01, "rho_visual": 0.65, "current_bbox": [380, 200, 440, 320], "current_answer": "19.5%"},
            "failure_type": "normal_correct_state",
            "effective_set": [Action.STOP]
        },
        {
            "id": "tablebench_05",
            "query": "Extract the cell value at row 8, column 2 where column headers are partially clipped.",
            "ground_truth": "748.2",
            "initial_state": {"c_plan": 0.86, "c_exec": 0.80, "c_judg": 0.38, "c_consensus": 0.70, "d_layout": 0.36, "d_traj": 0.10, "d_evidence": 0.32, "d_answer": 0.12, "rho_visual": 0.75, "current_bbox": [320, 120, 370, 200], "current_answer": "74.82"},
            "failure_type": "spatial_shift_20px",
            "effective_set": [Action.REGROUND]
        },
        {
            "id": "tablebench_06",
            "query": "Transcribe the degraded superscript note attached to row 12.",
            "ground_truth": "[UNINTELLIGIBLE]",
            "initial_state": {"c_plan": 0.18, "c_exec": 0.15, "c_judg": 0.10, "c_consensus": 0.06, "d_layout": 0.90, "d_traj": 0.72, "d_evidence": 0.94, "d_answer": 0.88, "rho_visual": 0.88, "current_bbox": [480, 350, 520, 410], "current_answer": "Note ???"},
            "failure_type": "unrecoverable_noise",
            "effective_set": [Action.ABSTAIN]
        },
        {
            "id": "tablebench_07",
            "query": "Which country generated the highest per-capita GDP in the European regional table?",
            "ground_truth": "Luxembourg ($135,680)",
            "initial_state": {"c_plan": 0.48, "c_exec": 0.82, "c_judg": 0.54, "c_consensus": 0.50, "d_layout": 0.10, "d_traj": 0.48, "d_evidence": 0.15, "d_answer": 0.24, "rho_visual": 0.70, "current_bbox": [150, 80, 320, 450], "current_answer": "Switzerland ($93,450)"},
            "failure_type": "plan_step_deletion",
            "effective_set": [Action.REPLAN]
        },
        {
            "id": "tablebench_08",
            "query": "What is the numerical code listed in the SKU column for item 14?",
            "ground_truth": "SKU-88219-B",
            "initial_state": {"c_plan": 0.90, "c_exec": 0.38, "c_judg": 0.52, "c_consensus": 0.66, "d_layout": 0.06, "d_traj": 0.06, "d_evidence": 0.12, "d_answer": 0.58, "rho_visual": 0.65, "current_bbox": [390, 80, 430, 200], "current_answer": "SKU-88219-8"},
            "failure_type": "ocr_noise",
            "effective_set": [Action.REEXEC, Action.REGROUND]
        },
        {
            "id": "tablebench_09",
            "query": "Verify whether the total assets row exactly balances total liabilities plus equity.",
            "ground_truth": "Yes, both equal $1,240,500",
            "initial_state": {"c_plan": 0.84, "c_exec": 0.82, "c_judg": 0.75, "c_consensus": 0.62, "d_layout": 0.15, "d_traj": 0.08, "d_evidence": 0.16, "d_answer": 0.08, "rho_visual": 0.65, "current_bbox": [440, 100, 500, 480], "current_answer": "Yes, balanced"},
            "failure_type": "spatial_shift_10px",
            "effective_set": [Action.VERIFY, Action.REGROUND]
        },
        {
            "id": "tablebench_10",
            "query": "Calculate the compound annual growth rate (CAGR) over the 5-year table period.",
            "ground_truth": "7.8%",
            "initial_state": {"c_plan": 0.85, "c_exec": 0.44, "c_judg": 0.58, "c_consensus": 0.60, "d_layout": 0.07, "d_traj": 0.12, "d_evidence": 0.10, "d_answer": 0.60, "rho_visual": 0.60, "current_bbox": [200, 150, 360, 450], "current_answer": "5.4%"},
            "failure_type": "arithmetic_swap",
            "effective_set": [Action.REEXEC, Action.VERIFY]
        }
    ]
    for x in tablebench_items:
        x["dataset"] = "tablebench"
        dataset.append(x)

    # =========================================================================
    # 4. MMLongBench-Doc (10 Instances)
    # =========================================================================
    mmlong_items = [
        {
            "id": "mmlong_01",
            "query": "Locate the auditor opinion letter on Page 42 and state whether it is unqualified.",
            "ground_truth": "Unqualified opinion issued by KPMG",
            "initial_state": {"c_plan": 0.42, "c_exec": 0.82, "c_judg": 0.52, "c_consensus": 0.48, "d_layout": 0.10, "d_traj": 0.58, "d_evidence": 0.16, "d_answer": 0.26, "rho_visual": 0.65, "current_bbox": [100, 100, 300, 500], "current_answer": "Qualified with explanatory paragraph"},
            "failure_type": "reading_order_inversion",
            "effective_set": [Action.REPLAN, Action.REGROUND]
        },
        {
            "id": "mmlong_02",
            "query": "What is the cross-referenced section in footnote 18 linking to the patent appendix?",
            "ground_truth": "Appendix D-2 (Page 88)",
            "initial_state": {"c_plan": 0.86, "c_exec": 0.80, "c_judg": 0.32, "c_consensus": 0.72, "d_layout": 0.46, "d_traj": 0.08, "d_evidence": 0.42, "d_answer": 0.12, "rho_visual": 0.70, "current_bbox": [450, 80, 520, 320], "current_answer": "Appendix D-1"},
            "failure_type": "spatial_shift_30px",
            "effective_set": [Action.REGROUND]
        },
        {
            "id": "mmlong_03",
            "query": "Sum the total severance liabilities reported on Page 14 and Page 31.",
            "ground_truth": "$14.6M",
            "initial_state": {"c_plan": 0.88, "c_exec": 0.42, "c_judg": 0.58, "c_consensus": 0.62, "d_layout": 0.06, "d_traj": 0.10, "d_evidence": 0.10, "d_answer": 0.60, "rho_visual": 0.60, "current_bbox": [200, 100, 350, 450], "current_answer": "$11.2M"},
            "failure_type": "arithmetic_swap",
            "effective_set": [Action.REEXEC, Action.VERIFY]
        },
        {
            "id": "mmlong_04",
            "query": "What is the registered corporate address cited on the title page of the 10-K filing?",
            "ground_truth": "100 Innovation Way, Suite 400, Cambridge, MA",
            "initial_state": {"c_plan": 0.95, "c_exec": 0.94, "c_judg": 0.96, "c_consensus": 0.98, "d_layout": 0.02, "d_traj": 0.02, "d_evidence": 0.03, "d_answer": 0.01, "rho_visual": 0.50, "current_bbox": [120, 150, 200, 480], "current_answer": "100 Innovation Way, Suite 400, Cambridge, MA"},
            "failure_type": "normal_correct_state",
            "effective_set": [Action.STOP]
        },
        {
            "id": "mmlong_05",
            "query": "Identify the primary environmental risk factor added in the 2025 amendment on Page 19.",
            "ground_truth": "Extended wildfire liability under SB-901",
            "initial_state": {"c_plan": 0.46, "c_exec": 0.82, "c_judg": 0.52, "c_consensus": 0.50, "d_layout": 0.10, "d_traj": 0.50, "d_evidence": 0.14, "d_answer": 0.22, "rho_visual": 0.70, "current_bbox": [250, 80, 400, 520], "current_answer": "Drought-related irrigation cutbacks"},
            "failure_type": "plan_step_deletion",
            "effective_set": [Action.REPLAN]
        },
        {
            "id": "mmlong_06",
            "query": "Transcribe the illegible archival stamp on the historical patent exhibit on Page 112.",
            "ground_truth": "[UNINTELLIGIBLE]",
            "initial_state": {"c_plan": 0.18, "c_exec": 0.14, "c_judg": 0.10, "c_consensus": 0.05, "d_layout": 0.92, "d_traj": 0.78, "d_evidence": 0.95, "d_answer": 0.90, "rho_visual": 0.85, "current_bbox": [520, 380, 580, 480], "current_answer": "Stamp # ??-1940"},
            "failure_type": "unrecoverable_noise",
            "effective_set": [Action.ABSTAIN]
        },
        {
            "id": "mmlong_07",
            "query": "What is the serial number of the industrial chiller listed in Appendix Table C-4 on Page 67?",
            "ground_truth": "CHL-9042-XR",
            "initial_state": {"c_plan": 0.90, "c_exec": 0.36, "c_judg": 0.50, "c_consensus": 0.65, "d_layout": 0.06, "d_traj": 0.05, "d_evidence": 0.10, "d_answer": 0.60, "rho_visual": 0.65, "current_bbox": [310, 100, 360, 240], "current_answer": "CHL-9042-XB"},
            "failure_type": "ocr_noise",
            "effective_set": [Action.REEXEC, Action.REGROUND]
        },
        {
            "id": "mmlong_08",
            "query": "Verify whether the executive compensation table complies with the Dodd-Frank disclosure on Page 54.",
            "ground_truth": "Yes, pay-versus-performance ratio disclosed",
            "initial_state": {"c_plan": 0.82, "c_exec": 0.80, "c_judg": 0.74, "c_consensus": 0.60, "d_layout": 0.16, "d_traj": 0.08, "d_evidence": 0.18, "d_answer": 0.08, "rho_visual": 0.60, "current_bbox": [380, 120, 460, 450], "current_answer": "Yes, compliant"},
            "failure_type": "spatial_shift_10px",
            "effective_set": [Action.VERIFY, Action.REGROUND]
        },
        {
            "id": "mmlong_09",
            "query": "Extract the pension discount rate specified in Note 12 on Page 73.",
            "ground_truth": "4.65%",
            "initial_state": {"c_plan": 0.86, "c_exec": 0.80, "c_judg": 0.36, "c_consensus": 0.70, "d_layout": 0.38, "d_traj": 0.10, "d_evidence": 0.34, "d_answer": 0.12, "rho_visual": 0.65, "current_bbox": [280, 150, 340, 300], "current_answer": "4.85%"},
            "failure_type": "spatial_shift_20px",
            "effective_set": [Action.REGROUND]
        },
        {
            "id": "mmlong_10",
            "query": "Calculate the net change in operating cash flow between the summary on Page 8 and Page 34.",
            "ground_truth": "+$28.4M",
            "initial_state": {"c_plan": 0.86, "c_exec": 0.42, "c_judg": 0.58, "c_consensus": 0.62, "d_layout": 0.08, "d_traj": 0.12, "d_evidence": 0.10, "d_answer": 0.58, "rho_visual": 0.60, "current_bbox": [180, 120, 340, 480], "current_answer": "+$18.4M"},
            "failure_type": "arithmetic_swap",
            "effective_set": [Action.REEXEC, Action.VERIFY]
        }
    ]
    for x in mmlong_items:
        x["dataset"] = "mmlongbench"
        dataset.append(x)

    return dataset
