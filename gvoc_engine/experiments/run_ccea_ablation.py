"""
Experimental Evaluation and Central Ablation:
Context-Preserving, Cost-Aware Visual Evidence Acquisition (CCEA).

Ablation comparison across 5 conditions:
1. Full_Overview: Monolithic inference on full image.
2. Target_Only_Crop: Tight crop around target evidence (dropping context).
3. CCEA_Context_Preserving: Crop covering target + required headers, labels, and units.
4. CCEA_Cost_Aware_Controller: Adaptive decision policy balancing expected gain vs compute cost.
5. Fixed_Center_Crop: Uniform center baseline (50% crop).

Integrates:
- Strict normalized EM, token F1, and unit-sensitive scoring.
- Ledger A: Actual Experimental Usage (granular tracking: planned, attempted, successful, failed, retried, blocked).
- Ledger B: Deployed-Policy Cost (simulating the true deployment cost of the C4 policy).
- Conservative pre-flight BudgetGuard: blocks requests before transmission and halts incomplete runs.
- Zero ground-truth leakage into the controller.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from discovery.live_validation.scoring import (
    score_vdu_answer,
    score_vdu_strict,
    score_vdu_unit_sensitive,
    score_vdu_comprehensive
)
from gvoc_engine.core.state import GroundedState
from gvoc_engine.core.action_space import Action
from gvoc_engine.core.budget_guard import BudgetGuard, BudgetExceededError
from gvoc_engine.interventions.context_preservation import (
    Box,
    ContextKind,
    EvidenceRegion,
    assess_crop,
    propose_context_preserving_crop,
    expand_box,
    validate_box,
)
from gvoc_engine.interventions.ccea_intervention import (
    CCEAIntervention,
    CCEAResult,
    extract_and_save_crop,
)
from gvoc_engine.controllers.ccea_router import CCEACostAwareRouter
from gvoc_engine.interventions.auto_locator import AutomaticEvidenceLocator


class MockAblationClient:
    """Deterministic client for offline testing and plumbing validation."""
    def __init__(self, budget_guard: Optional[BudgetGuard] = None):
        self.calls = 0
        self.budget_guard = budget_guard

    def call_vlm(self, prompt: str, image_path: str, max_tokens: int = 100):
        if self.budget_guard:
            allowed, reason = self.budget_guard.check_pre_flight(
                has_image=(image_path is not None and image_path != ""),
                max_tokens=max_tokens,
                prompt_char_len=len(prompt)
            )
            if not allowed:
                raise BudgetExceededError(reason)
            self.budget_guard.record_attempt(prompt_tokens=120, completion_tokens=25, total_tokens=145)

        self.calls += 1
        p_tok, c_tok, lat = 120, 25, 0.08
        if "without any headers or labels" in prompt or "Target_Only" in prompt:
            return "42.0", p_tok, c_tok, lat
        return "1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036", p_tok, c_tok, lat

    def call(self, prompt: str, image_path: Optional[str] = None, max_tokens: int = 100):
        text, p, c, lat = self.call_vlm(prompt, image_path or "", max_tokens=max_tokens)
        return {
            "status": "success",
            "text": text,
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": p + c,
            "latency_sec": lat,
            "is_live_call": False
        }


def load_curated_eval_instances(
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Loads genuine document instances from discovery/live_validation/pilot_dataset/instances.json."""
    instances_path = Path("discovery/live_validation/pilot_dataset/instances.json")
    if not instances_path.exists():
        raise FileNotFoundError(f"Dataset not found at {instances_path}")
    
    with open(instances_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if end_idx is not None:
        slice_data = data[start_idx:end_idx]
    elif limit is not None:
        slice_data = data[start_idx:start_idx + limit]
    else:
        slice_data = data[start_idx:]

    valid_items = []
    for item in slice_data:
        img_p = Path(item["image_path"])
        if img_p.exists():
            valid_items.append(item)
    return valid_items


def create_synthetic_evidence_region(width: int, height: int, inst_id: str) -> EvidenceRegion:
    """Constructs documented target region and required semantic context boxes for ablation testing."""
    t_y0, t_x0 = int(0.40 * height), int(0.35 * width)
    t_y1, t_x1 = int(0.48 * height), int(0.55 * width)
    target_box = (t_y0, t_x0, t_y1, t_x1)

    col_header = (int(0.20 * height), int(0.35 * width), int(0.28 * height), int(0.55 * width))
    row_label  = (int(0.40 * height), int(0.10 * width), int(0.48 * height), int(0.32 * width))
    unit_box   = (int(0.12 * height), int(0.10 * width), int(0.18 * height), int(0.30 * width))
    footnote   = (int(0.85 * height), int(0.10 * width), int(0.92 * height), int(0.60 * width))

    return EvidenceRegion(
        target_box=target_box,
        required_context=(
            (ContextKind.COLUMN_LABEL, col_header),
            (ContextKind.ROW_LABEL, row_label),
            (ContextKind.UNIT, unit_box),
            (ContextKind.FOOTNOTE, footnote),
        ),
        evidence_id=f"evidence_{inst_id}"
    )


def get_evidence_region_for_instance(inst_id: str, width: int, height: int) -> EvidenceRegion:
    """Loads verified grounded spatial annotations if available, otherwise falls back to synthetic."""
    verified_path = Path("discovery/live_validation/pilot_dataset/verified_spatial_annotations_10_21.json")
    if verified_path.exists():
        with open(verified_path, "r", encoding="utf-8") as f:
            v_data = json.load(f)
        if inst_id in v_data:
            item = v_data[inst_id]
            t_box = tuple(item["target_box_px"])
            c_box = tuple(item["context_box_px"])
            col_header = (c_box[0], t_box[1], max(c_box[0] + 5, t_box[0]), t_box[3])
            row_label = (t_box[0], c_box[1], t_box[2], max(c_box[1] + 5, t_box[1]))
            return EvidenceRegion(
                target_box=t_box,
                required_context=(
                    (ContextKind.COLUMN_LABEL, col_header),
                    (ContextKind.ROW_LABEL, row_label),
                ),
                evidence_id=f"evidence_{inst_id}"
            )
    return create_synthetic_evidence_region(width, height, inst_id)


def extract_overview_state(
    ans_overview: str,
    width: int,
    height: int,
    prompt_tokens: int,
    completion_tokens: int
) -> GroundedState:
    """
    Extracts intermediate state features from overview prediction telemetry without ground-truth leakage.
    - d_layout: estimated from image area / aspect ratio complexity.
    - d_evidence: estimated from answer verbosity, brevity, and ambiguity indicators.
    - c_plan, c_exec, c_judg, c_consensus: derived from completion characteristics.
    """
    ans_clean = (ans_overview or "").strip().lower()
    hedging_markers = ["unclear", "cannot determine", "not visible", "unreadable", "cut off"]
    is_hedging = any(m in ans_clean for m in hedging_markers)
    
    if is_hedging or len(ans_clean) == 0:
        d_evidence = 0.40
        c_judg = 0.40
    elif len(ans_clean) < 3:
        d_evidence = 0.25
        c_judg = 0.65
    else:
        d_evidence = 0.18
        c_judg = 0.75

    pixel_count = width * height
    if pixel_count >= 3_000_000:
        d_layout = 0.28
    elif pixel_count >= 1_500_000:
        d_layout = 0.22
    else:
        d_layout = 0.15

    return GroundedState(
        c_plan=0.75,
        c_exec=0.70,
        c_judg=c_judg,
        c_consensus=0.80,
        d_layout=d_layout,
        d_evidence=d_evidence,
        current_answer=ans_overview,
        b_remaining=2000,
        n_tokens_spent=prompt_tokens + completion_tokens
    )


def run_ccea_ablation(
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    num_samples: Optional[int] = None,
    use_live: bool = False,
    budget_guard: Optional[BudgetGuard] = None,
    records_out_path: Optional[str] = None,
    held_out_out_path: Optional[str] = None,
    auto_localization: bool = False
):
    print("=" * 80)
    print("RUNNING CENTRAL ABLATION: CONTEXT-PRESERVING VS. UNCONSTRAINED ACQUISITION")
    print(f"Slice: [{start_idx}:{end_idx}] | Mode: {'LIVE API' if use_live else 'OFFLINE MOCK / PIPELINE VALIDATION'} | Auto-Locator: {auto_localization}")
    print("=" * 80 + "\n")

    instances = load_curated_eval_instances(start_idx=start_idx, end_idx=end_idx, limit=num_samples)
    print(f"Loaded {len(instances)} instances with valid image assets.\n")

    if budget_guard is None:
        if len(instances) == 1:
            budget_guard = BudgetGuard(
                max_budget_usd=0.005,
                operational_budget_usd=0.003,
                max_total_tokens=30000,
                operational_token_limit=18000,
                planned_requests=5
            )
        else:
            budget_guard = BudgetGuard(
                max_budget_usd=0.015,
                operational_budget_usd=0.010,
                max_total_tokens=75000,
                operational_token_limit=60000,
                planned_requests=len(instances) * 5,
                max_consecutive_missing_metadata=3
            )

    if use_live:
        from discovery.live_validation.live_client import LiveVLMClient
        client = LiveVLMClient("mistral", budget_guard=budget_guard)
    else:
        client = MockAblationClient(budget_guard=budget_guard)

    intervention = CCEAIntervention(output_dir="discovery/live_validation/confirmatory_experiment/transformed_images")
    router = CCEACostAwareRouter(kappa=0.40, lambda_cost=0.25, tau_safe=0.45, tau_clean=0.75)
    auto_locator = AutomaticEvidenceLocator(enable_vlm_grounding=False) if auto_localization else None

    conditions = [
        "1_full_overview",
        "2_target_only_crop",
        "3_ccea_context_preserving",
        "4_ccea_cost_aware_controller",
        "5_fixed_center_crop"
    ]

    stats = {
        cond: {
            "correct_count": 0,
            "unit_correct_count": 0,
            "total_f1": 0.0,
            "total_tokens": 0,
            "total_latency": 0.0,
            "context_retained_count": 0,
            "context_lost_count": 0,
            "completed_count": 0,
            "blocked_count": 0,
            "lost_by_kind": {k.value: 0 for k in ContextKind},
        }
        for cond in conditions
    }

    controller_decisions = {"stop": 0, "acquire": 0, "abstain": 0}
    all_records = []
    is_complete = True
    halt_reason = None

    # Ledger A: Actual Experimental Usage (granular tracking)
    planned_requests = len(instances) * 5
    ledger_a = {
        "planned_requests": planned_requests,
        "attempted_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "retried_requests": 0,
        "blocked_requests": 0,
        "total_requests": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "requests_by_condition": {cond: 0 for cond in conditions}
    }

    # Ledger B: Deployed Policy Usage (simulating C4 in standalone production)
    ledger_b = {
        "deployed_requests": 0,
        "deployed_tokens": 0,
        "deployed_latency": 0.0,
        "deployed_cost_usd": 0.0,
        "baseline_c3_requests": 0,
        "baseline_c3_tokens": 0,
        "baseline_c3_cost_usd": 0.0,
    }

    for idx, inst in enumerate(instances):
        if not is_complete:
            break

        inst_id = inst["instance_id"]
        q = inst["query"]
        gt = inst["ground_truth"]
        img_path = inst["image_path"]
        with Image.open(img_path) as img:
            w, h = img.size

        if auto_localization and auto_locator is not None:
            sp_path = Path("discovery/live_validation/pilot_dataset/spatial_annotations_10_21.json")
            sp_map = {}
            if sp_path.exists():
                with open(sp_path, "r", encoding="utf-8") as f:
                    sp_list = json.load(f)
                sp_map = {x["instance_id"]: x for x in sp_list}
            sp_item = sp_map.get(inst_id, {})
            raw_words = sp_item.get("words", [])
            raw_boxes = sp_item.get("bounding_boxes", [])
            ocr_tokens = []
            for word, b in zip(raw_words, raw_boxes):
                if not b or len(b) != 4:
                    continue
                bx0 = int(min(b[0], b[2]) / 1000.0 * w)
                bx1 = int(max(b[0], b[2]) / 1000.0 * w)
                by0 = int(min(b[1], b[3]) / 1000.0 * h)
                by1 = int(max(b[1], b[3]) / 1000.0 * h)
                if by1 - by0 < 15: by1 = min(h, by0 + 20)
                if bx1 - bx0 < 10: bx1 = min(w, bx0 + 15)
                ocr_tokens.append({"text": str(word), "box": (by0, bx0, by1, bx1)})
            
            loc_res = auto_locator.locate(
                image_path=img_path,
                query=q,
                image_width=w,
                image_height=h,
                client=None,
                ocr_tokens=ocr_tokens,
                prefer_context_preserving=True
            )
            ctx_tuples = tuple((ContextKind.ROW_LABEL, cb) for cb in loc_res.context_boxes)
            evidence = EvidenceRegion(
                target_box=loc_res.target_box or (0, 0, h, w),
                required_context=ctx_tuples,
                evidence_id=f"auto_{inst_id}"
            )
        else:
            evidence = get_evidence_region_for_instance(inst_id, w, h)
        print(f"[{idx+1}/{len(instances)}] Instance {inst_id} ({w}x{h}): {q[:55]}...")

        inst_record = {
            "instance_id": inst_id,
            "dataset": inst.get("dataset", "docvqa"),
            "query": q,
            "ground_truth": gt,
            "image_dimensions": [w, h],
            "target_box": list(evidence.target_box) if evidence.target_box else None,
            "conditions": {}
        }

        # -------------------------------------------------------------
        # Condition 1: Full_Overview (No crop)
        # -------------------------------------------------------------
        p_full = f"Answer concisely based on the visual document:\nQuestion: {q}\nAnswer:"
        t0 = time.time()
        try:
            if hasattr(client, "call_vlm"):
                ans_full, p_tok, c_tok, lat = client.call_vlm(p_full, img_path, max_tokens=60)
            else:
                res_dict = client.call(p_full, image_path=img_path, max_tokens=60)
                ans_full = res_dict.get("text", "")
                p_tok = res_dict.get("prompt_tokens") if res_dict.get("prompt_tokens") is not None else 2800
                c_tok = res_dict.get("completion_tokens") if res_dict.get("completion_tokens") is not None else 60
                lat = res_dict.get("latency_sec") or 0.0

            ledger_a["attempted_requests"] += 1
            ledger_a["successful_requests"] += 1
            ledger_a["total_requests"] += 1
            ledger_a["requests_by_condition"]["1_full_overview"] += 1
            ledger_a["total_tokens"] += (p_tok + c_tok)
            ledger_a["total_cost_usd"] += ((p_tok + c_tok) / 1000000.0) * 0.15

            sc_full = score_vdu_comprehensive(ans_full, gt)
            stats["1_full_overview"]["correct_count"] += sc_full["strict_em"]
            stats["1_full_overview"]["unit_correct_count"] += sc_full["unit_sensitive_em"]
            stats["1_full_overview"]["total_f1"] += sc_full["token_f1"]
            stats["1_full_overview"]["total_tokens"] += (p_tok + c_tok)
            stats["1_full_overview"]["total_latency"] += lat
            stats["1_full_overview"]["context_retained_count"] += 1
            stats["1_full_overview"]["completed_count"] += 1

            inst_record["conditions"]["1_full_overview"] = {
                "status": "SUCCESS",
                "prediction": ans_full,
                "strict_em": sc_full["strict_em"],
                "unit_sensitive_em": sc_full["unit_sensitive_em"],
                "f1": sc_full["token_f1"],
                "tokens": p_tok + c_tok,
                "latency": round(lat, 4),
                "cost_usd": round((p_tok + c_tok) * 0.00015 / 1000.0, 6)
            }
        except BudgetExceededError as be:
            is_complete = False
            halt_reason = str(be)
            ledger_a["blocked_requests"] += 1
            stats["1_full_overview"]["blocked_count"] += 1
            inst_record["conditions"]["1_full_overview"] = {
                "status": "BLOCKED_BUDGET_EXCEEDED",
                "prediction": "BLOCKED_BUDGET_GUARD",
                "strict_em": None,
                "unit_sensitive_em": None,
                "f1": 0.0,
                "tokens": 0,
                "latency": 0.0,
                "cost_usd": 0.0
            }
            all_records.append(inst_record)
            break
        except Exception as e:
            is_complete = False
            halt_reason = f"Execution error in C1: {e}"
            ledger_a["failed_requests"] += 1
            inst_record["conditions"]["1_full_overview"] = {
                "status": "ERROR",
                "prediction": "ERROR",
                "strict_em": None,
                "unit_sensitive_em": None,
                "f1": 0.0,
                "tokens": 0,
                "latency": 0.0,
                "cost_usd": 0.0
            }
            all_records.append(inst_record)
            break

        # -------------------------------------------------------------
        # Condition 2: Target_Only_Crop (Oracle-Localized Ablation)
        # -------------------------------------------------------------
        try:
            res_target_only = intervention.execute(
                image_path=img_path,
                query=q,
                client=client,
                evidence_region=evidence,
                initial_answer=ans_full,
                preserve_context=False
            )
            ledger_a["attempted_requests"] += 1
            ledger_a["successful_requests"] += 1
            ledger_a["total_requests"] += 1
            ledger_a["requests_by_condition"]["2_target_only_crop"] += 1
            ledger_a["total_tokens"] += res_target_only.total_tokens
            ledger_a["total_cost_usd"] += res_target_only.cost_usd

            sc_tgt = score_vdu_comprehensive(res_target_only.final_answer, gt)
            stats["2_target_only_crop"]["correct_count"] += sc_tgt["strict_em"]
            stats["2_target_only_crop"]["unit_correct_count"] += sc_tgt["unit_sensitive_em"]
            stats["2_target_only_crop"]["total_f1"] += sc_tgt["token_f1"]
            stats["2_target_only_crop"]["total_tokens"] += (p_tok + c_tok + res_target_only.total_tokens)
            stats["2_target_only_crop"]["total_latency"] += (lat + res_target_only.latency)
            stats["2_target_only_crop"]["completed_count"] += 1

            if res_target_only.assessment and res_target_only.assessment.valid:
                stats["2_target_only_crop"]["context_retained_count"] += 1
            else:
                stats["2_target_only_crop"]["context_lost_count"] += 1
                if res_target_only.assessment:
                    for missing_k in res_target_only.assessment.missing_context:
                        stats["2_target_only_crop"]["lost_by_kind"][missing_k.value] += 1

            inst_record["conditions"]["2_target_only_crop"] = {
                "status": "SUCCESS",
                "prediction": res_target_only.final_answer,
                "crop_box": list(res_target_only.crop_box) if res_target_only.crop_box else None,
                "crop_path": res_target_only.crop_path,
                "retained_context": [c.value if hasattr(c, "value") else str(c) for c in res_target_only.retained_context],
                "missing_context": [c.value if hasattr(c, "value") else str(c) for c in res_target_only.missing_context],
                "strict_em": sc_tgt["strict_em"],
                "unit_sensitive_em": sc_tgt["unit_sensitive_em"],
                "f1": sc_tgt["token_f1"],
                "tokens": p_tok + c_tok + res_target_only.total_tokens,
                "latency": round(lat + res_target_only.latency, 4),
                "cost_usd": round(res_target_only.cost_usd, 6)
            }
        except BudgetExceededError as be:
            is_complete = False
            halt_reason = str(be)
            ledger_a["blocked_requests"] += 1
            stats["2_target_only_crop"]["blocked_count"] += 1
            inst_record["conditions"]["2_target_only_crop"] = {
                "status": "BLOCKED_BUDGET_EXCEEDED",
                "prediction": "BLOCKED_BUDGET_GUARD",
                "strict_em": None,
                "unit_sensitive_em": None,
                "f1": 0.0,
                "tokens": 0,
                "latency": 0.0,
                "cost_usd": 0.0
            }
            all_records.append(inst_record)
            break

        # -------------------------------------------------------------
        # Condition 3: CCEA_Context_Preserving (Oracle-Localized Context Retention)
        # -------------------------------------------------------------
        try:
            res_ccea = intervention.execute(
                image_path=img_path,
                query=q,
                client=client,
                evidence_region=evidence,
                initial_answer=ans_full,
                preserve_context=True
            )
            ledger_a["attempted_requests"] += 1
            ledger_a["successful_requests"] += 1
            ledger_a["total_requests"] += 1
            ledger_a["requests_by_condition"]["3_ccea_context_preserving"] += 1
            ledger_a["total_tokens"] += res_ccea.total_tokens
            ledger_a["total_cost_usd"] += res_ccea.cost_usd

            sc_ccea = score_vdu_comprehensive(res_ccea.final_answer, gt)
            stats["3_ccea_context_preserving"]["correct_count"] += sc_ccea["strict_em"]
            stats["3_ccea_context_preserving"]["unit_correct_count"] += sc_ccea["unit_sensitive_em"]
            stats["3_ccea_context_preserving"]["total_f1"] += sc_ccea["token_f1"]
            stats["3_ccea_context_preserving"]["total_tokens"] += (p_tok + c_tok + res_ccea.total_tokens)
            stats["3_ccea_context_preserving"]["total_latency"] += (lat + res_ccea.latency)
            stats["3_ccea_context_preserving"]["completed_count"] += 1

            if res_ccea.assessment and res_ccea.assessment.valid:
                stats["3_ccea_context_preserving"]["context_retained_count"] += 1
            else:
                stats["3_ccea_context_preserving"]["context_lost_count"] += 1

            ledger_b["baseline_c3_requests"] += 2
            ledger_b["baseline_c3_tokens"] += (p_tok + c_tok + res_ccea.total_tokens)
            ledger_b["baseline_c3_cost_usd"] += (((p_tok + c_tok + res_ccea.total_tokens) / 1000000.0) * 0.15)

            inst_record["conditions"]["3_ccea_context_preserving"] = {
                "status": "SUCCESS",
                "prediction": res_ccea.final_answer,
                "crop_box": list(res_ccea.crop_box) if res_ccea.crop_box else None,
                "crop_path": res_ccea.crop_path,
                "retained_context": [c.value if hasattr(c, "value") else str(c) for c in res_ccea.retained_context],
                "missing_context": [c.value if hasattr(c, "value") else str(c) for c in res_ccea.missing_context],
                "strict_em": sc_ccea["strict_em"],
                "unit_sensitive_em": sc_ccea["unit_sensitive_em"],
                "f1": sc_ccea["token_f1"],
                "tokens": p_tok + c_tok + res_ccea.total_tokens,
                "latency": round(lat + res_ccea.latency, 4),
                "cost_usd": round(res_ccea.cost_usd, 6)
            }
        except BudgetExceededError as be:
            is_complete = False
            halt_reason = str(be)
            ledger_a["blocked_requests"] += 1
            stats["3_ccea_context_preserving"]["blocked_count"] += 1
            inst_record["conditions"]["3_ccea_context_preserving"] = {
                "status": "BLOCKED_BUDGET_EXCEEDED",
                "prediction": "BLOCKED_BUDGET_GUARD",
                "strict_em": None,
                "unit_sensitive_em": None,
                "f1": 0.0,
                "tokens": 0,
                "latency": 0.0,
                "cost_usd": 0.0
            }
            all_records.append(inst_record)
            break

        # -------------------------------------------------------------
        # Condition 4: CCEA_Cost_Aware_Controller (Live Sequential Policy)
        # -------------------------------------------------------------
        state = extract_overview_state(ans_full, w, h, p_tok, c_tok)
        chosen_action, meta = router.select_action(
            state,
            remaining_budget=2000,
            evidence_region=evidence,
            image_width=w,
            image_height=h
        )

        try:
            if chosen_action == Action.REGROUND:
                controller_decisions["acquire"] += 1
                res_c4 = intervention.execute(
                    image_path=img_path,
                    query=q,
                    client=client,
                    evidence_region=evidence,
                    initial_answer=ans_full,
                    preserve_context=True
                )
                ledger_a["attempted_requests"] += 1
                ledger_a["successful_requests"] += 1
                ledger_a["total_requests"] += 1
                ledger_a["requests_by_condition"]["4_ccea_cost_aware_controller"] += 1
                ledger_a["total_tokens"] += res_c4.total_tokens
                ledger_a["total_cost_usd"] += res_c4.cost_usd

                final_ans_ctrl = res_c4.final_answer
                tokens_ctrl = (p_tok + c_tok) + res_c4.total_tokens
                lat_ctrl = lat + res_c4.latency
                ctx_ok = res_c4.assessment and res_c4.assessment.valid
                deployed_calls_ctrl = 2
                deployed_tokens_ctrl = tokens_ctrl
                deployed_lat_ctrl = lat_ctrl
            elif chosen_action == Action.ABSTAIN:
                controller_decisions["abstain"] += 1
                final_ans_ctrl = "ABSTAIN_INSUFFICIENT_EVIDENCE"
                tokens_ctrl = (p_tok + c_tok)
                lat_ctrl = lat
                ctx_ok = False
                deployed_calls_ctrl = 1
                deployed_tokens_ctrl = tokens_ctrl
                deployed_lat_ctrl = lat_ctrl
            else:  # STOP
                controller_decisions["stop"] += 1
                final_ans_ctrl = ans_full
                tokens_ctrl = (p_tok + c_tok)
                lat_ctrl = lat
                ctx_ok = True
                deployed_calls_ctrl = 1
                deployed_tokens_ctrl = tokens_ctrl
                deployed_lat_ctrl = lat_ctrl

            ledger_b["deployed_requests"] += deployed_calls_ctrl
            ledger_b["deployed_tokens"] += deployed_tokens_ctrl
            ledger_b["deployed_latency"] += deployed_lat_ctrl
            ledger_b["deployed_cost_usd"] += ((deployed_tokens_ctrl / 1000000.0) * 0.15)

            sc_ctrl = score_vdu_comprehensive(final_ans_ctrl, gt)
            stats["4_ccea_cost_aware_controller"]["correct_count"] += sc_ctrl["strict_em"]
            stats["4_ccea_cost_aware_controller"]["unit_correct_count"] += sc_ctrl["unit_sensitive_em"]
            stats["4_ccea_cost_aware_controller"]["total_f1"] += sc_ctrl["token_f1"]
            stats["4_ccea_cost_aware_controller"]["total_tokens"] += tokens_ctrl
            stats["4_ccea_cost_aware_controller"]["total_latency"] += lat_ctrl
            stats["4_ccea_cost_aware_controller"]["completed_count"] += 1
            if ctx_ok:
                stats["4_ccea_cost_aware_controller"]["context_retained_count"] += 1
            else:
                stats["4_ccea_cost_aware_controller"]["context_lost_count"] += 1

            inst_record["conditions"]["4_ccea_cost_aware_controller"] = {
                "status": "SUCCESS",
                "chosen_action": chosen_action.value,
                "prediction": final_ans_ctrl,
                "strict_em": sc_ctrl["strict_em"],
                "unit_sensitive_em": sc_ctrl["unit_sensitive_em"],
                "f1": sc_ctrl["token_f1"],
                "tokens": tokens_ctrl,
                "latency": round(lat_ctrl, 4),
                "deployed_calls": deployed_calls_ctrl,
                "cost_usd": round(tokens_ctrl * 0.00015 / 1000.0, 6)
            }
        except BudgetExceededError as be:
            is_complete = False
            halt_reason = str(be)
            ledger_a["blocked_requests"] += 1
            stats["4_ccea_cost_aware_controller"]["blocked_count"] += 1
            inst_record["conditions"]["4_ccea_cost_aware_controller"] = {
                "status": "BLOCKED_BUDGET_EXCEEDED",
                "chosen_action": chosen_action.value,
                "prediction": "BLOCKED_BUDGET_GUARD",
                "strict_em": None,
                "unit_sensitive_em": None,
                "f1": 0.0,
                "tokens": 0,
                "latency": 0.0,
                "cost_usd": 0.0
            }
            all_records.append(inst_record)
            break

        # -------------------------------------------------------------
        # Condition 5: Fixed_Center_Crop (50% uniform center baseline)
        # -------------------------------------------------------------
        c5_ymin, c5_xmin = int(0.25 * h), int(0.25 * w)
        c5_ymax, c5_xmax = int(0.75 * h), int(0.75 * w)
        c5_box = (c5_ymin, c5_xmin, c5_ymax, c5_xmax)
        c5_path, c5_dims = extract_and_save_crop(
            img_path,
            c5_box,
            output_dir="discovery/live_validation/confirmatory_experiment/transformed_images"
        )
        p_c5 = f"Answer concisely based on the visual document crop:\nQuestion: {q}\nAnswer:"
        try:
            if hasattr(client, "call_vlm"):
                ans_c5, p_tok_c5, c_tok_c5, lat_c5 = client.call_vlm(p_c5, c5_path, max_tokens=60)
            else:
                res_dict_c5 = client.call(p_c5, image_path=c5_path, max_tokens=60)
                ans_c5 = res_dict_c5.get("text", "")
                p_tok_c5 = res_dict_c5.get("prompt_tokens") if res_dict_c5.get("prompt_tokens") is not None else 2800
                c_tok_c5 = res_dict_c5.get("completion_tokens") if res_dict_c5.get("completion_tokens") is not None else 60
                lat_c5 = res_dict_c5.get("latency_sec") or 0.0

            ledger_a["attempted_requests"] += 1
            ledger_a["successful_requests"] += 1
            ledger_a["total_requests"] += 1
            ledger_a["requests_by_condition"]["5_fixed_center_crop"] += 1
            ledger_a["total_tokens"] += (p_tok_c5 + c_tok_c5)
            ledger_a["total_cost_usd"] += (((p_tok_c5 + c_tok_c5) / 1000000.0) * 0.15)

            sc_c5 = score_vdu_comprehensive(ans_c5, gt)
            stats["5_fixed_center_crop"]["correct_count"] += sc_c5["strict_em"]
            stats["5_fixed_center_crop"]["unit_correct_count"] += sc_c5["unit_sensitive_em"]
            stats["5_fixed_center_crop"]["total_f1"] += sc_c5["token_f1"]
            stats["5_fixed_center_crop"]["total_tokens"] += (p_tok_c5 + c_tok_c5)
            stats["5_fixed_center_crop"]["total_latency"] += lat_c5
            stats["5_fixed_center_crop"]["completed_count"] += 1
            
            c5_assessment = assess_crop(c5_box, evidence, w, h)
            if c5_assessment.valid:
                stats["5_fixed_center_crop"]["context_retained_count"] += 1
            else:
                stats["5_fixed_center_crop"]["context_lost_count"] += 1
                for missing_k in c5_assessment.missing_context:
                    stats["5_fixed_center_crop"]["lost_by_kind"][missing_k.value] += 1

            inst_record["conditions"]["5_fixed_center_crop"] = {
                "status": "SUCCESS",
                "prediction": ans_c5,
                "crop_box": list(c5_box),
                "crop_path": c5_path,
                "strict_em": sc_c5["strict_em"],
                "unit_sensitive_em": sc_c5["unit_sensitive_em"],
                "f1": sc_c5["token_f1"],
                "tokens": p_tok_c5 + c_tok_c5,
                "latency": round(lat_c5, 4),
                "cost_usd": round((p_tok_c5 + c_tok_c5) * 0.00015 / 1000.0, 6)
            }
        except BudgetExceededError as be:
            is_complete = False
            halt_reason = str(be)
            ledger_a["blocked_requests"] += 1
            stats["5_fixed_center_crop"]["blocked_count"] += 1
            inst_record["conditions"]["5_fixed_center_crop"] = {
                "status": "BLOCKED_BUDGET_EXCEEDED",
                "prediction": "BLOCKED_BUDGET_GUARD",
                "strict_em": None,
                "unit_sensitive_em": None,
                "f1": 0.0,
                "tokens": 0,
                "latency": 0.0,
                "cost_usd": 0.0
            }
            all_records.append(inst_record)
            break
        except Exception as e:
            is_complete = False
            halt_reason = f"Execution error in C5: {e}"
            ledger_a["failed_requests"] += 1
            inst_record["conditions"]["5_fixed_center_crop"] = {
                "status": "ERROR",
                "prediction": "ERROR",
                "strict_em": None,
                "unit_sensitive_em": None,
                "f1": 0.0,
                "tokens": 0,
                "latency": 0.0,
                "cost_usd": 0.0
            }
            all_records.append(inst_record)
            break

        all_records.append(inst_record)

    # Sync retried and failed counts from client guard if available
    if hasattr(client, "budget_guard") and client.budget_guard:
        guard_sum = client.budget_guard.get_summary()
        ledger_a["retried_requests"] = guard_sum.get("retried_requests", 0)
        ledger_a["failed_requests"] = guard_sum.get("failed_requests", 0)
        ledger_a["attempted_requests"] = guard_sum.get("attempted_requests", ledger_a["attempted_requests"])

    # Wrap full report with independent accounting ledgers and run integrity flags
    full_output = {
        "metadata": {
            "num_instances": len(instances),
            "slice": [start_idx, end_idx],
            "mode": "LIVE API" if use_live else "OFFLINE MOCK / PIPELINE VALIDATION",
            "model": "pixtral-12b-2409",
            "pricing": "$0.15 / 1M tokens ($0.00015 / 1k tokens)",
            "is_complete": is_complete,
            "run_status": "COMPLETED" if is_complete else "INCOMPLETE_BUDGET_HALT",
            "halt_reason": halt_reason
        },
        "ledger_a_experimental_usage": {
            "planned_requests": ledger_a["planned_requests"],
            "attempted_requests": ledger_a["attempted_requests"],
            "successful_requests": ledger_a["successful_requests"],
            "failed_requests": ledger_a["failed_requests"],
            "retried_requests": ledger_a["retried_requests"],
            "blocked_requests": ledger_a["blocked_requests"],
            "total_requests": ledger_a["total_requests"],
            "total_tokens": ledger_a["total_tokens"],
            "total_cost_usd": round(ledger_a["total_cost_usd"], 6),
            "requests_by_condition": ledger_a["requests_by_condition"]
        },
        "ledger_b_deployed_policy": {
            "c4_deployed_requests": ledger_b["deployed_requests"],
            "c4_deployed_tokens": ledger_b["deployed_tokens"],
            "c4_deployed_latency_sec": round(ledger_b["deployed_latency"], 3),
            "c4_deployed_cost_usd": round(ledger_b["deployed_cost_usd"], 6),
            "baseline_c3_requests": ledger_b["baseline_c3_requests"],
            "baseline_c3_tokens": ledger_b["baseline_c3_tokens"],
            "baseline_c3_cost_usd": round(ledger_b["baseline_c3_cost_usd"], 6),
            "token_difference_pct": round(((ledger_b["deployed_tokens"] - ledger_b["baseline_c3_tokens"]) / max(1, ledger_b["baseline_c3_tokens"])) * 100.0, 2)
        },
        "records": all_records
    }

    records_out = Path(records_out_path) if records_out_path else Path("discovery/live_validation/ccea_ablation_records.json")
    held_out_out = Path(held_out_out_path) if held_out_out_path else Path("discovery/live_validation/ccea_held_out_audit.json")
    for out_p in [records_out, held_out_out]:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(full_output, f, indent=2)
    print(f"[Artifact] Saved detailed per-instance ablation records to {records_out} and {held_out_out}")

    if not is_complete:
        print("\n" + "!" * 95)
        print(f"[WARNING] RUN INCOMPLETE: Execution halted by BudgetGuard.")
        print(f"Halt Reason: {halt_reason}")
        print(f"Planned requests: {ledger_a['planned_requests']} | Attempted: {ledger_a['attempted_requests']} | Blocked: {ledger_a['blocked_requests']}")
        print("CRITICAL INTEGRITY RULE: Partial runs CANNOT silently enter the 5-condition comparison.")
        print("!" * 95 + "\n")
        return stats

    # Print Summary Table for completed runs
    N = len(instances)
    print("\n" + "=" * 95)
    print(f"ABLATION RESULTS TABLE (N = {N} instances)")
    print("=" * 95)
    print(f"{'Condition':<32} | {'Strict EM (%)':<14} | {'Unit EM (%)':<12} | {'Avg F1':<8} | {'Tokens':<8} | {'Ctx Retained'}")
    print("-" * 95)

    for cond, s in stats.items():
        acc = (s["correct_count"] / N) * 100.0 if N > 0 else 0.0
        unit_acc = (s["unit_correct_count"] / N) * 100.0 if N > 0 else 0.0
        avg_f1 = s["total_f1"] / N if N > 0 else 0.0
        avg_tok = round(s["total_tokens"] / N) if N > 0 else 0
        ret_rate = f"{s['context_retained_count']}/{N} ({round(s['context_retained_count']/N*100) if N > 0 else 0}%)"
        print(f"{cond:<32} | {acc:<14.1f} | {unit_acc:<12.1f} | {avg_f1:<8.3f} | {avg_tok:<8} | {ret_rate}")

    print("\n" + "-" * 95)
    print(f"LEDGER A (Actual Experimental Usage across all conditions):")
    print(f"  - Planned requests:   {ledger_a['planned_requests']}")
    print(f"  - Attempted requests: {ledger_a['attempted_requests']}")
    print(f"  - Successful:         {ledger_a['successful_requests']}")
    print(f"  - Failed / Retried:   {ledger_a['failed_requests']} / {ledger_a['retried_requests']}")
    print(f"  - Blocked requests:   {ledger_a['blocked_requests']}")
    print(f"  - Total tokens:       {ledger_a['total_tokens']}")
    print(f"  - Total cost:         ${ledger_a['total_cost_usd']:.5f} USD")
    print("-" * 95)
    print(f"LEDGER B (Deployed C4 Policy vs Fixed C3 Baseline):")
    print(f"  - C4 Deployed requests: {ledger_b['deployed_requests']} (vs C3: {ledger_b['baseline_c3_requests']})")
    print(f"  - C4 Deployed tokens:   {ledger_b['deployed_tokens']} (vs C3: {ledger_b['baseline_c3_tokens']})")
    print(f"  - C4 Deployed cost:     ${ledger_b['deployed_cost_usd']:.5f} USD (vs C3: ${ledger_b['baseline_c3_cost_usd']:.5f} USD)")
    print(f"  - Controller actions:   {controller_decisions}")
    print("=" * 95 + "\n")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index in instances.json")
    parser.add_argument("--end_idx", type=int, default=None, help="Ending index in instances.json")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples (if end_idx not specified)")
    parser.add_argument("--live", action="store_true", help="Execute genuine live API calls")
    parser.add_argument("--auto_locator", action="store_true", help="Use automatic non-oracle OCR locator instead of oracle spatial annotations")
    args = parser.parse_args()

    run_ccea_ablation(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        num_samples=args.num_samples if args.end_idx is None else None,
        use_live=args.live,
        auto_localization=args.auto_locator
    )
