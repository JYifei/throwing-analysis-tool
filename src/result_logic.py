from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional


DEFAULT_SCORE_PARAMS = {
    "t1": 1.0,
    "t2": 2.2,
    "t3": 4.6,
    "s1": 95.0,
    "s2": 40.0,
}


def load_score_params_from_env() -> Dict[str, float]:
    return {
        "t1": float(os.environ.get("THROWING_SCORE_T1", DEFAULT_SCORE_PARAMS["t1"])),
        "t2": float(os.environ.get("THROWING_SCORE_T2", DEFAULT_SCORE_PARAMS["t2"])),
        "t3": float(os.environ.get("THROWING_SCORE_T3", DEFAULT_SCORE_PARAMS["t3"])),
        "s1": float(os.environ.get("THROWING_SCORE_S1", DEFAULT_SCORE_PARAMS["s1"])),
        "s2": float(os.environ.get("THROWING_SCORE_S2", DEFAULT_SCORE_PARAMS["s2"])),
    }


def piecewise_score_from_mean_error(
    mean_err: float,
    *,
    t1: float,
    t2: float,
    t3: float,
    s1: float,
    s2: float,
) -> float:
    if not math.isfinite(mean_err):
        return 0.0

    if not (0.0 < t1 < t2 < t3):
        raise ValueError(f"Require 0 < t1 < t2 < t3, got {(t1, t2, t3)}")

    if not (0.0 <= s2 <= s1 <= 100.0):
        raise ValueError(f"Require 0 <= s2 <= s1 <= 100, got {(s1, s2)}")

    e = max(0.0, float(mean_err))

    if e <= t1:
        # Segment 1: gentle slope
        score = 100.0 - (100.0 - s1) * (e / t1)

    elif e <= t2:
        # Segment 2: steeper slope
        score = s1 - (s1 - s2) * ((e - t1) / (t2 - t1))

    elif e <= t3:
        # Segment 3: gentle slope to zero
        score = s2 - s2 * ((e - t2) / (t3 - t2))

    else:
        score = 0.0

    return max(0.0, min(100.0, float(score)))


def build_score_obj(
    mean_err: float,
    *,
    t1: float,
    t2: float,
    t3: float,
    s1: float,
    s2: float,
) -> Dict[str, Any]:
    matching = piecewise_score_from_mean_error(
        mean_err,
        t1=t1,
        t2=t2,
        t3=t3,
        s1=s1,
        s2=s2,
    )

    return {
        "matching_percent": round(float(matching), 1),
        "mean_normalized_error": round(float(mean_err), 4) if math.isfinite(mean_err) else None,
        "mapping": {
            "type": "piecewise_linear_3seg",
            "t1": float(t1),
            "t2": float(t2),
            "t3": float(t3),
            "s1": float(s1),
            "s2": float(s2),
            "rule": "0->100, t1->s1, t2->s2, t3->0, piecewise linear",
        },
    }


def score_message(p: float) -> str:
    if p > 95:
        return "Great throw!"
    if p > 75:
        return "Good throw!"
    if p > 50:
        return "Good try!"
    return "Let's try again:"


def feedback_limit_by_score(p: float) -> int:
    if p > 95:
        return 0
    if p > 75:
        return 1
    return 2


def feedback_text_for_item(item: Dict[str, Any]) -> Optional[str]:
    key = str(item.get("key", ""))
    student = item.get("student", {}) or {}
    notes = student.get("notes", []) or []

    if key == "c1":
        return "Bring the throwing arm back to form an L-shape"

    if key == "c2":
        return "Open the shoulder so that the non-throwing side is facing the target"

    if key == "c3":
        if "step_with_wrong_foot" in notes:
            return "Step forward with the opposite foot"
        if "take_a_bigger_step" in notes:
            return "Take a bigger step forward with the opposite foot"
        if "take_a_step_forward" in notes:
            return "Step forward with the opposite foot"
        return "Step forward with the opposite foot"

    if key == "c4":
        return "Follow through across your body"

    return None


def feedback_text_for_extra_signal(signal_key: str) -> Optional[str]:
    if signal_key == "throw_higher":
        return "Throw higher"
    if signal_key == "throw_harder":
        return "Throw harder"
    return None


def _extra_signal_triggered(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        if "triggered" in value:
            return bool(value["triggered"])
        if "active" in value:
            return bool(value["active"])
        if "passed" in value:
            return not bool(value["passed"])
    return False

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_float(d: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        v = d.get(key, default)
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _normalized_shortfall(value: float, thr: float) -> float:
    """
    For metrics where larger is better.
    0 means no deficit, 1 means very bad / missing badly.
    """
    if not math.isfinite(value) or not math.isfinite(thr):
        return 1.0
    if thr <= 1e-6:
        return 0.0 if value >= thr else 1.0
    return _clip01((thr - value) / thr)

def _piecewise_larger_better(value: float, bad: float, passed: float) -> float:
    if not math.isfinite(value):
        return 1.0
    if passed <= bad:
        return 0.0 if value >= passed else 1.0
    if value >= passed:
        return 0.0
    if value <= bad:
        return 1.0
    return _clip01((passed - value) / (passed - bad))


def _piecewise_smaller_better(value: float, passed: float, bad: float) -> float:
    if not math.isfinite(value):
        return 1.0
    if bad <= passed:
        return 0.0 if value <= passed else 1.0
    if value <= passed:
        return 0.0
    if value >= bad:
        return 1.0
    return _clip01((value - passed) / (bad - passed))

def _criterion_deficit(item: Dict[str, Any]) -> float:
    """
    Return a normalized deficit in [0, 1].
    0 = no problem, 1 = very severe problem.
    """
    key = str(item.get("key", ""))
    student = item.get("student", {}) or {}
    metrics = student.get("metrics", {}) or {}
    thresholds = student.get("thresholds", {}) or {}
    notes = student.get("notes", []) or []

    if bool(student.get("passed", False)):
        return 0.0

    # C1: best-pose score shortfall
    if key == "c1":
        score = _safe_float(metrics, "best_score")
        thr = _safe_float(thresholds, "C1_PASS_THR")
        if math.isfinite(score) and math.isfinite(thr) and thr > 1e-6:
            return _clip01((thr - score) / thr)
        return 1.0

    # C2: shoulder opening shortfall
    if key == "c2":
        gain = _safe_float(metrics, "shoulder_open_gain")
        thr = _safe_float(thresholds, "SHOULDER_OPEN_GAIN_THR")

        # gain is already baseline-corrected, so 0 means "no opening beyond baseline"
        if math.isfinite(gain) and math.isfinite(thr) and thr > 1e-6:
            return _clip01((thr - gain) / thr)
        return 1.0

    # C3
    if key == "c3":
        if "step_with_wrong_foot" in notes:
            return 1.0

        val = _safe_float(metrics, "disp_x_norm")
        bad = _safe_float(thresholds, "STEP_X_TINY_THR")   # 0.1
        passed = _safe_float(thresholds, "STEP_X_THR")     # 0.5
        return _piecewise_larger_better(val, bad, passed)

    # C4
    if key == "c4":
        lead = _safe_float(metrics, "lead_max_tail_norm")
        hip_dist = _safe_float(metrics, "min_dn_tail")

        LEAD_PASS = _safe_float(thresholds, "LEAD_THR")    # 0.0
        LEAD_BAD = -0.15

        HIP_PASS = _safe_float(thresholds, "NEAR_HIP_THR") # 0.60
        HIP_BAD = 1.00

        notes_set = set(str(x) for x in notes)

        d_lead = 0.0
        if "throwing_shoulder_not_leading_target" in notes_set:
            d_lead = _piecewise_larger_better(lead, LEAD_BAD, LEAD_PASS)

        d_hip = 0.0
        if "hand_not_close_enough_to_opposite_hip" in notes_set:
            d_hip = _piecewise_smaller_better(hip_dist, HIP_PASS, HIP_BAD)

        return max(d_lead, d_hip)

    # Fallback
    return 1.0


def _criterion_importance_weight(item: Dict[str, Any]) -> float:
    key = str(item.get("key", ""))

    # Teaching-oriented priority
    if key == "c3":
        return 1
    if key == "c2":
        return 1
    if key == "c1":
        return 1
    if key == "c4":
        return 1
    return 1.00


def _criterion_severity(item: Dict[str, Any]) -> float:
    return _criterion_importance_weight(item) * _criterion_deficit(item)

def annotate_criteria_with_severity(criteria_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mutate and return criteria_obj by adding per-item severity info
    into each student's block, plus a top-level ranking for debugging.
    """
    items = criteria_obj.get("items", []) or []

    ranking: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        student = item.get("student", {}) or {}

        deficit = _criterion_deficit(item)
        weight = _criterion_importance_weight(item)
        severity = _criterion_severity(item)

        student["deficit"] = round(float(deficit), 4)
        student["importance_weight"] = round(float(weight), 4)
        student["severity"] = round(float(severity), 4)

        ranking.append({
            "key": str(item.get("key", "")),
            "name": str(item.get("name", "")),
            "passed": bool(student.get("passed", False)),
            "deficit": round(float(deficit), 4),
            "importance_weight": round(float(weight), 4),
            "severity": round(float(severity), 4),
            "original_order": idx,
        })

        item["student"] = student

    ranking.sort(key=lambda x: (-x["severity"], x["original_order"]))

    criteria_obj["severity_ranking"] = ranking
    return criteria_obj


def collect_ui_like_feedback(criteria_obj: Dict[str, Any], matching_percent: float) -> List[str]:
    limit = feedback_limit_by_score(float(matching_percent))
    if limit <= 0:
        return []

    feedbacks: List[str] = []
    items = criteria_obj.get("items", []) or []

    ranked_failed = []
    for idx, item in enumerate(items):
        student = item.get("student", {}) or {}
        if bool(student.get("passed", False)):
            continue

        sev = _criterion_severity(item)
        ranked_failed.append((sev, idx, item))

    # Higher severity first; keep original order as tie-breaker
    ranked_failed.sort(key=lambda x: (-x[0], x[1]))

    for sev, idx, item in ranked_failed:
        txt = feedback_text_for_item(item)
        if txt and txt not in feedbacks:
            feedbacks.append(txt)
        if len(feedbacks) >= limit:
            return feedbacks[:limit]

    # Keep extra signals after the main criteria
    extra = criteria_obj.get("extra_feedback_signals", {}) or {}
    if isinstance(extra, dict):
        for key, value in extra.items():
            if not _extra_signal_triggered(value):
                continue
            txt = feedback_text_for_extra_signal(str(key))
            if txt and txt not in feedbacks:
                feedbacks.append(txt)
            if len(feedbacks) >= limit:
                return feedbacks[:limit]

    return feedbacks[:limit]