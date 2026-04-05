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


def collect_ui_like_feedback(criteria_obj: Dict[str, Any], matching_percent: float) -> List[str]:
    limit = feedback_limit_by_score(float(matching_percent))
    if limit <= 0:
        return []

    feedbacks: List[str] = []

    items = criteria_obj.get("items", []) or []
    for item in items:
        student = item.get("student", {}) or {}
        passed = bool(student.get("passed", False))
        if passed:
            continue

        txt = feedback_text_for_item(item)
        if txt and txt not in feedbacks:
            feedbacks.append(txt)

        if len(feedbacks) >= limit:
            return feedbacks[:limit]

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