from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd

from result_logic import (
    load_score_params_from_env,
    build_score_obj,
    score_message,
)


DEFAULT_SCORE_PARAMS = load_score_params_from_env()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _extract_raw_dtw_score(score_obj: Dict[str, Any]) -> Optional[float]:
    """
    Extract the raw DTW score from score.json.

    Supported formats:
    1. UI-style score.json:
       {
         "matching_percent": ...,
         "dtw": {
           "overall_matching_score": ...
         }
       }

    2. Raw DTW score.json:
       {
         "overall_matching_score": ...
       }

    3. Partially wrapped score.json:
       {
         "dtw": {
           "overall_matching_score": ...
         }
       }
    """
    if not isinstance(score_obj, dict):
        return None

    dtw_obj = score_obj.get("dtw")

    if isinstance(dtw_obj, dict):
        val = _safe_float(dtw_obj.get("overall_matching_score"))
        if val is not None:
            return val

    val = _safe_float(score_obj.get("overall_matching_score"))
    if val is not None:
        return val

    # Fallback only.
    # This is less ideal because matching_percent is already mapped,
    # not the raw DTW value.
    return None


def _extract_mean_error(score_obj: Dict[str, Any]) -> str:
    """
    Keep a useful error column for the summary CSV.

    Priority:
    1. dtw.mean_normalized_error
    2. top-level mean_normalized_error
    3. dtw.overall_matching_score
    4. top-level overall_matching_score
    """
    if not isinstance(score_obj, dict):
        return ""

    dtw_obj = score_obj.get("dtw")

    if isinstance(dtw_obj, dict):
        for key in ["mean_normalized_error", "overall_matching_score"]:
            val = dtw_obj.get(key)
            if val is not None:
                return str(val)

    for key in ["mean_normalized_error", "overall_matching_score"]:
        val = score_obj.get(key)
        if val is not None:
            return str(val)

    return ""


def _read_criteria_basic(throw_dir: Path) -> Dict[str, str]:
    """
    Optional: preserve C1-C4 columns if criteria.json exists.
    If criteria.json does not exist, output No Data.
    """
    row: Dict[str, str] = {}

    criteria_path = throw_dir / "criteria.json"
    if not criteria_path.exists():
        for c in ["C1", "C2", "C3", "C4"]:
            row[f"{c}_Pass"] = "-"
            row[f"{c}_Reason"] = "No Data"
        return row

    try:
        data = json.loads(criteria_path.read_text(encoding="utf-8"))
    except Exception:
        for c in ["C1", "C2", "C3", "C4"]:
            row[f"{c}_Pass"] = "-"
            row[f"{c}_Reason"] = "No Data"
        return row

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
    elif isinstance(data, dict) and isinstance(data.get("criteria"), list):
        items = data["criteria"]
    elif isinstance(data, list):
        items = data
    else:
        items = []

    found = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        key = str(item.get("key", "")).upper()
        if key not in {"C1", "C2", "C3", "C4"}:
            continue

        found.add(key)

        student_result = item.get("student", {}) or {}
        passed = bool(student_result.get("passed", False))
        notes = student_result.get("notes", []) or []

        row[f"{key}_Pass"] = "Y" if passed else "N"
        row[f"{key}_Reason"] = "" if passed else "; ".join(map(str, notes))

    for c in ["C1", "C2", "C3", "C4"]:
        if c not in found:
            row[f"{c}_Pass"] = "-"
            row[f"{c}_Reason"] = "No Data"

    return row


def rescore_existing_runs(
    runs_root: Path,
    *,
    output_csv: Path,
    score_params: Dict[str, float],
) -> None:
    """
    Rescore existing batch results.

    This function does NOT:
    - read videos
    - run pose detection
    - run DTW again
    - generate compare.mp4
    - generate criteria.json

    It only reads existing score.json files and writes a new summary CSV.
    """
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root does not exist: {runs_root}")

    rows: List[Dict[str, Any]] = []

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue

        throws_dir = run_dir / "throws"
        if not throws_dir.exists():
            continue

        for throw_dir in sorted(throws_dir.iterdir()):
            if not throw_dir.is_dir() or not throw_dir.name.startswith("throw_"):
                continue

            score_path = throw_dir / "score.json"

            row: Dict[str, Any] = {
                "Run": run_dir.name,
                "Throw": throw_dir.name,
                "Score": "",
                "Mean_Error": "",
                "Raw_DTW_Overall": "",
                "Score_Message": "",
                "Feedback_1": "",
                "Feedback_2": "",
            }

            if not score_path.exists():
                row["Score_Message"] = "No score.json"
                row.update(_read_criteria_basic(throw_dir))
                rows.append(row)
                continue

            try:
                score_obj = json.loads(score_path.read_text(encoding="utf-8"))
            except Exception as e:
                row["Score_Message"] = f"Failed to read score.json: {e}"
                row.update(_read_criteria_basic(throw_dir))
                rows.append(row)
                continue

            raw_dtw = _extract_raw_dtw_score(score_obj)
            row["Mean_Error"] = _extract_mean_error(score_obj)

            if raw_dtw is None:
                row["Score_Message"] = "No raw DTW overall_matching_score"
                row.update(_read_criteria_basic(throw_dir))
                rows.append(row)
                continue

            new_score_obj = build_score_obj(float(raw_dtw), **score_params)
            new_score = new_score_obj.get("matching_percent", "")

            row["Raw_DTW_Overall"] = raw_dtw
            row["Score"] = new_score

            score_float = _safe_float(new_score) or 0.0
            row["Score_Message"] = score_message(score_float)

            row.update(_read_criteria_basic(throw_dir))
            rows.append(row)

    if not rows:
        print("[INFO] No existing throw results found.")
        return

    df = pd.DataFrame(rows)

    preferred_cols = [
        "Run",
        "Throw",
        "Score",
        "Mean_Error",
        "Raw_DTW_Overall",
        "Score_Message",
        "Feedback_1",
        "Feedback_2",
        "C1_Pass",
        "C1_Reason",
        "C2_Pass",
        "C2_Reason",
        "C3_Pass",
        "C3_Reason",
        "C4_Pass",
        "C4_Reason",
    ]

    existing_cols = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols]

    df = df.sort_values(by=["Run", "Throw"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[SUCCESS] Rescored summary written to: {output_csv}")
    print(f"[INFO] score_params = {score_params}")
    print(f"[INFO] rows = {len(df)}")


def main() -> None:
    ap = argparse.ArgumentParser()

    # Keep the same input shape as batch_ui_like.py.
    # video_dir is accepted for compatibility, but this script does not read videos.
    ap.add_argument("--video_dir", type=str, required=True, help="Kept for compatibility. Not used.")
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--reference_csv", type=str, default="reference/model.csv")
    ap.add_argument("--thresholds_json", type=str, default="config/dtw_thresholds.json")
    ap.add_argument("--green_yellow", type=float, default=0.25)
    ap.add_argument("--yellow_red", type=float, default=0.50)
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--t1", type=float, default=DEFAULT_SCORE_PARAMS["t1"])
    ap.add_argument("--t2", type=float, default=DEFAULT_SCORE_PARAMS["t2"])
    ap.add_argument("--t3", type=float, default=DEFAULT_SCORE_PARAMS["t3"])
    ap.add_argument("--s1", type=float, default=DEFAULT_SCORE_PARAMS["s1"])
    ap.add_argument("--s2", type=float, default=DEFAULT_SCORE_PARAMS["s2"])

    args = ap.parse_args()

    # Same output filename as batch_ui_like.py.
    output_csv = Path("batch_summary_report.csv").resolve()

    score_params = {
        "t1": args.t1,
        "t2": args.t2,
        "t3": args.t3,
        "s1": args.s1,
        "s2": args.s2,
    }

    print("[INFO] This script does NOT process videos.")
    print("[INFO] It only rescales existing score.json files.")
    print(f"[INFO] video_dir is ignored: {args.video_dir}")
    print(f"[INFO] reference_csv is ignored: {args.reference_csv}")
    print(f"[INFO] thresholds_json is ignored: {args.thresholds_json}")
    print(f"[INFO] green_yellow/yellow_red are ignored: {args.green_yellow}, {args.yellow_red}")
    print(f"[INFO] limit is ignored: {args.limit}")

    rescore_existing_runs(
        runs_root=Path(args.runs_root),
        output_csv=output_csv,
        score_params=score_params,
    )


if __name__ == "__main__":
    main()