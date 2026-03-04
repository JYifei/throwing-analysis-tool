# batch_ui_like.py
from __future__ import annotations

import os
import json
import shutil
import traceback
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import cv2
import pandas as pd
import numpy as np

# ---- import exactly like your FastAPI main.py ----
from fsm_new import ensure_browser_mp4, run_once
from render_side_by_side import (
    global_bbox_from_csv,
    crop_video_to_rect,
    crop_csv_to_rect,
    _expand_rect_to_size,
    render_side_by_side_points,
    compute_mean_normalized_error,
    MAIN_JOINTS,
    read_map_csv,
    _load_xy_pairs_cached,
)
from criteria_eval import evaluate_criteria


# -------------------------
# Defaults (same as app)
# -------------------------
DEFAULT_MODEL_CLIP = Path(os.environ.get("THROWING_MODEL_CLIP", "reference/model_clip.mp4")).resolve()
RUNS_ROOT = Path(os.environ.get("THROWING_RUNS_ROOT", "runs")).resolve()
DEFAULT_REFERENCE_CSV = Path(os.environ.get("THROWING_REFERENCE_CSV", "reference/model.csv")).resolve()
DEFAULT_THRESHOLDS_JSON = Path(os.environ.get("THROWING_THRESHOLDS_JSON", "config/dtw_thresholds.json")).resolve()

INPUT_VIDEO_NAME = "input.mp4"

def _get_median_torso_length(csv_path: str) -> Optional[float]:
    """
    通过读取 CSV 中的肩膀和髋部坐标，计算该视频中人物躯干的平均(中位数)像素长度。
    """
    try:
        df = pd.read_csv(csv_path)
        # 获取左右肩、左右髋
        lsx = pd.to_numeric(df['left_shoulder_x'], errors='coerce')
        lsy = pd.to_numeric(df['left_shoulder_y'], errors='coerce')
        rsx = pd.to_numeric(df['right_shoulder_x'], errors='coerce')
        rsy = pd.to_numeric(df['right_shoulder_y'], errors='coerce')
        
        lhx = pd.to_numeric(df['left_hip_x'], errors='coerce')
        lhy = pd.to_numeric(df['left_hip_y'], errors='coerce')
        rhx = pd.to_numeric(df['right_hip_x'], errors='coerce')
        rhy = pd.to_numeric(df['right_hip_y'], errors='coerce')
        
        # 算中心点
        msx, msy = (lsx + rsx) / 2.0, (lsy + rsy) / 2.0
        mhx, mhy = (lhx + rhx) / 2.0, (lhy + rhy) / 2.0
        
        # 算欧式距离
        dist = np.sqrt((msx - mhx)**2 + (msy - mhy)**2)
        median_dist = float(np.nanmedian(dist))
        
        return median_dist if np.isfinite(median_dist) and median_dist > 0 else None
    except Exception:
        return None

def _read_facing(csv_path: str) -> Optional[str]:
    try:
        df = pd.read_csv(csv_path)
        if "body_facing" not in df.columns:
            return None
        s = df["body_facing"].dropna().astype(str)
        if s.empty:
            return None
        return s.value_counts().idxmax()
    except Exception:
        return None


def _new_job_dir_from_stem(runs_root: Path, stem: str) -> Path:
    """
    Use video filename (stem) as job folder name.
    If exists, append _1, _2, ... to avoid overwrite.
    """
    base = runs_root / stem
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        (base / "inputs").mkdir(parents=True, exist_ok=True)
        (base / "outputs").mkdir(parents=True, exist_ok=True)
        return base

    # auto-increment if already exists
    idx = 1
    while True:
        candidate = runs_root / f"{stem}_{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            (candidate / "inputs").mkdir(parents=True, exist_ok=True)
            (candidate / "outputs").mkdir(parents=True, exist_ok=True)
            return candidate
        idx += 1


def _write_job_status(job_dir: Path, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
    obj = {"status": status}
    if extra:
        obj.update(extra)
    out = job_dir / "outputs" / "job_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _collect_videos(video_dir: Path) -> List[Path]:
    exts = {".mp4", ".mov", ".m4v"}
    vids = []
    for p in sorted(video_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            vids.append(p)
    return vids


def _ui_like_postprocess_one_job(
    job_dir: Path,
    job_id: str,
    *,
    reference_csv: Path,
    thresholds_json: Path,
    green_yellow: float = 0.25,
    yellow_red: float = 0.50,
) -> None:
    """
    This is a direct adaptation of your FastAPI create_job() post-processing.
    """
    manifest_path = job_dir / "outputs" / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("manifest.json missing")

    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    throws = manifest_obj.get("throws", []) or []

    for t in throws:
        throw_id = t.get("throw_id")
        if not throw_id:
            continue

        throw_dir = job_dir / "throws" / throw_id
        map_csv = throw_dir / "clip_pose_align_map.csv"
        if not map_csv.exists():
            continue

        student_clip = throw_dir / "clip.mp4"
        stu_csv = throw_dir / "clip.csv"
        if not (student_clip.exists() and stu_csv.exists()):
            continue

        # ---------- 1) Compute student crop rect ----------
        cap_tmp = cv2.VideoCapture(str(student_clip))
        ok, first_frame = cap_tmp.read()
        cap_tmp.release()
        if not ok or first_frame is None:
            continue
        hs, ws = first_frame.shape[:2]
        rect_s = global_bbox_from_csv(str(stu_csv), ws, hs, margin_px=200)
        if not rect_s:
            continue

        # ---------- 2) Model crop rect ----------
        cap_m = cv2.VideoCapture(str(DEFAULT_MODEL_CLIP))
        okm, firstm = cap_m.read()
        cap_m.release()
        if not okm or firstm is None:
            continue
        hm, wm = firstm.shape[:2]
        rect_m = global_bbox_from_csv(str(reference_csv), wm, hm, margin_px=200)
        if not rect_m:
            continue

        # ---------- 3) Expand rects to same W/H ----------
        TW = max(rect_s[2] - rect_s[0], rect_m[2] - rect_m[0])
        TH = max(rect_s[3] - rect_s[1], rect_m[3] - rect_m[1])
        rect_s = _expand_rect_to_size(rect_s, TW, TH, ws, hs)
        rect_m = _expand_rect_to_size(rect_m, TW, TH, wm, hm)

        # ---------- 4) Student cropped assets ----------
        crop_clip = throw_dir / "clip_crop.mp4"
        crop_csv = throw_dir / "clip_crop.csv"
        crop_video_to_rect(str(student_clip), str(crop_clip), rect_s, flip=False)
        crop_csv_to_rect(str(stu_csv), str(crop_csv), rect_s)

        # ---------- 5) Model cropped assets ----------
        model_crop = throw_dir / "model_crop.mp4"
        model_crop_csv = throw_dir / "model_crop.csv"
        crop_video_to_rect(str(DEFAULT_MODEL_CLIP), str(model_crop), rect_m, flip=False)
        crop_csv_to_rect(str(reference_csv), str(model_crop_csv), rect_m)

        # ---------- 6) Auto flip ----------
        stu_facing = _read_facing(str(stu_csv))
        ref_facing = _read_facing(str(reference_csv))
        flip_needed = (stu_facing is not None and ref_facing is not None and stu_facing != ref_facing)

        # ---------- 7) Render compare.mp4 ----------
        out_name = "compare.mp4"
        out_path = throw_dir / out_name

        render_side_by_side_points(
            student_mp4=str(crop_clip),
            model_mp4=str(model_crop),
            map_csv=str(map_csv),
            student_csv=str(crop_csv),
            model_csv=str(model_crop_csv),
            out_path=str(out_path),
            flip_model=flip_needed,
            green_yellow=float(green_yellow),
            yellow_red=float(yellow_red),
        )
        ensure_browser_mp4(Path(out_path))

        # ---------- 8) criteria.json ----------
        criteria_path = throw_dir / "criteria.json"
        criteria_obj = evaluate_criteria(
            student_csv=str(crop_csv),
            model_csv=str(model_crop_csv),
            map_csv=str(map_csv),
            model_windows=None,
        )
        criteria_path.write_text(json.dumps(criteria_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        # ---------- 9) score.json (mean_err -> matching_percent) ----------
        # ---------- 9) score.json (mean_err -> matching_percent) ----------
        score_path = throw_dir / "score.json"

        # --- 【核心修改】基于解剖学(躯干长度)进行动态缩放 ---
        stu_torso = _get_median_torso_length(str(crop_csv))
        mdl_torso = _get_median_torso_length(str(model_crop_csv))
        
        if stu_torso is not None and mdl_torso is not None:
            # 用真实身体比例进行缩放
            dynamic_scale = stu_torso / mdl_torso
        else:
            # 如果没测出躯干(极少见)，降级退回使用画面高度缩放
            dynamic_scale = hs / max(hm, 1)
        # ---------------------------------------------------

        mean_err = compute_mean_normalized_error(
            map_s_to_m=read_map_csv(str(map_csv)),
            stu_xy=_load_xy_pairs_cached(str(crop_csv)),
            mdl_xy=_load_xy_pairs_cached(str(model_crop_csv)),
            ws=ws,
            out_h=hs,
            wm=wm,
            model_scale=dynamic_scale,  # <== 这里传入我们算出的真实身体比例
            flip_model=flip_needed,
            joints=[j for j in MAIN_JOINTS],
        )

        k = 0.20
        l = 1.80
        if not np.isfinite(mean_err):
            matching = 0.0
        elif mean_err <= k:
            matching = 100.0
        elif mean_err >= l:
            matching = 0.0
        else:
            matching = 100.0 * (l - mean_err) / (l - k)

        score_obj = {
            "matching_percent": round(float(matching), 1),
            "mean_normalized_error": round(float(mean_err), 4),
            "mapping": {
                "k_full_score": k,
                "l_zero_score": l,
                "rule": "e<=k ->100; k<e<l -> linear; e>=l ->0",
            },
        }
        score_path.write_text(json.dumps(score_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        # ---------- 10) Update manifest entry ----------
        for tt in manifest_obj.get("throws", []) or []:
            if tt.get("throw_id") == throw_id:
                tt["video"] = f"/runs/{job_id}/throws/{throw_id}/{out_name}"
                tt["compare_video"] = f"/runs/{job_id}/throws/{throw_id}/{out_name}"
                tt["criteria"] = f"/runs/{job_id}/throws/{throw_id}/criteria.json"
                tt["score"] = f"/runs/{job_id}/throws/{throw_id}/score.json"
                break

    manifest_path.write_text(json.dumps(manifest_obj, ensure_ascii=False, indent=2), encoding="utf-8")

# =====================================================================
# 新增模块：自动遍历 Runs 文件夹并提取 Criteria + Score 结果生成 CSV
# =====================================================================
def summarize_all_runs(runs_root: Path, output_csv: str = "batch_summary_report.csv") -> None:
    if not runs_root.exists():
        print(f"[WARN] 找不到运行记录文件夹: {runs_root}")
        return

    summary_data = []

    # 遍历 runs_root 下的所有文件夹 (不论是 T1, 还是 时间戳)
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        
        run_name = run_dir.name
        throws_dir = run_dir / "throws"
        
        if not throws_dir.exists():
            continue
            
        # 遍历每个 throw_xxx 文件夹
        for throw_dir in throws_dir.iterdir():
            if not throw_dir.is_dir() or not throw_dir.name.startswith("throw_"):
                continue
                
            throw_id = throw_dir.name
            
            # 初始化这一行的数据
            row = {"Run": run_name, "Throw": throw_id}

            # --- 1) 读取评分 (Score) 和 原始误差 (Mean Error) ---
            # score.json 存放了 matching_percent 和 mean_normalized_error
            score_path = throw_dir / "score.json"
            score_val = ""
            err_val = ""
            if score_path.exists():
                try:
                    with open(score_path, "r", encoding="utf-8") as f:
                        s_data = json.load(f)
                        
                        # 获取 matching_percent (即 0-100 的得分)
                        val = s_data.get("matching_percent")
                        if val is not None:
                            score_val = str(val)
                            
                        # 获取最原始的 DTW 误差 (Mean Normalized Error)
                        err = s_data.get("mean_normalized_error")
                        if err is not None:
                            err_val = str(err)
                            
                except Exception:
                    pass
                    
            row["Score"] = score_val
            row["Mean_Error"] = err_val

            # --- 2) 读取 Criteria ---
            # 兼容多种命名，优先读取 criteria.json
            json_files_to_check = ["criteria.json", "eval.json"]
            criteria_data = None
            
            for jf in json_files_to_check:
                jf_path = throw_dir / jf
                if jf_path.exists():
                    try:
                        with open(jf_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if "items" in data:
                                criteria_data = data["items"]
                                break
                            elif "criteria" in data:
                                criteria_data = data["criteria"]
                                break
                            elif isinstance(data, list) and len(data) > 0 and "key" in data[0]:
                                criteria_data = data
                                break
                    except Exception:
                        pass
            
            # 如果没找到 criteria 数据，填空
            if not criteria_data:
                for c in ["C1", "C2", "C3", "C4"]:
                    row[f"{c}_Pass"] = "-"
                    row[f"{c}_Reason"] = "No Data"
                summary_data.append(row)
                continue

            for item in criteria_data:
                key = item.get("key")
                if not key:
                    continue
                
                student_result = item.get("student", {})
                passed = student_result.get("passed", False)
                notes = student_result.get("notes", [])
                
                row[f"{key.upper()}_Pass"] = "Y" if passed else "N"
                row[f"{key.upper()}_Reason"] = "" if passed else "; ".join(notes)
            
            summary_data.append(row)

    if not summary_data:
        print("[INFO] 没有找到可用于总结的数据。")
        return
        
    df = pd.DataFrame(summary_data)
    
    # 动态排版列顺序: Run, Throw, Score, Mean_Error, C1..., C2...
    cols = ["Run", "Throw", "Score", "Mean_Error"]
    for c in ["C1", "C2", "C3", "C4"]:
        if f"{c}_Pass" in df.columns: cols.append(f"{c}_Pass")
        if f"{c}_Reason" in df.columns: cols.append(f"{c}_Reason")
    
    valid_cols = [c for c in cols if c in df.columns]
    df = df[valid_cols]
    
    # 按文件夹名字字典序排序
    df = df.sort_values(by=["Run", "Throw"])
    
    out_path = Path(output_csv).resolve()
    
    # 增加文件占用容错
    try:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n[SUCCESS] 所有结果(含评分和真实误差)已汇总至: {out_path}")
    except PermissionError:
        print(f"\n[ERROR] 写入失败！文件 {out_path} 正被其他程序(如 Excel)打开，请关闭后重试。")

def run_batch(
    video_dir: Path,
    *,
    runs_root: Path = RUNS_ROOT,
    reference_csv: Path = DEFAULT_REFERENCE_CSV,
    thresholds_json: Path = DEFAULT_THRESHOLDS_JSON,
    green_yellow: float = 0.25,
    yellow_red: float = 0.50,
    limit: Optional[int] = None,
) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)

    vids = _collect_videos(video_dir)
    if limit is not None:
        vids = vids[: int(limit)]

    print(f"[INFO] video_dir={video_dir}")
    print(f"[INFO] found {len(vids)} videos")
    print(f"[INFO] runs_root={runs_root}")

    for i, src_video in enumerate(vids, start=1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(vids)}] {src_video.name}")

        stem = src_video.stem  # 文件名（不含扩展名）
        job_dir = _new_job_dir_from_stem(runs_root, stem)
        job_id = job_dir.name

        try:
            _write_job_status(job_dir, "running", extra={"source_video": str(src_video)})

            # mimic server: copy upload -> inputs/input.mp4
            dst_video = job_dir / "inputs" / INPUT_VIDEO_NAME
            shutil.copy2(src_video, dst_video)

            # 1) Main pipeline (same flags as server)
            result = run_once(
                video_path=str(dst_video),
                out_dir=str(job_dir),
                reference_csv=str(reference_csv),
                thresholds_json=str(thresholds_json),
                enable_annotation=False,  # IMPORTANT (same as server)
                enable_dtw=True,          # IMPORTANT (align map comes from here)
                show_realtime=False,
            )

            # 2) UI-like postprocess: compare + criteria + score + manifest rewrite
            _ui_like_postprocess_one_job(
                job_dir=job_dir,
                job_id=job_id,
                reference_csv=reference_csv,
                thresholds_json=thresholds_json,
                green_yellow=green_yellow,
                yellow_red=yellow_red,
            )

            _write_job_status(job_dir, "done", extra={"result": result})
            print(f"[DONE] job_id={job_id}")

        except Exception as e:
            _write_job_status(job_dir, "failed", extra={"error": str(e), "traceback": traceback.format_exc()})
            print(f"[FAILED] job_id={job_id} err={e}")

    print("\n" + "=" * 80)
    print("[ALL DONE] 批处理完成，正在生成总结报告...")
    
    # 3) 批处理结束后，自动扫描 runs 文件夹，生成总结 CSV
    summarize_all_runs(runs_root, output_csv="batch_summary_report.csv")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", type=str, required=True, help="Folder containing videos (mp4/mov/m4v)")
    ap.add_argument("--runs_root", type=str, default=str(RUNS_ROOT))
    ap.add_argument("--reference_csv", type=str, default=str(DEFAULT_REFERENCE_CSV))
    ap.add_argument("--thresholds_json", type=str, default=str(DEFAULT_THRESHOLDS_JSON))
    ap.add_argument("--green_yellow", type=float, default=0.25)
    ap.add_argument("--yellow_red", type=float, default=0.50)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run_batch(
        video_dir=Path(args.video_dir),
        runs_root=Path(args.runs_root),
        reference_csv=Path(args.reference_csv),
        thresholds_json=Path(args.thresholds_json),
        green_yellow=args.green_yellow,
        yellow_red=args.yellow_red,
        limit=args.limit,
    )