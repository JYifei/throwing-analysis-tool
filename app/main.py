from __future__ import annotations

import cv2
import os
import json
import shutil
import zipfile
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import uuid4
from datetime import datetime
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from src.fsm_new import ensure_browser_mp4, run_once
from src.render_side_by_side import (
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
from src.criteria_eval import evaluate_criteria



DEFAULT_MODEL_CLIP = Path(os.environ.get("THROWING_MODEL_CLIP", "reference/model_clip.mp4")).resolve()


APP_TITLE = "Throwing Analysis Tool"
RUNS_ROOT = Path(os.environ.get("THROWING_RUNS_ROOT", "runs")).resolve()

# Defaults (you can later move these to config/env)
DEFAULT_REFERENCE_CSV = Path(os.environ.get("THROWING_REFERENCE_CSV", "reference/model.csv")).resolve()
DEFAULT_THRESHOLDS_JSON = Path(os.environ.get("THROWING_THRESHOLDS_JSON", "config/dtw_thresholds.json")).resolve()

# Where uploaded videos live inside each job folder
INPUT_VIDEO_NAME = "input.mp4"


app = FastAPI(title=APP_TITLE)

@app.get("/", response_class=HTMLResponse)
def ui_home():
    html_path = Path(__file__).resolve().parent / "ui.html"
    return html_path.read_text(encoding="utf-8")


# Expose runs/ as static so you can directly open plots/videos by URL if needed
# This is convenient for debugging and simple UI.
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=str(RUNS_ROOT)), name="runs")

def _get_median_torso_length(csv_path: str) -> Optional[float]:
    """
    通过读取 CSV 中的肩膀和髋部坐标，计算该视频中人物躯干的平均(中位数)像素长度。
    """
    try:
        df = pd.read_csv(csv_path)
        lsx = pd.to_numeric(df['left_shoulder_x'], errors='coerce')
        lsy = pd.to_numeric(df['left_shoulder_y'], errors='coerce')
        rsx = pd.to_numeric(df['right_shoulder_x'], errors='coerce')
        rsy = pd.to_numeric(df['right_shoulder_y'], errors='coerce')
        
        lhx = pd.to_numeric(df['left_hip_x'], errors='coerce')
        lhy = pd.to_numeric(df['left_hip_y'], errors='coerce')
        rhx = pd.to_numeric(df['right_hip_x'], errors='coerce')
        rhy = pd.to_numeric(df['right_hip_y'], errors='coerce')
        
        msx, msy = (lsx + rsx) / 2.0, (lsy + rsy) / 2.0
        mhx, mhy = (lhx + rhx) / 2.0, (lhy + rhy) / 2.0
        
        dist = np.sqrt((msx - mhx)**2 + (msy - mhy)**2)
        median_dist = float(np.nanmedian(dist))
        
        return median_dist if np.isfinite(median_dist) and median_dist > 0 else None
    except Exception:
        return None


def _read_facing(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
        if "body_facing" not in df.columns:
            return None
        s = df["body_facing"].dropna().astype(str)
        if s.empty:
            return None
        return s.value_counts().idxmax()  # 多数投票
    except Exception:
        return None


def _new_job_dir() -> Path:
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    job_dir = RUNS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    (job_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (job_dir / "outputs").mkdir(parents=True, exist_ok=True)
    return job_dir


def _save_upload(file: UploadFile, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)


def _write_job_status(job_dir: Path, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
    obj = {"status": status}
    if extra:
        obj.update(extra)
    with (job_dir / "outputs" / "job_status.json").open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@app.get("/health")
def health():
    return {"ok": True, "runs_root": str(RUNS_ROOT)}


@app.post("/api/jobs")
def create_job(
    video: UploadFile = File(...),
    reference_csv: str = Form(str(DEFAULT_REFERENCE_CSV)),
    thresholds_json: str = Form(str(DEFAULT_THRESHOLDS_JSON)),
    green_yellow: float = Form(0.25),  # normalized distance
    yellow_red: float = Form(0.50),
):

    if video.content_type is not None and not video.content_type.startswith("video/"):
        # allow anyway; some browsers send application/octet-stream
        pass

    job_dir = _new_job_dir()
    job_id = job_dir.name

    try:
        _write_job_status(job_dir, "running")

        video_path = job_dir / "inputs" / INPUT_VIDEO_NAME
        _save_upload(video, video_path)

        # 1) Main pipeline
        result = run_once(
            video_path=str(video_path),
            out_dir=str(job_dir),
            reference_csv=reference_csv,
            thresholds_json=thresholds_json,
            enable_annotation=False,  # IMPORTANT
            enable_dtw=True,          # IMPORTANT (align map comes from here)
            show_realtime=False,
        )


        # 2) Side-by-side generation + write back to manifest
        try:
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

                # ---------- 3) Expand rects to same W/H so coordinate spaces are comparable ----------
                TW = max(rect_s[2] - rect_s[0], rect_m[2] - rect_m[0])
                TH = max(rect_s[3] - rect_s[1], rect_m[3] - rect_m[1])
                rect_s = _expand_rect_to_size(rect_s, TW, TH, ws, hs)
                rect_m = _expand_rect_to_size(rect_m, TW, TH, wm, hm)
                
                
                # ---------- 4) Student cropped assets (MOVED HERE: AFTER expand) ----------
                crop_clip = throw_dir / "clip_crop.mp4"
                crop_csv = throw_dir / "clip_crop.csv"

                # 建议先覆盖生成，避免旧文件与新 rect 不一致
                crop_video_to_rect(str(student_clip), str(crop_clip), rect_s, flip=False)
                crop_csv_to_rect(str(stu_csv), str(crop_csv), rect_s)

                # ---------- 5) Build model crop video + crop CSV (NEW) ----------
                model_crop = throw_dir / "model_crop.mp4"
                model_crop_csv = throw_dir / "model_crop.csv"

                crop_video_to_rect(str(DEFAULT_MODEL_CLIP), str(model_crop), rect_m, flip=False)
                crop_csv_to_rect(str(reference_csv), str(model_crop_csv), rect_m)

                # ---------- 6) Auto flip decision ----------
                flip_needed = False
                stu_facing = _read_facing(str(stu_csv))
                ref_facing = _read_facing(str(reference_csv))
                flip_needed = (stu_facing is not None and ref_facing is not None and stu_facing != ref_facing)

                # ---------- 7) Render final video (ONLY ONE MODE) ----------
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

                # ---------- 8) Compute criteria.json (REAL) ----------
                criteria_path = throw_dir / "criteria.json"

                criteria_obj = evaluate_criteria(
                    student_csv=str(crop_csv),          # clip_crop.csv
                    model_csv=str(model_crop_csv),      # model_crop.csv
                    map_csv=str(map_csv),               # clip_pose_align_map.csv
                    model_windows=None,                 # 先用 src/criteria_eval.py 里的 DEFAULT_MODEL_WINDOWS
                )

                criteria_path.write_text(
                    json.dumps(criteria_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )


                # ---------- 9) score.json (mean_err -> matching_percent) ----------
                score_path = throw_dir / "score.json"

                # --- 【同步修改】基于解剖学(躯干长度)进行动态缩放 ---
                stu_torso = _get_median_torso_length(str(crop_csv))
                mdl_torso = _get_median_torso_length(str(model_crop_csv))
                
                if stu_torso is not None and mdl_torso is not None:
                    dynamic_scale = stu_torso / mdl_torso
                else:
                    dynamic_scale = hs / max(hm, 1)
                # ---------------------------------------------------

                mean_err = compute_mean_normalized_error(
                    map_s_to_m=read_map_csv(str(map_csv)),
                    stu_xy=_load_xy_pairs_cached(str(crop_csv)),
                    mdl_xy=_load_xy_pairs_cached(str(model_crop_csv)),
                    ws=ws,
                    out_h=hs,
                    wm=wm,
                    model_scale=dynamic_scale,  # <== 使用人体物理比例缩放
                    flip_model=flip_needed,
                    joints=[j for j in MAIN_JOINTS],
                )

                k = 0.10
                l = 1.30
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
                        "rule": f"e<={k} ->100; {k}<e<{l} -> linear; e>={l} ->0",
                    },
                }
                score_path.write_text(json.dumps(score_obj, ensure_ascii=False, indent=2), encoding="utf-8")


                # ---------- 9) Update manifest entry ----------
                for tt in manifest_obj.get("throws", []) or []:
                    if tt.get("throw_id") == throw_id:
                        # Keep original "video" (usually clip.mp4) for backward compatibility
                        tt["video"] = f"/runs/{job_id}/throws/{throw_id}/{out_name}"
                        tt["compare_video"] = f"/runs/{job_id}/throws/{throw_id}/{out_name}"
                        tt["criteria"] = f"/runs/{job_id}/throws/{throw_id}/criteria.json"
                        tt["score"] = f"/runs/{job_id}/throws/{throw_id}/score.json"
                        break

            # write manifest once
            manifest_path.write_text(json.dumps(manifest_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        except Exception as e:
            print(f"[WARN] side-by-side render skipped/failed: {e}")

        _write_job_status(job_dir, "done", extra={"result": result})

        return JSONResponse(
            {
                "job_id": job_id,
                "status": "done",
                "result": result,
                "job_url": f"/api/jobs/{job_id}",
                "download_url": f"/api/jobs/{job_id}/download",
            }
        )

    except Exception as e:
        _write_job_status(job_dir, "failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Job failed: {e}")


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job_dir = RUNS_ROOT / job_id
    status_path = job_dir / "outputs" / "job_status.json"
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    if status_path.exists():
        with status_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj

    # fallback if status missing
    return {"status": "unknown"}



@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str):
    job_dir = RUNS_ROOT / job_id
    if not job_dir.exists() or not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")

    # (2) Only allow download when job is done (prevents partial/failed exports)
    status_path = job_dir / "outputs" / "job_status.json"
    if status_path.exists():
        try:
            status_obj = json.loads(status_path.read_text(encoding="utf-8"))
            status = status_obj.get("status")
            if status != "done":
                raise HTTPException(status_code=409, detail=f"Job not ready: {status}")
        except HTTPException:
            raise
        except Exception as e:
            # If status file is corrupted/unreadable, treat as not ready to avoid confusing exports
            raise HTTPException(status_code=409, detail=f"Job status unreadable: {e}")
    else:
        # If status file doesn't exist, job is likely running/failed; avoid exporting partial results
        raise HTTPException(status_code=409, detail="Job not ready: missing job_status.json")

    # (3) Put zip in a dedicated temp subfolder (cleaner & avoids self-including recursion)
    tmp_root = Path(tempfile.gettempdir()) / "throwing-analysis-tool"
    tmp_root.mkdir(parents=True, exist_ok=True)

    zip_path = tmp_root / f"{job_id}_results.zip"

    # Remove any existing zip (avoid returning stale results)
    if zip_path.exists():
        zip_path.unlink()

    # Create zip
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in job_dir.rglob("*"):
            if p.is_file():
                # Ensure zip uses relative paths inside the archive
                zf.write(p, arcname=p.relative_to(job_dir))

    return FileResponse(
        path=str(zip_path),
        filename=f"{job_id}_results.zip",
        media_type="application/zip",
    )
