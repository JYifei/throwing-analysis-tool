from __future__ import annotations

import os
import json
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import uuid4
from datetime import datetime
from fastapi.responses import HTMLResponse

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from src.fsm_new import ensure_browser_mp4


# Import your backend entrypoint
# Make sure your project root is on PYTHONPATH when running uvicorn.
from src.fsm_new import run_once
from src.render_side_by_side import render_side_by_side, update_manifest

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
    enable_annotation: bool = Form(True),
    enable_dtw: bool = Form(True),
    reference_csv: str = Form(str(DEFAULT_REFERENCE_CSV)),
    thresholds_json: str = Form(str(DEFAULT_THRESHOLDS_JSON)),
    sbs_threshold: float = Form(0.35),
    sbs_pause_seconds: float = Form(0.8),
    sbs_min_gap: int = Form(12),
):
    if video.content_type is not None and not video.content_type.startswith("video/"):
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
            enable_annotation=enable_annotation,
            enable_dtw=enable_dtw,
            show_realtime=False,
        )

        # 2) Side-by-side generation + write back to manifest
        try:
            manifest_path = job_dir / "outputs" / "manifest.json"
            if enable_dtw and manifest_path.exists():
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

                    aligned_frame_csv = throw_dir / "clip_pose_aligned_frame.csv"
                    student_annot = throw_dir / "annotated.mp4"
                    student_clip = throw_dir / "clip.mp4"
                    student_video = str(student_annot if student_annot.exists() else student_clip)

                    out_name = "side_by_side.mp4"
                    out_name_paused = "side_by_side_paused.mp4"
                    out_path = throw_dir / out_name
                    out_path_paused = throw_dir / out_name_paused
                    
                    # sanitize UI params
                    sbs_threshold = max(0.0, min(float(sbs_threshold), 1.0))
                    sbs_pause_seconds = max(0.0, min(float(sbs_pause_seconds), 5.0))
                    sbs_min_gap = max(0, int(sbs_min_gap))

                    render_side_by_side(
                        student_mp4=student_video,
                        model_video=str(DEFAULT_MODEL_CLIP),
                        map_csv=str(map_csv),
                        out_path=str(out_path),
                        out_path_paused=(str(out_path_paused) if aligned_frame_csv.exists() else None),
                        aligned_frame_csv=(str(aligned_frame_csv) if aligned_frame_csv.exists() else None),
                        err_threshold=sbs_threshold,
                        pause_seconds=sbs_pause_seconds,
                        min_gap_frames=sbs_min_gap,
                        flip_model=False,
                        debug_text=False,
                    )
                    
                    ensure_browser_mp4(Path(out_path))
                    if aligned_frame_csv.exists():
                        ensure_browser_mp4(Path(out_path_paused))

                    
                    # write ABSOLUTE URL into manifest (same style as "video")
                    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
                    for tt in manifest_obj.get("throws", []) or []:
                        if tt.get("throw_id") == throw_id:
                            tt["side_by_side_video"] = f"/runs/{job_id}/throws/{throw_id}/{out_name}"
                            if aligned_frame_csv.exists():
                                tt["side_by_side_paused_video"] = f"/runs/{job_id}/throws/{throw_id}/{out_name_paused}"
                            break
                    manifest_path.write_text(json.dumps(manifest_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        except Exception as e:
            print(f"[WARN] side-by-side render skipped/failed: {e}")

        _write_job_status(job_dir, "done", extra={"result": result})

        # 3) IMPORTANT: always return JSON here
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
