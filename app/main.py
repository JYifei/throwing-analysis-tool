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

# Import your backend entrypoint
# Make sure your project root is on PYTHONPATH when running uvicorn.
from src.fsm_new import run_once


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
    # optional toggles; keep minimal
    enable_annotation: bool = Form(True),
    enable_dtw: bool = Form(True),
    # optional override paths (advanced users only; can hide in UI later)
    reference_csv: str = Form(str(DEFAULT_REFERENCE_CSV)),
    thresholds_json: str = Form(str(DEFAULT_THRESHOLDS_JSON)),
):
    """
    Minimal synchronous job runner:
    - upload video
    - run run_once()
    - return result dict
    """
    # basic validation
    if video.content_type is not None and not video.content_type.startswith("video/"):
        # still allow if browser doesn't provide content_type properly; up to you
        pass

    job_dir = _new_job_dir()
    job_id = job_dir.name

    try:
        _write_job_status(job_dir, "running")

        video_path = job_dir / "inputs" / INPUT_VIDEO_NAME
        _save_upload(video, video_path)

        # Call your backend pipeline (synchronous)
        result = run_once(
            video_path=str(video_path),
            out_dir=str(job_dir),
            reference_csv=reference_csv,
            thresholds_json=thresholds_json,
            enable_annotation=enable_annotation,
            enable_dtw=enable_dtw,
            show_realtime=False,  # hard-disable in app mode
        )

        _write_job_status(job_dir, "done", extra={"result": result})

        return JSONResponse(
            {
                "job_id": job_id,
                "status": "done",
                "result": result,
                # handy URLs
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
