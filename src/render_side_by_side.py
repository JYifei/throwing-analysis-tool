import os
import argparse
import cv2
import numpy as np
import pandas as pd
import json


ANGLE_COLS = {"elbow_angle", "shoulder_angle", "hip_angle"}
ANGLE_MAX = 180.0


def read_map_csv(map_csv_path: str) -> np.ndarray:
    """
    Read stu_idx -> ref_idx mapping.
    Expected columns: stu_idx, ref_idx
    """
    df = pd.read_csv(map_csv_path)
    if "stu_idx" not in df.columns or "ref_idx" not in df.columns:
        raise ValueError(f"map csv must contain stu_idx/ref_idx, got {df.columns.tolist()}")
    df = df.sort_values("stu_idx")
    return df["ref_idx"].astype(int).to_numpy()


def compute_overall_err(aligned_frame_csv: str) -> np.ndarray:
    """
    Compute per-frame overall error from aligned per-feature errors.
    - Angle columns are in degrees -> normalize by /180, clip to [0,1]
    - Non-angle columns assumed already in [0,1]
    Returns overall_err array length = n_frames
    """
    df = pd.read_csv(aligned_frame_csv)
    if "frame" not in df.columns:
        raise ValueError("aligned frame csv must contain 'frame' column")

    feat_cols = [c for c in df.columns if c != "frame"]
    if not feat_cols:
        raise ValueError("aligned frame csv has no feature columns")

    x = df[feat_cols].to_numpy(dtype=float)

    # Normalize angles
    for j, c in enumerate(feat_cols):
        if c in ANGLE_COLS:
            x[:, j] = np.clip(x[:, j] / ANGLE_MAX, 0.0, 1.0)
        else:
            x[:, j] = np.clip(x[:, j], 0.0, 1.0)

    overall = np.nanmean(x, axis=1)
    overall = np.where(np.isnan(overall), 0.0, overall)
    return overall


def pick_pause_frames_by_threshold(overall_err: np.ndarray,
                                  threshold: float,
                                  min_gap: int = 12) -> list:
    """
    Hard-threshold pause selection with multiple pauses per long segment:
    - Find contiguous segments where overall_err >= threshold
    - Within each segment, split into chunks of length ~min_gap
      and pick the peak frame (argmax) in each chunk.
    This guarantees multiple pauses if the segment is long enough.
    """
    n = len(overall_err)
    if n == 0:
        return []

    above = overall_err >= threshold
    pause_frames = []

    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue

        # segment [start, end)
        start = i
        while i < n and above[i]:
            i += 1
        end = i

        # pick multiple peaks inside this segment, one per chunk
        j = start
        while j < end:
            chunk_end = min(end, j + max(min_gap, 1))
            seg = overall_err[j:chunk_end]
            peak_offset = int(np.argmax(seg))
            peak_frame = j + peak_offset
            pause_frames.append(peak_frame)
            j = chunk_end

    # de-duplicate and enforce strict ordering
    pause_frames = sorted(set(pause_frames))
    return pause_frames


def get_frame_at(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(idx), 0))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def make_side_by_side(student_frame: np.ndarray,
                      model_frame: np.ndarray,
                      out_h: int = None) -> np.ndarray:
    if student_frame is None or model_frame is None:
        return None

    hs, ws = student_frame.shape[:2]
    hm, wm = model_frame.shape[:2]

    target_h = out_h if out_h is not None else min(hs, hm)

    student_resized = cv2.resize(student_frame, (int(ws * target_h / hs), target_h))
    model_resized = cv2.resize(model_frame, (int(wm * target_h / hm), target_h))

    return np.concatenate([student_resized, model_resized], axis=1)


def update_manifest(throw_dir: str,
                    out_name: str,
                    out_name_paused: str,
                    manifest_path: str) -> None:
    """
    Update outputs/manifest.json: add side-by-side paths for the matching throw entry.
    We match by throw_id == basename(throw_dir) (e.g., throw_001).
    Paths stored as relative to job root (same style as existing manifest).
    """
    if not manifest_path or not os.path.exists(manifest_path):
        return

    throw_id = os.path.basename(os.path.normpath(throw_dir))
    job_root = os.path.dirname(os.path.dirname(os.path.normpath(throw_dir)))  # .../runs/<job_id>

    def rel(p: str) -> str:
        return os.path.relpath(p, job_root).replace("\\", "/")

    sbs_path = os.path.join(throw_dir, out_name)
    sbs_paused_path = os.path.join(throw_dir, out_name_paused)

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return

    throws = manifest.get("throws", [])
    changed = False
    for t in throws:
        if t.get("throw_id") == throw_id:
            t["side_by_side_video"] = rel(sbs_path) if os.path.exists(sbs_path) else t.get("side_by_side_video")
            if os.path.exists(sbs_paused_path):
                t["side_by_side_paused_video"] = rel(sbs_paused_path)
            changed = True
            break

    if changed:
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def render_side_by_side(student_mp4: str,
                        model_video: str,
                        map_csv: str,
                        out_path: str,
                        out_path_paused: str = None,
                        aligned_frame_csv: str = None,
                        err_threshold: float = 0.35,
                        pause_seconds: float = 0.8,
                        min_gap_frames: int = 12,
                        flip_model: bool = False,
                        debug_text: bool = False):
    """
    Render side-by-side synchronized video using pose align map.
    Optionally render a paused version based on hard threshold.
    """

    map_s_to_m = read_map_csv(map_csv)

    cap_s = cv2.VideoCapture(student_mp4)
    cap_m = cv2.VideoCapture(model_video)

    if not cap_s.isOpened():
        raise RuntimeError(f"Cannot open student video: {student_mp4}")
    if not cap_m.isOpened():
        raise RuntimeError(f"Cannot open model video: {model_video}")

    fps = cap_s.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0

    n_s = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT))
    n_m = int(cap_m.get(cv2.CAP_PROP_FRAME_COUNT))

    n_use = min(n_s, len(map_s_to_m)) if len(map_s_to_m) > 0 else 0
    if n_use <= 0:
        raise RuntimeError("Alignment map is empty or student video has no frames")

    pause_frames = []
    pause_repeat = 0
    if out_path_paused and aligned_frame_csv and os.path.exists(aligned_frame_csv):
        overall_err = compute_overall_err(aligned_frame_csv)[:n_use]
        pause_frames = pick_pause_frames_by_threshold(overall_err, err_threshold, min_gap=min_gap_frames)
        pause_repeat = int(round(pause_seconds * fps))

    first_s = get_frame_at(cap_s, 0)
    first_m = get_frame_at(cap_m, int(map_s_to_m[0]))
    if flip_model and first_m is not None:
        first_m = cv2.flip(first_m, 1)

    combo0 = make_side_by_side(first_s, first_m)
    if combo0 is None:
        raise RuntimeError("Cannot read initial frames for output sizing")

    out_h, out_w = combo0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    writer_paused = None
    if out_path_paused:
        writer_paused = cv2.VideoWriter(out_path_paused, fourcc, fps, (out_w, out_h))

    pause_set = set(pause_frames)

    for s_idx in range(n_use):
        m_idx = int(map_s_to_m[s_idx])
        m_idx = max(0, min(m_idx, n_m - 1))

        frame_s = get_frame_at(cap_s, s_idx)
        frame_m = get_frame_at(cap_m, m_idx)

        if frame_s is None or frame_m is None:
            continue

        if flip_model:
            frame_m = cv2.flip(frame_m, 1)

        combo = make_side_by_side(frame_s, frame_m, out_h=out_h)
        if combo is None:
            continue

        if debug_text:
            cv2.putText(combo, f"S:{s_idx}  M:{m_idx}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(combo)

        if writer_paused is not None:
            writer_paused.write(combo)
            if s_idx in pause_set and pause_repeat > 0:
                for _ in range(pause_repeat):
                    writer_paused.write(combo)

    writer.release()
    if writer_paused is not None:
        writer_paused.release()

    cap_s.release()
    cap_m.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--throw_dir", required=True, help="Path to throw_001 folder containing clip.mp4 and csvs")

    # Keep your original CLI, but allow overriding videos explicitly.
    ap.add_argument("--model_video", required=True, help="Path to reference model clip video (mp4/mov)")
    ap.add_argument("--student_video", default=None, help="Override student video path (default: annotated.mp4 if exists else clip.mp4)")
    ap.add_argument("--model_annotated", default=None, help="Optional model annotated video path (preferred for right panel)")

    ap.add_argument("--out_name", default="side_by_side.mp4")
    ap.add_argument("--out_name_paused", default="side_by_side_paused.mp4")
    ap.add_argument("--threshold", type=float, default=0.35, help="Hard threshold on overall_err (0..1)")
    ap.add_argument("--pause_seconds", type=float, default=0.8, help="Pause duration when triggered")
    ap.add_argument("--min_gap", type=int, default=12, help="Min gap between pause triggers (frames)")
    ap.add_argument("--flip_model", action="store_true", help="Flip model horizontally for display")
    ap.add_argument("--debug_text", action="store_true", help="Overlay S/M indices for DTW verification")

    # Manifest integration for webapp
    ap.add_argument("--write_manifest", action="store_true", help="Write side-by-side paths into outputs/manifest.json")
    ap.add_argument("--manifest_path", default=None, help="Override manifest path (default: <job_root>/outputs/manifest.json)")

    args = ap.parse_args()

    throw_dir = args.throw_dir
    map_csv = os.path.join(throw_dir, "clip_pose_align_map.csv")
    aligned_frame_csv = os.path.join(throw_dir, "clip_pose_aligned_frame.csv")

    # Student video: prefer annotated.mp4 if exists, else clip.mp4
    if args.student_video:
        student_mp4 = args.student_video
    else:
        candidate_annot = os.path.join(throw_dir, "annotated.mp4")
        candidate_clip = os.path.join(throw_dir, "clip.mp4")
        student_mp4 = candidate_annot if os.path.exists(candidate_annot) else candidate_clip

    # Model video: prefer model_annotated if provided and exists
    model_video = args.model_annotated if (args.model_annotated and os.path.exists(args.model_annotated)) else args.model_video

    out_path = os.path.join(throw_dir, args.out_name)
    out_path_paused = os.path.join(throw_dir, args.out_name_paused)

    paused_ok = os.path.exists(aligned_frame_csv)

    render_side_by_side(
        student_mp4=student_mp4,
        model_video=model_video,
        map_csv=map_csv,
        out_path=out_path,
        out_path_paused=(out_path_paused if paused_ok else None),
        aligned_frame_csv=(aligned_frame_csv if paused_ok else None),
        err_threshold=args.threshold,
        pause_seconds=args.pause_seconds,
        min_gap_frames=args.min_gap,
        flip_model=args.flip_model,
        debug_text=args.debug_text,
    )

    # Update manifest for webapp
    if args.write_manifest:
        job_root = os.path.dirname(os.path.dirname(os.path.normpath(throw_dir)))
        manifest_path = args.manifest_path if args.manifest_path else os.path.join(job_root, "outputs", "manifest.json")
        update_manifest(
            throw_dir=throw_dir,
            out_name=args.out_name,
            out_name_paused=args.out_name_paused,
            manifest_path=manifest_path,
        )


if __name__ == "__main__":
    main()
