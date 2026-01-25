# annotate_basic.py
import json
from pathlib import Path
from collections import deque
import re

import cv2
import numpy as np
import pandas as pd
import os
import subprocess



NAMED_CONNECTIONS = [
    ("left_shoulder","left_elbow"),
    ("left_elbow","left_wrist"),
    ("right_shoulder","right_elbow"),
    ("right_elbow","right_wrist"),
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_hip"),
    ("right_shoulder","right_hip"),
    ("left_hip","right_hip"),
    ("left_hip","left_knee"),
    ("left_knee","left_ankle"),
    ("left_ankle","left_foot_index"),
    ("right_hip","right_knee"),
    ("right_knee","right_ankle"),
    ("right_ankle","right_foot_index"),
]

print("[DEBUG] annotate_basic loaded from:", __file__)

    
def ensure_browser_mp4(mp4_path: Path, ffmpeg_bin: str = "ffmpeg") -> bool:
    """
    Normalize MP4 for browser playback:
    - H.264 (libx264)
    - yuv420p pixel format
    - faststart (moov atom at beginning)
    - no audio (-an) for maximum compatibility
    """
    mp4_path = Path(mp4_path)
    if not mp4_path.exists():
        print(f"[WARN] ensure_browser_mp4: file not found: {mp4_path}")
        return False

    tmp_path = mp4_path.with_suffix(".tmp.mp4")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(mp4_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "veryfast",
        "-crf", "23",
        "-an",
        str(tmp_path),
    ]

    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            print("[WARN] ensure_browser_mp4 failed:")
            print("  file:", mp4_path)
            print("  cmd :", " ".join(cmd))
            err_lines = (completed.stderr or "").splitlines()
            print("  ffmpeg stderr tail:\n", "\n".join(err_lines[-30:]))
            if tmp_path.exists():
                try: tmp_path.unlink()
                except Exception: pass
            return False

        os.replace(str(tmp_path), str(mp4_path))
        print(f"[INFO] ensure_browser_mp4 OK: {mp4_path}")
        return True

    except FileNotFoundError:
        print(f"[WARN] ensure_browser_mp4: ffmpeg not found: {ffmpeg_bin}")
        return False
    except Exception as e:
        print(f"[WARN] ensure_browser_mp4 unexpected error: {type(e).__name__}: {e}")
        if tmp_path.exists():
            try: tmp_path.unlink()
            except Exception: pass
        return False


def _discover_keypoints_columns(df: pd.DataFrame):
    """
    Try to discover keypoint columns in clip.csv robustly.
    Expected patterns:
      - kp{idx}_x, kp{idx}_y (and optional kp{idx}_v / kp{idx}_score)
      - landmark_{idx}_x, landmark_{idx}_y
      - pose_{idx}_x, pose_{idx}_y
    Returns:
      - a list of tuples: [(idx, x_col, y_col, conf_col_or_None), ...] sorted by idx
    """
    patterns = [
        (re.compile(r"^kp(\d+)_x$"), "kp"),
        (re.compile(r"^landmark_(\d+)_x$"), "landmark"),
        (re.compile(r"^pose_(\d+)_x$"), "pose"),
    ]

    cols = set(df.columns)
    found = []

    for rx, _tag in patterns:
        for c in df.columns:
            m = rx.match(c)
            if not m:
                continue
            idx = int(m.group(1))
            x_col = c
            # infer y col
            y_col = x_col.replace("_x", "_y")
            if y_col not in cols:
                continue
            # optional confidence columns
            conf_candidates = [
                x_col.replace("_x", "_v"),
                x_col.replace("_x", "_vis"),
                x_col.replace("_x", "_score"),
                x_col.replace("_x", "_conf"),
            ]
            conf_col = next((cc for cc in conf_candidates if cc in cols), None)
            found.append((idx, x_col, y_col, conf_col))

        if found:
            break

    found.sort(key=lambda t: t[0])
    return found


def _discover_ball_columns(df: pd.DataFrame):
    """
    Discover ball columns. Common possibilities:
      - ball_x, ball_y, ball_conf
      - ball_center_x, ball_center_y, ball_score
      - x_ball, y_ball
    """
    cols = set(df.columns)

    candidates = [
        ("ball_x", "ball_y", "ball_conf"),
        ("ball_x", "ball_y", "ball_score"),
        ("ball_center_x", "ball_center_y", "ball_conf"),
        ("ball_center_x", "ball_center_y", "ball_score"),
        ("x_ball", "y_ball", "ball_conf"),
        ("x_ball", "y_ball", "ball_score"),
        ("ball_cx", "ball_cy", "ball_conf"),
    ]
    for x, y, c in candidates:
        if x in cols and y in cols:
            conf = c if c in cols else None
            return x, y, conf

    # fallback: find any pair containing 'ball' and ending with x/y
    x_like = [c for c in df.columns if "ball" in c.lower() and c.lower().endswith(("x", "_x"))]
    for x_col in x_like:
        y_col = x_col[:-1] + "y"
        if y_col in cols:
            return x_col, y_col, None

    return None, None, None

def draw_bracket_between_points(img, p1, p2, offset=14, tick=10, thickness=2, color=(255, 255, 255)):
    """
    Draw a simple bracket-like marker between two points.
    p1, p2: (x, y) in pixel coords
    offset: how far the bracket is away from the segment
    tick: length of the small perpendicular ticks at both ends
    """
    import numpy as np
    import cv2

    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])

    # segment direction
    vx, vy = x2 - x1, y2 - y1
    norm = (vx * vx + vy * vy) ** 0.5
    if norm < 1e-6:
        return

    # unit normal (perpendicular)
    nx, ny = -vy / norm, vx / norm

    # shift the main bracket line away from the segment
    sx1, sy1 = int(x1 + nx * offset), int(y1 + ny * offset)
    sx2, sy2 = int(x2 + nx * offset), int(y2 + ny * offset)

    # main line
    cv2.line(img, (sx1, sy1), (sx2, sy2), color, thickness)

    # ticks at ends (pointing toward the original segment)
    tx1, ty1 = int(sx1 - nx * tick), int(sy1 - ny * tick)
    tx2, ty2 = int(sx2 - nx * tick), int(sy2 - ny * tick)

    cv2.line(img, (sx1, sy1), (tx1, ty1), color, thickness)
    cv2.line(img, (sx2, sy2), (tx2, ty2), color, thickness)

def draw_vertical_ruler(img, x, y1, y2, tick=10, thickness=2, color=(255, 255, 255)):
    """
    Draw a vertical ruler between y1 and y2 at fixed x, with small horizontal ticks.
    x: pixel x
    y1, y2: pixel y
    """
    import cv2

    x = int(x)
    y1 = int(y1)
    y2 = int(y2)
    if y1 == y2:
        return
    if y1 > y2:
        y1, y2 = y2, y1

    # main vertical line
    cv2.line(img, (x, y1), (x, y2), color, thickness)

    # ticks (horizontal)
    cv2.line(img, (x - tick, y1), (x + tick, y1), color, thickness)
    cv2.line(img, (x - tick, y2), (x + tick, y2), color, thickness)

def draw_angle_marker(img, vtx, p1, p2, length=28, thickness=2, color=(255, 255, 255)):
    """
    Draw a minimal angle marker at vtx using two short rays toward p1 and p2.
    vtx, p1, p2: (x, y) pixel coords
    """
    import cv2
    import numpy as np

    vx, vy = float(vtx[0]), float(vtx[1])
    a = np.array([float(p1[0]) - vx, float(p1[1]) - vy], dtype=float)
    b = np.array([float(p2[0]) - vx, float(p2[1]) - vy], dtype=float)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return

    a = a / na
    b = b / nb

    p1s = (int(vx + a[0] * length), int(vy + a[1] * length))
    p2s = (int(vx + b[0] * length), int(vy + b[1] * length))

    cv2.line(img, (int(vx), int(vy)), p1s, color, thickness)
    cv2.line(img, (int(vx), int(vy)), p2s, color, thickness)


def discover_named_points(df):
    names = [
        "left_shoulder","right_shoulder",
        "left_elbow","right_elbow",
        "left_wrist","right_wrist",
        "left_hip","right_hip",
        "left_knee","right_knee",
        "left_ankle","right_ankle",
        "left_foot_index","right_foot_index",
    ]
    pts = {}
    for n in names:
        x = f"{n}_x"
        y = f"{n}_y"
        if x in df.columns and y in df.columns:
            pts[n] = (x, y)
    return pts

def draw_dashed_line(img, p1, p2, dash_len=10, gap_len=8, thickness=2, color=(255, 255, 255)):
    """
    Draw a dashed line from p1 to p2.
    p1, p2: (x, y) pixel coords
    """
    import cv2
    import numpy as np

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    dx, dy = x2 - x1, y2 - y1
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 1e-6:
        return

    ux, uy = dx / dist, dy / dist
    step = dash_len + gap_len
    n = int(dist // step) + 1

    for i in range(n):
        s = i * step
        e = min(s + dash_len, dist)

        sx, sy = x1 + ux * s, y1 + uy * s
        ex, ey = x1 + ux * e, y1 + uy * e

        cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)), color, thickness)


def annotate_one_throw(
    throw_dir: Path,
    df_clip: pd.DataFrame,
    clip_mp4: Path,
    start_frame: int,
    release_frame: int,
    end_frame: int,
    throw_id: int,
    score_obj: dict | None = None,
    trail_len: int = 10,
    kp_conf_th: float = 0.10,
    ball_conf_th: float = 0.10,
):

    out_path = throw_dir / "annotated.mp4"
    print("[DEBUG] score_obj is None?", score_obj is None)
    if score_obj is not None:
        print("[DEBUG] score_obj keys:", list(score_obj.keys()))
        print("[DEBUG] dtw feature keys:", list((score_obj.get("dtw", {}).get("features", {}) or {}).keys())[:20])


    cap = cv2.VideoCapture(str(clip_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip video: {clip_mp4}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 30.0, (w, h))

    # Discover schema
    named_pts = discover_named_points(df_clip)   # <-- NEW: your named keypoints
    bx, by, bc = _discover_ball_columns(df_clip)


    # Use frame matching if available
    has_frame_col = "frame" in df_clip.columns
    # Build quick lookup from frame->row index
    if has_frame_col:
        frame_to_idx = {int(f): i for i, f in enumerate(df_clip["frame"].values)}
    else:
        frame_to_idx = None

    ball_trail = deque(maxlen=trail_len)

    frame_idx_in_video = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Determine "global frame" for phase labeling:
        # Prefer df_clip["frame"] if present; else approximate by start_frame + local index.
        if has_frame_col:
            # if local index aligns with df rows, use local row’s frame
            # We will try to map via local order: take df row at frame_idx_in_video if exists
            if frame_idx_in_video < len(df_clip):
                global_f = int(df_clip.iloc[frame_idx_in_video]["frame"])
            else:
                global_f = int(start_frame + frame_idx_in_video)
        else:
            global_f = int(start_frame + frame_idx_in_video)

        # Find row data
        if has_frame_col and frame_to_idx is not None and global_f in frame_to_idx:
            row = df_clip.iloc[frame_to_idx[global_f]]
        elif frame_idx_in_video < len(df_clip):
            row = df_clip.iloc[frame_idx_in_video]
        else:
            row = None

        # Overlay header text
        t_sec = frame_idx_in_video / (fps if fps and fps > 0 else 30.0)
        header = f"throw_{throw_id:03d} | local_frame={frame_idx_in_video:04d} | t={t_sec:6.2f}s"
        # Info panel (top-right)
        panel_w = 520
        panel_h = 60
        x0 = max(0, w - panel_w - 16)
        y0 = 12

        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)

        line1 = f"throw_{throw_id:03d}"
        line2 = f"local_frame={frame_idx_in_video:04d}  t={t_sec:6.2f}s"

        cv2.putText(frame, line1, (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, line2, (x0 + 12, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Overlay header text
        t_sec = frame_idx_in_video / (fps if fps and fps > 0 else 30.0)
        header = f"throw_{throw_id:03d} | local_frame={frame_idx_in_video:04d} | t={t_sec:6.2f}s"
        cv2.putText(frame, header, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # DTW summary (minimal, text-only)
        if score_obj is not None:
            try:
                dtw = score_obj.get("dtw", {})
                feats = dtw.get("features", {}) or {}
                overall = dtw.get("overall_matching_score", None)

                angle_feats = {"elbow_angle", "shoulder_angle", "hip_angle"}

                def level_color(level: str):
                    if level == "bad":
                        return (0, 0, 255)      # red
                    return (0, 255, 255)        # warn -> yellow

                # Collect ALL non-ok features
                bad_items = []
                warn_items = []

                for k, v in feats.items():
                    if not isinstance(v, dict):
                        # backward-compat if someone produces float features
                        continue
                    level = v.get("level", "ok")
                    if level == "ok" or level == "unknown":
                        continue

                    try:
                        fv = float(v.get("value"))
                    except Exception:
                        continue

                    # format text
                    if k in angle_feats:
                        text = f"{k}: {fv:.1f} deg"
                    else:
                        text = f"{k}: {fv*100:.1f}%"

                    item = (text, level)
                    if level == "bad":
                        bad_items.append(item)
                    else:
                        warn_items.append(item)

                # Fixed panel lines: overall + BAD then WARN
                lines = []
                colors = []

                if overall is not None:
                    try:
                        ov = float(overall)
                        lines.append(f"DTW overall: {ov*100:.1f}%")
                        colors.append((255, 255, 255))
                    except Exception:
                        pass

                # show all bad first, then all warn
                for text, level in bad_items:
                    lines.append(text)
                    colors.append(level_color(level))
                for text, level in warn_items:
                    lines.append(text)
                    colors.append(level_color(level))

                # draw (fixed location)
                y0 = 52
                for i, s in enumerate(lines):
                    if i >= 8:
                        break  # 防止面板过长；如果你想全显示我们下一步做滚动/分页
                    cv2.putText(frame, s, (16, y0 + 22*i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

            except Exception:
                pass

        # Phase markers
        phase = None
        if global_f == start_frame:
            phase = "START"
        elif global_f == release_frame:
            phase = "RELEASE"
        elif global_f == end_frame:
            phase = "END"

        if phase is not None:
            # Top bar (thin) + small label
            bar_h = 8
            cv2.rectangle(frame, (0, 0), (w, bar_h), (255, 255, 255), -1)
            cv2.putText(frame, phase, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


        # Draw skeleton (named keypoints) - draw for all frames
        if row is not None and named_pts:
            pts = {}
            # draw points
            for name, (x_col, y_col) in named_pts.items():
                x = row.get(x_col, np.nan)
                y = row.get(y_col, np.nan)
                if pd.isna(x) or pd.isna(y):
                    continue
                px = int(float(x))
                py = int(float(y))
                pts[name] = (px, py)
                cv2.circle(frame, (px, py), 3, (255, 255, 255), -1)

            # draw connections
            for a, b in NAMED_CONNECTIONS:
                if a in pts and b in pts:
                    cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1)

        # ===== DTW-driven bracket: feet_lr_dist =====
        if row is not None and score_obj is not None and named_pts:
            try:
                feat = score_obj.get("dtw", {}).get("features", {}).get("feet_lr_dist", None)
                # feat is expected to be a dict like: {"value":..., "level":...}
                if isinstance(feat, dict) and feat.get("level", "ok") != "ok":
                    # Use ankles for stability
                    if "left_ankle" in named_pts and "right_ankle" in named_pts:
                        lx_col, ly_col = named_pts["left_ankle"]
                        rx_col, ry_col = named_pts["right_ankle"]

                        lx, ly = row.get(lx_col, np.nan), row.get(ly_col, np.nan)
                        rx, ry = row.get(rx_col, np.nan), row.get(ry_col, np.nan)

                        if not pd.isna(lx) and not pd.isna(ly) and not pd.isna(rx) and not pd.isna(ry):
                            pL = (float(lx), float(ly))
                            pR = (float(rx), float(ry))
                            level = feat.get("level", "ok")
                            if level == "bad":
                                color = (0, 0, 255)      # red (BGR)
                            else:
                                color = (0, 255, 255)    # warn -> yellow (BGR)
                            draw_bracket_between_points(frame, pL, pR, offset=16, tick=10, thickness=2, color=color)
            except Exception as e:
                print(f"[ANNOTATE][throw_{throw_id:03d}][frame={frame_idx_in_video}] feet_lr_dist failed:", repr(e))


        # ===== DTW-driven vertical ruler: shoulder_to_wrist_y_dist =====
        if row is not None and score_obj is not None and named_pts:
            try:
                feat = score_obj.get("dtw", {}).get("features", {}).get("shoulder_to_wrist_y_dist", None)
                if isinstance(feat, dict) and feat.get("level", "ok") != "ok":
                    level = feat.get("level", "warn")
                    if level == "bad":
                        color = (0, 0, 255)      # red (BGR)
                    else:
                        color = (0, 255, 255)    # warn -> yellow (BGR)

                    # choose side by dominant_side
                    meta = score_obj.get("meta", {}) or {}
                    side = meta.get("dominant_side", "left")  # "left" or "right"

                    sh_name = f"{side}_shoulder"
                    wr_name = f"{side}_wrist"

                    if sh_name in named_pts and wr_name in named_pts:
                        shx_col, shy_col = named_pts[sh_name]
                        wrx_col, wry_col = named_pts[wr_name]

                        shx, shy = row.get(shx_col, np.nan), row.get(shy_col, np.nan)
                        wrx, wry = row.get(wrx_col, np.nan), row.get(wry_col, np.nan)

                        if not pd.isna(shx) and not pd.isna(shy) and not pd.isna(wrx) and not pd.isna(wry):
                            shx, shy = float(shx), float(shy)
                            wrx, wry = float(wrx), float(wry)

                            # place the ruler slightly to the right of the mid-x, to avoid overlapping skeleton
                            rx = wrx
                            draw_vertical_ruler(frame, rx, shy, wry, tick=10, thickness=2, color=color)

            except Exception as e:
                print(f"[ANNOTATE][throw_{throw_id:03d}][frame={frame_idx_in_video}] shoulder_to_wrist_y_dist failed:", repr(e))

            
                # ===== DTW-driven vertical ruler: shoulder_to_elbow_y_dist =====
        if row is not None and score_obj is not None and named_pts:
            try:
                feat = score_obj.get("dtw", {}).get("features", {}).get("shoulder_to_elbow_y_dist", None)
                if isinstance(feat, dict) and feat.get("level", "ok") != "ok":
                    level = feat.get("level", "warn")
                    if level == "bad":
                        color = (0, 0, 255)      # red (BGR)
                    else:
                        color = (0, 255, 255)    # warn -> yellow (BGR)

                    meta = score_obj.get("meta", {}) or {}
                    side = meta.get("dominant_side", "left")  # "left" or "right"

                    sh_name = f"{side}_shoulder"
                    el_name = f"{side}_elbow"

                    if sh_name in named_pts and el_name in named_pts:
                        shx_col, shy_col = named_pts[sh_name]
                        elx_col, ely_col = named_pts[el_name]

                        shx, shy = row.get(shx_col, np.nan), row.get(shy_col, np.nan)
                        elx, ely = row.get(elx_col, np.nan), row.get(ely_col, np.nan)

                        if not pd.isna(shx) and not pd.isna(shy) and not pd.isna(elx) and not pd.isna(ely):
                            shx, shy = float(shx), float(shy)
                            elx, ely = float(elx), float(ely)

                            rx = elx   # 绑定在肘的位置（和你 wrist 绑定风格一致）
                            draw_vertical_ruler(frame, rx, shy, ely, tick=10, thickness=2, color=color)

            except Exception as e:
                print(f"[ANNOTATE][throw_{throw_id:03d}][frame={frame_idx_in_video}] shoulder_to_elbow_y_dist failed:", repr(e))
      
        # ===== DTW-driven dashed line: wrist_to_samehip_dist =====
        if row is not None and score_obj is not None and named_pts:
            try:
                feat = score_obj.get("dtw", {}).get("features", {}).get("wrist_to_samehip_dist", None)
                if isinstance(feat, dict) and feat.get("level", "ok") != "ok":
                    level = feat.get("level", "warn")
                    if level == "bad":
                        color = (0, 0, 255)      # red
                    else:
                        color = (0, 255, 255)    # yellow

                    meta = score_obj.get("meta", {}) or {}
                    side = meta.get("dominant_side", "left")  # "left" or "right"

                    wr_name = f"{side}_wrist"
                    hip_name = f"{side}_hip"

                    if wr_name in named_pts and hip_name in named_pts:
                        wrx_col, wry_col = named_pts[wr_name]
                        hpx_col, hpy_col = named_pts[hip_name]

                        wrx, wry = row.get(wrx_col, np.nan), row.get(wry_col, np.nan)
                        hpx, hpy = row.get(hpx_col, np.nan), row.get(hpy_col, np.nan)

                        if not pd.isna(wrx) and not pd.isna(wry) and not pd.isna(hpx) and not pd.isna(hpy):
                            pW = (float(wrx), float(wry))
                            pH = (float(hpx), float(hpy))

                            # dashed line (usually diagonal)
                            draw_dashed_line(frame, pW, pH, dash_len=10, gap_len=8, thickness=2, color=color)

            except Exception as e:
                print(f"[ANNOTATE][throw_{throw_id:03d}][frame={frame_idx_in_video}] wrist_to_samehip_dist failed:", repr(e))
        
                # ===== DTW-driven dashed line: elbow_to_samehip_dist =====
        if row is not None and score_obj is not None and named_pts:
            try:
                feat = score_obj.get("dtw", {}).get("features", {}).get("elbow_to_samehip_dist", None)
                if isinstance(feat, dict) and feat.get("level", "ok") != "ok":
                    level = feat.get("level", "warn")
                    if level == "bad":
                        color = (0, 0, 255)      # red
                    else:
                        color = (0, 255, 255)    # yellow

                    meta = score_obj.get("meta", {}) or {}
                    side = meta.get("dominant_side", "left")  # "left" or "right"

                    el_name = f"{side}_elbow"
                    hip_name = f"{side}_hip"

                    if el_name in named_pts and hip_name in named_pts:
                        elx_col, ely_col = named_pts[el_name]
                        hpx_col, hpy_col = named_pts[hip_name]

                        elx, ely = row.get(elx_col, np.nan), row.get(ely_col, np.nan)
                        hpx, hpy = row.get(hpx_col, np.nan), row.get(hpy_col, np.nan)

                        if not pd.isna(elx) and not pd.isna(ely) and not pd.isna(hpx) and not pd.isna(hpy):
                            pE = (float(elx), float(ely))
                            pH = (float(hpx), float(hpy))

                            draw_dashed_line(frame, pE, pH, dash_len=10, gap_len=8, thickness=2, color=color)

            except Exception as e:
                print(f"[ANNOTATE][throw_{throw_id:03d}][frame={frame_idx_in_video}] elbow_to_samehip_dist failed:", repr(e))
        
        # ===== DTW-driven angle markers: elbow/shoulder/hip =====
        if row is not None and score_obj is not None and named_pts:
            try:
                feats = score_obj.get("dtw", {}).get("features", {}) or {}
                meta = score_obj.get("meta", {}) or {}
                side = meta.get("dominant_side", "left")  # "left"/"right"

                def level_color(level: str):
                    if level == "bad":
                        return (0, 0, 255)      # red
                    return (0, 255, 255)        # warn -> yellow

                def get_pt(name):
                    if name not in named_pts:
                        return None
                    xcol, ycol = named_pts[name]
                    x, y = row.get(xcol, np.nan), row.get(ycol, np.nan)
                    if pd.isna(x) or pd.isna(y):
                        return None
                    return (float(x), float(y))

                # --- elbow_angle: shoulder - elbow - wrist
                feat = feats.get("elbow_angle", None)
                if isinstance(feat, dict) and feat.get("level", "ok") != "ok":
                    c = level_color(feat.get("level", "warn"))
                    sh = get_pt(f"{side}_shoulder")
                    el = get_pt(f"{side}_elbow")
                    wr = get_pt(f"{side}_wrist")
                    if sh and el and wr:
                        draw_angle_marker(frame, el, sh, wr, length=26, thickness=2, color=c)

                # --- shoulder_angle: hip - shoulder - elbow  (torso vs upper arm)
                feat = feats.get("shoulder_angle", None)
                if isinstance(feat, dict) and feat.get("level", "ok") != "ok":
                    c = level_color(feat.get("level", "warn"))
                    hp = get_pt(f"{side}_hip")
                    sh = get_pt(f"{side}_shoulder")
                    el = get_pt(f"{side}_elbow")
                    if hp and sh and el:
                        draw_angle_marker(frame, sh, hp, el, length=26, thickness=2, color=c)

                # --- hip_angle: shoulder - hip - knee (torso vs thigh)
                feat = feats.get("hip_angle", None)
                if isinstance(feat, dict) and feat.get("level", "ok") != "ok":
                    c = level_color(feat.get("level", "warn"))
                    sh = get_pt(f"{side}_shoulder")
                    hp = get_pt(f"{side}_hip")
                    kn = get_pt(f"{side}_knee")
                    if sh and hp and kn:
                        draw_angle_marker(frame, hp, sh, kn, length=26, thickness=2, color=c)

            except Exception as e:
                print(f"[ANNOTATE][throw_{throw_id:03d}][frame={frame_idx_in_video}] angles failed:", repr(e))

        # Draw ball + trail
        if row is not None and bx and by:
            x = row.get(bx, np.nan)
            y = row.get(by, np.nan)
            conf_ok = True
            if bc is not None:
                cval = row.get(bc, 1.0)
                try:
                    conf_ok = float(cval) >= ball_conf_th
                except Exception:
                    conf_ok = True

            if not pd.isna(x) and not pd.isna(y) and conf_ok:
                px = int(float(x))
                py = int(float(y))
                ball_trail.append((px, py))
                # trail
                for p in list(ball_trail)[:-1]:
                    cv2.circle(frame, p, 2, (255, 255, 255), -1)
                # current
                cv2.circle(frame, (px, py), 6, (255, 255, 255), 2)

            
        writer.write(frame)
        frame_idx_in_video += 1

    cap.release()
    writer.release()

    # Normalize for browser playback (H.264 + yuv420p + faststart)
    ensure_browser_mp4(out_path)

    return out_path



def annotate_from_events(video_dir: Path):
    events_path = video_dir / "events.json"
    if not events_path.exists():
        raise FileNotFoundError(f"events.json not found: {events_path}")

    data = json.loads(events_path.read_text(encoding="utf-8"))
    events = data.get("events", [])

    for ev in events:
        if ev.get("status") != "ok":
            continue

        throw_id = int(ev["throw_id"])
        throw_dir_rel = ev.get("throw_dir", f"throws/throw_{throw_id:03d}")
        throw_dir = video_dir / Path(throw_dir_rel)

        clip_video_rel = ev.get("clip_video", str(Path(throw_dir_rel) / "clip.mp4"))
        clip_csv_rel = ev.get("clip_csv", str(Path(throw_dir_rel) / "clip.csv"))

        clip_mp4 = video_dir / Path(clip_video_rel)
        clip_csv = video_dir / Path(clip_csv_rel)

        if not clip_mp4.exists():
            raise FileNotFoundError(f"clip.mp4 not found: {clip_mp4}")
        if not clip_csv.exists():
            raise FileNotFoundError(f"clip.csv not found: {clip_csv}")


        df_clip = pd.read_csv(clip_csv)

        # Optional: load DTW score.json if present
        score_obj = None
        score_path = throw_dir / "score.json"
        if score_path.exists():
            try:
                score_obj = json.loads(score_path.read_text(encoding="utf-8"))
            except Exception:
                score_obj = None

        annotate_one_throw(
            throw_dir=throw_dir,
            df_clip=df_clip,
            clip_mp4=clip_mp4,
            start_frame=int(ev["start_frame"]),
            release_frame=int(ev["release_frame"]),
            end_frame=int(ev["end_frame"]),
            throw_id=throw_id,
            score_obj=score_obj,
        )



if __name__ == "__main__":
    # Usage:
    #   python annotate_basic.py G:\Throwing_Project_v2\output\Eddie Throwing iPad recording 1
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python annotate_basic.py <video_dir>")
    annotate_from_events(Path(sys.argv[1]))
