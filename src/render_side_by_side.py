import os
import argparse
import cv2
import numpy as np
import pandas as pd

MAIN_JOINTS = [
    "left_ankle", "right_ankle",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
]


def read_map_csv(map_csv_path: str) -> np.ndarray:
    """
    Read stu_idx -> ref_idx mapping.
    Expected columns: stu_idx, ref_idx

    Returns an array map_s_to_m where:
      map_s_to_m[stu_idx] = ref_idx, or -1 if missing.
    """
    df = pd.read_csv(map_csv_path)
    if "stu_idx" not in df.columns or "ref_idx" not in df.columns:
        raise ValueError(f"map csv must contain stu_idx/ref_idx, got {df.columns.tolist()}")

    df = df.sort_values("stu_idx")
    stu = df["stu_idx"].astype(int).to_numpy()
    ref = df["ref_idx"].astype(int).to_numpy()

    if len(stu) == 0:
        return np.array([], dtype=int)

    max_stu = int(stu.max())
    out = np.full((max_stu + 1,), -1, dtype=int)
    out[stu] = ref
    return out


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

    # If caller wants "no resize" and heights already match, concat directly.
    if out_h is None and hs == hm:
        return np.concatenate([student_frame, model_frame], axis=1)

    target_h = out_h if out_h is not None else min(hs, hm)

    student_resized = cv2.resize(student_frame, (int(ws * target_h / hs), target_h))
    model_resized = cv2.resize(model_frame, (int(wm * target_h / hm), target_h))

    return np.concatenate([student_resized, model_resized], axis=1)

def _get_xy_safe(df, idx: int, joint: str):
    """Return (x,y) in pixels or None if missing."""
    xcol = f"{joint}_x"
    ycol = f"{joint}_y"
    if xcol not in df.columns or ycol not in df.columns:
        return None
    if idx < 0 or idx >= len(df):
        return None
    x = df.at[idx, xcol]
    y = df.at[idx, ycol]
    try:
        if pd.isna(x) or pd.isna(y):
            return None
    except Exception:
        pass
    return float(x), float(y)


def _discover_xy_pairs(df: pd.DataFrame):
    cols = set(df.columns)
    pairs = []
    for c in df.columns:
        if not c.endswith("_x"):
            continue
        y = c[:-2] + "_y"
        if y not in cols:
            continue
        if "ball" in c.lower():  # optional: exclude ball
            continue
        pairs.append((c, y))
    return pairs

def _flip_x(x: float, w: int) -> float:
    return float(w - 1) - float(x)


def _flip_rect_horiz(rect, frame_w: int):
    x1, y1, x2, y2 = rect
    nx1 = frame_w - x2
    nx2 = frame_w - x1
    return (nx1, y1, nx2, y2)


def _expand_rect_to_size(rect, target_w: int, target_h: int, frame_w: int, frame_h: int):
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    nx1 = int(round(cx - target_w / 2.0))
    nx2 = nx1 + target_w
    ny1 = int(round(cy - target_h / 2.0))
    ny2 = ny1 + target_h

    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > frame_w:
        shift = nx2 - frame_w
        nx1 -= shift
        nx2 = frame_w
        nx1 = max(0, nx1)
    if ny2 > frame_h:
        shift = ny2 - frame_h
        ny1 -= shift
        ny2 = frame_h
        ny1 = max(0, ny1)

    nx2 = max(nx1 + 1, nx2)
    ny2 = max(ny1 + 1, ny2)
    return (nx1, ny1, nx2, ny2)


def _crop(frame: np.ndarray, rect):
    x1, y1, x2, y2 = rect
    return frame[y1:y2, x1:x2]

def global_bbox_from_csv(csv_path: str, frame_w: int, frame_h: int, margin_px: int = 40):
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    pairs = _discover_xy_pairs(df)
    if not pairs:
        return None

    xs, ys = [], []
    for x_col, y_col in pairs:
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.any():
            xs.append(x[m])
            ys.append(y[m])

    if not xs:
        return None

    x_min = float(np.min(np.concatenate(xs)))
    x_max = float(np.max(np.concatenate(xs)))
    y_min = float(np.min(np.concatenate(ys)))
    y_max = float(np.max(np.concatenate(ys)))

    x1 = int(np.floor(x_min - margin_px))
    y1 = int(np.floor(y_min - margin_px))
    x2 = int(np.ceil (x_max + margin_px))
    y2 = int(np.ceil (y_max + margin_px))

    # clamp
    x1 = max(0, min(x1, frame_w - 2))
    y1 = max(0, min(y1, frame_h - 2))
    x2 = max(x1 + 1, min(x2, frame_w - 1))
    y2 = max(y1 + 1, min(y2, frame_h - 1))
    return (x1, y1, x2, y2)


def crop_video_to_rect(in_mp4: str, out_mp4: str, rect, flip: bool = False):
    cap = cv2.VideoCapture(in_mp4)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_mp4}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0

    x1, y1, x2, y2 = rect
    out_w = int(x2 - x1)
    out_h = int(y2 - y1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (out_w, out_h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        frame = frame[y1:y2, x1:x2]
        writer.write(frame)

    cap.release()
    writer.release()


def crop_csv_to_rect(in_csv: str, out_csv: str, rect):
    x1, y1, x2, y2 = rect
    df = pd.read_csv(in_csv)

    # shift all *_x/_y
    for c in df.columns:
        if c.endswith("_x"):
            df[c] = pd.to_numeric(df[c], errors="coerce") - x1
        elif c.endswith("_y"):
            df[c] = pd.to_numeric(df[c], errors="coerce") - y1

    df.to_csv(out_csv, index=False)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--throw_dir", required=True, help="Path to throw_001 folder containing clip.mp4 and csvs")
    ap.add_argument("--model_video", required=True, help="Path to reference model clip video (mp4/mov)")
    ap.add_argument("--student_video", required=True, help="Path to student clip video (mp4)")
    ap.add_argument("--flip_model", action="store_true", help="Flip model horizontally for display")
    ap.add_argument("--green_yellow", type=float, default=0.25, help="Normalized error threshold for green->yellow")
    ap.add_argument("--yellow_red", type=float, default=0.50, help="Normalized error threshold for yellow->red")
    args = ap.parse_args()

    throw_dir = args.throw_dir
    map_csv = os.path.join(throw_dir, "clip_pose_align_map.csv")

    out_path = os.path.join(throw_dir, "compare.mp4")

    render_side_by_side_points(
        student_mp4=args.student_video,
        model_mp4=args.model_video,
        map_csv=map_csv,
        student_csv=os.path.join(throw_dir, "clip_crop.csv"),
        model_csv=os.path.join(throw_dir, "model_crop.csv"),
        out_path=out_path,
        flip_model=args.flip_model,
        green_yellow=args.green_yellow,
        yellow_red=args.yellow_red,
    )

def _load_xy_pairs_cached(csv_path: str):
    """Load *_x/_y columns once to speed up per-frame access."""
    df = pd.read_csv(csv_path)
    pairs = _discover_xy_pairs(df)
    data = {}
    for x_col, y_col in pairs:
        xs = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        ys = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        data[(x_col, y_col)] = (xs, ys)
    return data


def _color_from_err(err: float, gy: float, yr: float):
    """
    Map normalized error -> BGR color.
    - green if <= gy
    - yellow if in (gy, yr]
    - red if > yr
    """
    if not np.isfinite(err):
        return (0, 255, 0)
    if err <= gy:
        return (0, 255, 0)        # green
    if err <= yr:
        return (0, 255, 255)      # yellow
    return (0, 0, 255)            # red


def compute_mean_normalized_error(
    *,
    map_s_to_m: np.ndarray,
    stu_xy: dict,
    mdl_xy: dict,
    ws: int,
    out_h: int,
    wm: int,
    model_scale: float,
    flip_model: bool,
    joints: list[str],
) -> float:
    """
    Compute mean normalized joint error over all aligned frames and joints.
    Uses the SAME error definition as the point coloring logic.
    Returns mean error (float). If no valid joints found, returns +inf.
    """
    errs = []

    # we assume stu_xy values are (xs, ys) arrays
    # pick any one to estimate length
    any_key = next(iter(stu_xy.keys()), None)
    if any_key is None:
        return float("inf")
    stu_len = len(stu_xy[any_key][0])

    n_use = min(len(map_s_to_m), stu_len)

    for s_idx in range(n_use):
        m_idx = int(map_s_to_m[s_idx])
        if m_idx < 0:
            continue

        for joint in joints:
            x_col = joint + "_x"
            y_col = joint + "_y"
            if (x_col, y_col) not in stu_xy or (x_col, y_col) not in mdl_xy:
                continue

            sx_arr, sy_arr = stu_xy[(x_col, y_col)]
            mx_arr, my_arr = mdl_xy[(x_col, y_col)]

            if s_idx >= len(sx_arr) or m_idx >= len(mx_arr):
                continue

            sx, sy = sx_arr[s_idx], sy_arr[s_idx]
            mx, my = mx_arr[m_idx], my_arr[m_idx]

            if not np.isfinite(sx) or not np.isfinite(sy) or not np.isfinite(mx) or not np.isfinite(my):
                continue

            if flip_model:
                mx = (wm - 1) - mx

            # model coords must be scaled the same way as when resizing model frame
            mx *= model_scale
            my *= model_scale

            dx = (sx - mx) / max(ws, 1)
            dy = (sy - my) / max(out_h, 1)
            err = float(np.sqrt(dx * dx + dy * dy))
            errs.append(err)

    if not errs:
        return float("inf")

    return float(np.mean(errs))

def render_side_by_side_points(
    student_mp4: str,
    model_mp4: str,
    map_csv: str,
    student_csv: str,
    model_csv: str,
    out_path: str,
    *,
    flip_model: bool = False,
    green_yellow: float = 0.25,
    yellow_red: float = 0.50,
    point_radius: int = 5,
):
    """
    Render side-by-side video:
    Left: student frame with colored joint points (no text annotations).
    Right: model frame (optionally flipped).
    Color is determined by per-joint deviation from model (normalized by frame size).
    """

    map_s_to_m = read_map_csv(map_csv)

    cap_s = cv2.VideoCapture(student_mp4)
    cap_m = cv2.VideoCapture(model_mp4)
    if not cap_s.isOpened():
        raise RuntimeError(f"Cannot open student video: {student_mp4}")
    if not cap_m.isOpened():
        raise RuntimeError(f"Cannot open model video: {model_mp4}")

    fps = cap_s.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0

    # Read first frames to get sizes
    first_s = get_frame_at(cap_s, 0)
    first_m = get_frame_at(cap_m, int(map_s_to_m[0]) if len(map_s_to_m) > 0 else 0)
    if first_s is None or first_m is None:
        raise RuntimeError("Cannot read first frames")
    hs, ws = first_s.shape[:2]
    hm, wm = first_m.shape[:2]

    # Cache pose arrays
    stu_xy = _load_xy_pairs_cached(student_csv)
    mdl_xy = _load_xy_pairs_cached(model_csv)

    # Use only the joint set that exists in BOTH csvs
    common = []
    for joint in MAIN_JOINTS:
        x_col = joint + "_x"
        y_col = joint + "_y"
        if (x_col, y_col) in stu_xy and (x_col, y_col) in mdl_xy:
            common.append((x_col, y_col))
    if not common:
        raise RuntimeError("No common *_x/_y joints found between student_csv and model_csv")

    # Output writer (keep student height as reference)
    out_h = hs
    out_w = ws + int(wm * out_h / hm)
    model_scale = out_h / max(hm, 1)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    n_s = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT))
    n_use = min(n_s, len(map_s_to_m))
    if n_use <= 0:
        raise RuntimeError("Alignment map is empty or student video has no frames")

    for s_idx in range(n_use):
        m_idx = int(map_s_to_m[s_idx])
        if m_idx < 0:
            continue
        sf = get_frame_at(cap_s, s_idx)
        mf = get_frame_at(cap_m, m_idx)
        if sf is None or mf is None:
            break

        if flip_model:
            mf = cv2.flip(mf, 1)

        # Draw points on student frame
        for (x_col, y_col) in common:
            sx_arr, sy_arr = stu_xy[(x_col, y_col)]
            mx_arr, my_arr = mdl_xy[(x_col, y_col)]

            if s_idx >= len(sx_arr) or m_idx >= len(mx_arr):
                continue

            sx, sy = sx_arr[s_idx], sy_arr[s_idx]
            mx, my = mx_arr[m_idx], my_arr[m_idx]
            if not (np.isfinite(sx) and np.isfinite(sy) and np.isfinite(mx) and np.isfinite(my)):
                continue

            # If model is flipped, flip in its ORIGINAL coordinate space first
            # If model is flipped, flip in its ORIGINAL coordinate space first
            if flip_model:
                mx = (wm - 1) - mx

            # === 修改点 1：保存原始尺寸下的坐标，专门用于在 mf 上画图 ===
            draw_mx = mx
            draw_my = my

            # ✅ then scale model coords into the OUTPUT (resized) coordinate space
            # 下面的 mx, my 将只用于计算与学生的误差 dx, dy
            mx = mx * model_scale
            my = my * model_scale


            # Normalize by frame size (crop rects were expanded to same W/H upstream)
            dx = (sx - mx) / max(ws, 1)
            dy = (sy - my) / max(out_h, 1)
            err = float(np.sqrt(dx * dx + dy * dy))

            color = _color_from_err(err, green_yellow, yellow_red)
            #cv2.circle(sf, (int(round(sx)), int(round(sy))), point_radius, color, thickness=-1)
            
            # === 修改点 2：使用未缩放的 draw_mx 和 draw_my 画白点 ===
            #cv2.circle(mf, (int(round(draw_mx)), int(round(draw_my))), point_radius, (200, 200, 200), thickness=-1)
        # Concatenate and write
        sbs = make_side_by_side(sf, mf, out_h=out_h)
        if sbs is None:
            break
        writer.write(sbs)

    cap_s.release()
    cap_m.release()
    writer.release()


if __name__ == "__main__":
    main()
