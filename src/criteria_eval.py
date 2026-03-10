# src/criteria_eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


# ===== Model Windows =====
DEFAULT_MODEL_WINDOWS: Dict[str, Tuple[int, int]] = {
    "c1": (0, 10),    
    "c2": (0, 40),    
    "c3": (10, 40),   
    "c4": (40, 55),   
}

@dataclass
class CriterionDebug:
    passed: bool
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    notes: List[str]
    valid_frames: int

def _fail(notes: List[str], *, metrics=None, thresholds=None, valid_frames: int = 0) -> CriterionDebug:
    return CriterionDebug(
        passed=False,
        metrics=metrics or {},
        thresholds=thresholds or {},
        notes=notes,
        valid_frames=int(valid_frames),
    )

def read_majority_facing(csv_path: str, col: str = "body_facing") -> Optional[str]:
    try:
        df = pd.read_csv(csv_path)
        if col not in df.columns:
            return None
        s = df[col].dropna().astype(str)
        if s.empty:
            return None
        return s.value_counts().idxmax()
    except Exception:
        return None

def load_map_stu_to_ref(map_csv: str) -> np.ndarray:
    df = pd.read_csv(map_csv)
    if "stu_idx" not in df.columns or "ref_idx" not in df.columns:
        raise ValueError(f"map csv must contain stu_idx/ref_idx, got {df.columns.tolist()}")
    df = df.sort_values("stu_idx")
    return df["ref_idx"].astype(int).to_numpy()

def invert_map_ref_to_stu(stu_to_ref: np.ndarray) -> Dict[int, List[int]]:
    ref_to_stu: Dict[int, List[int]] = {}
    for stu_idx, ref_idx in enumerate(stu_to_ref.tolist()):
        ref_to_stu.setdefault(int(ref_idx), []).append(int(stu_idx))
    return ref_to_stu

def model_range_to_student_range(
    ref_to_stu: Dict[int, List[int]],
    model_a: int,
    model_b: int,
) -> Optional[Tuple[int, int]]:
    hit: List[int] = []
    for r in range(int(model_a), int(model_b) + 1):
        if r in ref_to_stu:
            hit.extend(ref_to_stu[r])
    if not hit:
        return None
    return (min(hit), max(hit))

def load_xy(df: pd.DataFrame, joint: str) -> Tuple[np.ndarray, np.ndarray]:
    xcol = f"{joint}_x"
    ycol = f"{joint}_y"
    if xcol not in df.columns or ycol not in df.columns:
        raise ValueError(f"missing columns: {xcol}/{ycol}")
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    return x, y

def finite_mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m

def dist2(ax, ay, bx, by) -> np.ndarray:
    return (ax - bx) ** 2 + (ay - by) ** 2

def torso_scale(df: pd.DataFrame) -> np.ndarray:
    lsx, lsy = load_xy(df, "left_shoulder")
    rsx, rsy = load_xy(df, "right_shoulder")
    lhx, lhy = load_xy(df, "left_hip")
    rhx, rhy = load_xy(df, "right_hip")

    msx = (lsx + rsx) / 2.0
    msy = (lsy + rsy) / 2.0
    mhx = (lhx + rhx) / 2.0
    mhy = (lhy + rhy) / 2.0

    m = finite_mask(msx, msy, mhx, mhy)
    out = np.full_like(msx, np.nan, dtype=float)
    out[m] = np.sqrt(dist2(msx[m], msy[m], mhx[m], mhy[m]))
    return out


# =============================================================================
# Criteria implementations
# =============================================================================

def crit1_windup_downward_hand_arm_details(
    df: pd.DataFrame,
    a: int,
    b: int,
    throwing_side: str,
) -> CriterionDebug:
    if throwing_side == "left":
        wrist = "left_wrist"
    else:
        wrist = "right_wrist"

    wx, wy = load_xy(df, wrist)
    scale = torso_scale(df)

    a = max(0, int(a))
    b = min(len(wy) - 1, int(b))
    if b <= a:
        return _fail(["invalid_window"], valid_frames=0)

    seg_wy = wy[a:b+1]
    seg_sc = scale[a:b+1]

    m_w = np.isfinite(seg_wy) & np.isfinite(seg_sc)

    if int(m_w.sum()) < 3:
        return _fail(["too_few_valid_frames"], valid_frames=int(m_w.sum()))

    valid_wy = seg_wy[m_w]

    # First 20% of the preparation window
    kw = max(2, int(len(valid_wy) * 0.2))

    # Starting reference: maximum wrist y in the first 20% (highest position before drop)
    wy0 = float(np.max(valid_wy[:kw]))

    # Lowest wrist position within the full window (largest y)
    wy_max = float(np.max(valid_wy))

    # Robust torso scale normalization
    sc = float(np.nanmedian(seg_sc[np.isfinite(seg_sc)]))
    if not np.isfinite(sc) or sc <= 1e-6:
        sc = 1.0

    # Normalized wrist drop
    dw = float((wy_max - wy0) / sc)

    DW_THR = 0.4

    passed = (dw >= DW_THR)

    return CriterionDebug(
        passed=bool(passed),
        metrics={
            "dw": dw,
            "wy0": wy0,
            "wy_max": wy_max,
            "scale_median": float(sc),
        },
        thresholds={
            "DW_THR": DW_THR,
        },
        notes=[] if passed else ["windup_wrist_drop_not_enough"],
        valid_frames=int(m_w.sum()),
    )

def crit2_rotate_nonthrow_side_faces_target_details(
    df: pd.DataFrame,
    a: int,
    b: int,
    *,
    target_sign: int,
    nonthrow_side: str,
) -> CriterionDebug:
    # Derive throwing side from non-throwing side (to keep call sites unchanged)
    if nonthrow_side == "left":
        throwing_side = "right"
    else:
        throwing_side = "left"

    # Select wrist by throwing side
    if throwing_side == "left":
        wx, wy = load_xy(df, "left_wrist")
    else:
        wx, wy = load_xy(df, "right_wrist")

    lhx, lhy = load_xy(df, "left_hip")
    rhx, rhy = load_xy(df, "right_hip")
    lsx, lsy = load_xy(df, "left_shoulder")
    rsx, rsy = load_xy(df, "right_shoulder")

    scale = torso_scale(df)

    a = max(0, int(a))
    b = min(len(wx) - 1, int(b))
    if b <= a:
        return _fail(["invalid_window"], valid_frames=0)

    seg_wx = wx[a:b+1]
    seg_lhx = lhx[a:b+1]
    seg_rhx = rhx[a:b+1]
    seg_lsx = lsx[a:b+1]
    seg_rsx = rsx[a:b+1]
    seg_sc = scale[a:b+1]

    # Hip center x
    hip_cx = (seg_lhx + seg_rhx) / 2.0

    # Most pull-back moment: maximize target_sign * (hip_cx - wrist_x)
    backness = target_sign * (hip_cx - seg_wx)

    m = (
        np.isfinite(backness) &
        np.isfinite(seg_lsx) & np.isfinite(seg_rsx) &
        np.isfinite(seg_sc)
    )
    if int(m.sum()) < 5:
        return _fail(["too_few_valid_frames"], valid_frames=int(m.sum()))

    backness_valid = backness.copy()
    backness_valid[~m] = -np.inf
    idx_rel = int(np.argmax(backness_valid))
    t_back = a + idx_rel

    # Shoulder length at t_back (normalized)
    shoulder_len = float(np.abs(seg_lsx[idx_rel] - seg_rsx[idx_rel]))
    sc = float(seg_sc[idx_rel])
    if not np.isfinite(sc) or sc <= 1e-6:
        sc = float(np.nanmedian(seg_sc[np.isfinite(seg_sc)]))
        if not np.isfinite(sc) or sc <= 1e-6:
            sc = 1.0

    shoulder_len_norm = float(shoulder_len / sc)

    # Threshold (reuse old OPEN_MIN as a stable starting point)
    SHOULDER_LEN_THR = 0.55
    passed = (shoulder_len_norm >= SHOULDER_LEN_THR)

    return CriterionDebug(
        passed=bool(passed),
        metrics={
            "t_back": float(t_back),
            "backness_max": float(backness_valid[idx_rel]),
            "shoulder_len_norm": shoulder_len_norm,
            "shoulder_len_raw": float(shoulder_len),
            "scale_at_t_back": float(sc),
            "target_sign": float(target_sign),
            "nonthrow_side": float(0 if nonthrow_side == "left" else 1),  # keep traceability
        },
        thresholds={
            "SHOULDER_LEN_THR": SHOULDER_LEN_THR,
        },
        notes=[] if passed else ["shoulder_length_at_max_pullback_too_small"],
        valid_frames=int(m.sum()),
    )

def crit3_step_forward_with_nonthrow_foot_details(
    df: pd.DataFrame,
    a: int,
    b: int,
    *,
    target_sign: int,
    nonthrow_side: str,
) -> CriterionDebug:
    if nonthrow_side == "left":
        foot_x, _ = load_xy(df, "left_ankle")
    else:
        foot_x, _ = load_xy(df, "right_ankle")

    scale = torso_scale(df)

    a = max(0, int(a))
    b = min(len(foot_x) - 1, int(b))
    if b <= a:
        return _fail(["invalid_window"], valid_frames=0)

    seg_x = foot_x[a:b+1]
    seg_sc = scale[a:b+1]

    m = np.isfinite(seg_x) & np.isfinite(seg_sc)
    valid_x = seg_x[m]

    if len(valid_x) < 3:
        return _fail(["too_few_valid_frames"], valid_frames=int(m.sum()))

    # First 20% of the window as the reference
    k = max(2, int(len(valid_x) * 0.2))
    x0 = float(np.mean(valid_x[:k]))

    # Use the same peak-based displacement logic, but only in x
    if target_sign == 1:
        peak_x = float(np.max(valid_x))
        disp_x = peak_x - x0
    else:
        peak_x = float(np.min(valid_x))
        disp_x = x0 - peak_x

    sc = float(np.nanmedian(seg_sc[m]))
    if not np.isfinite(sc) or sc <= 1e-6:
        sc = 1.0

    disp_x_norm = float(disp_x / sc)

    # Two-level feedback thresholds (both normalized)
    STEP_X_TINY_THR = 0.1
    STEP_X_THR = 1.0

    passed = (disp_x_norm >= STEP_X_THR)

    notes = []
    if not passed:
        if disp_x_norm < STEP_X_TINY_THR:
            notes.append("take_a_step_forward")
        else:
            notes.append("take_a_bigger_step")

    return CriterionDebug(
        passed=bool(passed),
        metrics={
            "disp_x_norm": disp_x_norm,
            "disp_x_raw": float(disp_x),
            "scale_median": float(sc),
            "target_sign": float(target_sign),
        },
        thresholds={
            "STEP_X_TINY_THR": STEP_X_TINY_THR,
            "STEP_X_THR": STEP_X_THR,
        },
        notes=notes,
        valid_frames=int(m.sum()),
    )


def crit4_hand_across_body_towards_opposite_hip_details(
    df: pd.DataFrame,
    a: int,
    b: int,
    *,
    throwing_side: str,
    nonthrow_side: str,
) -> CriterionDebug:
    # Infer target direction sign from throwing side (keeps call sites unchanged)
    target_sign = -1 if throwing_side == "left" else +1

    # Throwing wrist
    if throwing_side == "left":
        wx, wy = load_xy(df, "left_wrist")
    else:
        wx, wy = load_xy(df, "right_wrist")

    # Opposite (non-throwing) hip
    if nonthrow_side == "left":
        hx, hy = load_xy(df, "left_hip")
    else:
        hx, hy = load_xy(df, "right_hip")

    # Shoulders for lead computation
    lsx, lsy = load_xy(df, "left_shoulder")
    rsx, rsy = load_xy(df, "right_shoulder")

    scale = torso_scale(df)

    a = max(0, int(a))
    b = min(len(wx) - 1, int(b))
    if b <= a:
        return _fail(["invalid_window"], valid_frames=0)

    # Window slices
    seg_wx, seg_wy = wx[a:b+1], wy[a:b+1]
    seg_hx, seg_hy = hx[a:b+1], hy[a:b+1]
    seg_lsx, seg_rsx = lsx[a:b+1], rsx[a:b+1]
    seg_sc = scale[a:b+1]

    n = len(seg_wx)
    if n < 5:
        return _fail(["too_short_window"], valid_frames=0)

    # Focus on the tail portion of the window (late throw phase)
    TAIL_FRAC = 0.35
    tail_start_rel = int(max(0, np.floor(n * (1.0 - TAIL_FRAC))))
    tail_start_idx = a + tail_start_rel

    # --- Distance metrics (Euclidean + x-only as backup) ---
    m_dist = (
        np.isfinite(seg_wx) & np.isfinite(seg_wy) &
        np.isfinite(seg_hx) & np.isfinite(seg_hy) &
        np.isfinite(seg_sc)
    )
    if int(m_dist.sum()) < 5:
        return _fail(["too_few_valid_frames"], valid_frames=int(m_dist.sum()))

    dist_eu = np.full(n, np.nan, dtype=float)
    dist_x = np.full(n, np.nan, dtype=float)
    dn_eu = np.full(n, np.nan, dtype=float)
    dn_x = np.full(n, np.nan, dtype=float)

    sc_safe = np.maximum(seg_sc, 1e-6)

    dist_eu[m_dist] = np.sqrt(
        (seg_wx[m_dist] - seg_hx[m_dist])**2 +
        (seg_wy[m_dist] - seg_hy[m_dist])**2
    )
    dist_x[m_dist] = np.abs(seg_wx[m_dist] - seg_hx[m_dist])

    dn_eu[m_dist] = dist_eu[m_dist] / sc_safe[m_dist]
    dn_x[m_dist] = dist_x[m_dist] / sc_safe[m_dist]

    # Tail-only minima
    dn_eu_tail = dn_eu[tail_start_rel:]
    dn_x_tail = dn_x[tail_start_rel:]

    m_tail_dist = np.isfinite(dn_eu_tail)
    if int(m_tail_dist.sum()) < 3:
        return _fail(["too_few_valid_frames_tail"], valid_frames=int(m_tail_dist.sum()))

    min_dn_tail = float(np.nanmin(dn_eu_tail))
    min_dx_norm_tail = float(np.nanmin(dn_x_tail))

    # --- Shoulder lead metric (max over tail), normalized by torso scale ---
    if throwing_side == "left":
        sh_throw_x = seg_lsx
        sh_non_x = seg_rsx
    else:
        sh_throw_x = seg_rsx
        sh_non_x = seg_lsx

    m_lead = np.isfinite(sh_throw_x) & np.isfinite(sh_non_x) & np.isfinite(seg_sc)
    if int(m_lead.sum()) < 5:
        return _fail(["too_few_valid_frames_lead"], valid_frames=int(m_lead.sum()))

    lead = np.full(n, np.nan, dtype=float)        # pixel lead
    lead_norm = np.full(n, np.nan, dtype=float)   # normalized lead

    lead[m_lead] = target_sign * (sh_throw_x[m_lead] - sh_non_x[m_lead])
    lead_norm[m_lead] = lead[m_lead] / sc_safe[m_lead]

    lead_tail = lead[tail_start_rel:]
    lead_norm_tail = lead_norm[tail_start_rel:]

    if int(np.isfinite(lead_norm_tail).sum()) < 3:
        return _fail(
            ["too_few_valid_frames_lead_tail"],
            valid_frames=int(np.isfinite(lead_norm_tail).sum())
        )

    lead_max_tail = float(np.nanmax(lead_tail))
    lead_max_tail_norm = float(np.nanmax(lead_norm_tail))

    # Thresholds
    LEAD_THR = 0.0
    NEAR_HIP_THR = 0.60

    lead_pass = (lead_max_tail_norm > LEAD_THR)
    dist_pass = (min_dn_tail < NEAR_HIP_THR)

    passed = bool(lead_pass and dist_pass)

    notes = []
    if not lead_pass:
        notes.append("throwing_shoulder_not_leading_target")
    if not dist_pass:
        notes.append("hand_not_close_enough_to_opposite_hip")

    sc_med = float(np.nanmedian(seg_sc[np.isfinite(seg_sc)]))
    if not np.isfinite(sc_med) or sc_med <= 1e-6:
        sc_med = 1.0

    return CriterionDebug(
        passed=passed,
        metrics={
            "lead_max_tail": lead_max_tail,                 # pixels (debug)
            "lead_max_tail_norm": lead_max_tail_norm,       # normalized (used for pass/fail)
            "min_dn_tail": min_dn_tail,                     # Euclidean normalized (primary)
            "min_dx_norm_tail": min_dx_norm_tail,           # x-only normalized (backup)
            "tail_start_idx": float(tail_start_idx),
            "tail_frac": float(TAIL_FRAC),
            "scale_median": float(sc_med),
            "target_sign": float(target_sign),
        },
        thresholds={
            "LEAD_THR": float(LEAD_THR),
            "NEAR_HIP_THR": float(NEAR_HIP_THR),
        },
        notes=notes,
        valid_frames=int(m_dist.sum()),
    )


def crit_throw_higher_details(
    df: pd.DataFrame,
    a: int,
    b: int,
    *,
    target_sign: int,
) -> CriterionDebug:
    """
    Detect whether the ball is released too low.
    We use the early post-release ball trajectory inside a short tail window.

    target_sign:
        +1 -> forward is increasing x
        -1 -> forward is decreasing x
    """
    bx, by = load_xy(df, "ball")

    a = max(0, int(a))
    b = min(len(bx) - 1, int(b))
    if b <= a:
        return _fail(["invalid_window"], valid_frames=0)

    seg_bx = bx[a:b+1]
    seg_by = by[a:b+1]

    m = np.isfinite(seg_bx) & np.isfinite(seg_by)
    valid_idx = np.where(m)[0]

    if len(valid_idx) < 3:
        return _fail(["too_few_valid_ball_frames"], valid_frames=int(m.sum()))

    # Use the earliest valid ball points in the tail window to approximate
    # the initial post-release direction, instead of using the full tail.
    p0_rel = int(valid_idx[0])
    p1_rel = int(valid_idx[min(3, len(valid_idx) - 1)])

    x0 = float(seg_bx[p0_rel])
    y0 = float(seg_by[p0_rel])
    x1 = float(seg_bx[p1_rel])
    y1 = float(seg_by[p1_rel])

    dx_forward = float(target_sign * (x1 - x0))
    dy = float(y1 - y0)  # image y increases downward

    if dx_forward <= 1e-6:
        return CriterionDebug(
            passed=False,
            metrics={
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "dx_forward": dx_forward,
                "dy": dy,
                "launch_angle_deg": float("nan"),
            },
            thresholds={
                "THROW_HIGHER_ANGLE_THR_DEG": 0.0,
            },
            notes=["ball_not_moving_forward_after_release"],
            valid_frames=int(m.sum()),
        )

    # Positive angle means the ball initially travels upward/forward.
    # Negative angle means the ball is going downward too early.
    launch_angle_deg = float(np.degrees(np.arctan2(-dy, dx_forward)))

    THROW_HIGHER_ANGLE_THR_DEG = 0.0
    passed = (launch_angle_deg > THROW_HIGHER_ANGLE_THR_DEG)

    return CriterionDebug(
        passed=bool(passed),
        metrics={
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "dx_forward": dx_forward,
            "dy": dy,
            "launch_angle_deg": launch_angle_deg,
        },
        thresholds={
            "THROW_HIGHER_ANGLE_THR_DEG": float(THROW_HIGHER_ANGLE_THR_DEG),
        },
        notes=[] if passed else ["throw_higher"],
        valid_frames=int(m.sum()),
    )

def crit_throw_harder_details(
    df: pd.DataFrame,
    a: int,
    b: int,
) -> CriterionDebug:
    """
    Estimate post-release ball speed from the earliest valid ball points
    inside a short tail window.

    This function itself only computes the speed-related metrics.
    The final pass/fail for student will be decided by comparing against model.
    """
    bx, by = load_xy(df, "ball")
    scale = torso_scale(df)

    a = max(0, int(a))
    b = min(len(bx) - 1, int(b))
    if b <= a:
        return _fail(["invalid_window"], valid_frames=0)

    seg_bx = bx[a:b+1]
    seg_by = by[a:b+1]
    seg_sc = scale[a:b+1]

    m = np.isfinite(seg_bx) & np.isfinite(seg_by) & np.isfinite(seg_sc)
    valid_idx = np.where(m)[0]

    if len(valid_idx) < 3:
        return _fail(["too_few_valid_ball_frames"], valid_frames=int(m.sum()))

    # Use early valid points to approximate initial post-release speed
    p0_rel = int(valid_idx[0])
    p1_rel = int(valid_idx[min(3, len(valid_idx) - 1)])

    x0 = float(seg_bx[p0_rel])
    y0 = float(seg_by[p0_rel])
    x1 = float(seg_bx[p1_rel])
    y1 = float(seg_by[p1_rel])

    dx = float(x1 - x0)
    dy = float(y1 - y0)
    dist_raw = float(np.sqrt(dx**2 + dy**2))

    sc0 = float(seg_sc[p0_rel])
    sc1 = float(seg_sc[p1_rel])
    sc = float(np.nanmedian([sc0, sc1]))
    if not np.isfinite(sc) or sc <= 1e-6:
        sc = float(np.nanmedian(seg_sc[np.isfinite(seg_sc)]))
        if not np.isfinite(sc) or sc <= 1e-6:
            sc = 1.0

    release_speed_norm = float(dist_raw / sc)

    return CriterionDebug(
        passed=True,  # temporary; final student pass/fail decided after comparing with model
        metrics={
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "dx": dx,
            "dy": dy,
            "release_speed_raw": dist_raw,
            "release_speed_norm": release_speed_norm,
            "scale_median": float(sc),
        },
        thresholds={},
        notes=[],
        valid_frames=int(m.sum()),
    )



# =============================================================================
# 主入口：给 main.py 调用
# =============================================================================
def infer_sides_from_facing(facing: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    throwing_side = None
    target_sign = None
    if facing is not None:
        f = facing.lower()
        if "left" in f:
            throwing_side = "left"
            target_sign = -1
        elif "right" in f:
            throwing_side = "right"
            target_sign = +1

    nonthrow_side = None
    if throwing_side == "left":
        nonthrow_side = "right"
    elif throwing_side == "right":
        nonthrow_side = "left"

    return throwing_side, nonthrow_side, target_sign


def evaluate_criteria(
    *,
    student_csv: str,
    model_csv: str,
    map_csv: str,
    model_windows: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict:
    if model_windows is None:
        model_windows = DEFAULT_MODEL_WINDOWS

    stu_df = pd.read_csv(student_csv)
    mdl_df = pd.read_csv(model_csv)
    total_stu_frames = len(stu_df)

    stu_facing = read_majority_facing(student_csv, col="body_facing")
    mdl_facing = read_majority_facing(model_csv, col="body_facing")

    stu_throwing_side, stu_nonthrow_side, stu_target_sign = infer_sides_from_facing(stu_facing)
    mdl_throwing_side, mdl_nonthrow_side, mdl_target_sign = infer_sides_from_facing(mdl_facing)

    stu_to_ref = load_map_stu_to_ref(map_csv)
    ref_to_stu = invert_map_ref_to_stu(stu_to_ref)

    def pack(cd: CriterionDebug) -> Dict:
        return {
            "passed": bool(cd.passed),
            "metrics": cd.metrics,
            "thresholds": cd.thresholds,
            "notes": cd.notes,
            "valid_frames": int(cd.valid_frames),
        }

    items = []

    # ---- c1
    a, b = model_windows["c1"]
    sr = model_range_to_student_range(ref_to_stu, a, b)

    stu_dbg = _fail(["no_student_range"], valid_frames=0)
    if sr is not None and stu_throwing_side is not None:
        stu_dbg = crit1_windup_downward_hand_arm_details(stu_df, sr[0], sr[1], stu_throwing_side)

    mdl_dbg = _fail(["missing_model_throwing_side"], valid_frames=0)
    if mdl_throwing_side is not None:
        mdl_dbg = crit1_windup_downward_hand_arm_details(mdl_df, a, b, mdl_throwing_side)

    items.append({
        "key": "c1",
        "name": "Windup initiated with downward movement of hand and arm",
        "model_range": [int(a), int(b)],
        "student_range": [int(sr[0]), int(sr[1])] if sr else None,
        "student": pack(stu_dbg),
        "model": pack(mdl_dbg),
    })

    # ---- c2
    a, b = model_windows["c2"]
    sr = model_range_to_student_range(ref_to_stu, a, b)

    stu_dbg = _fail(["no_student_range_or_meta"], valid_frames=0)
    if sr is not None and stu_target_sign is not None and stu_nonthrow_side is not None:
        stu_dbg = crit2_rotate_nonthrow_side_faces_target_details(
            stu_df, sr[0], sr[1], target_sign=stu_target_sign, nonthrow_side=stu_nonthrow_side
        )

    mdl_dbg = _fail(["missing_model_meta"], valid_frames=0)
    if mdl_target_sign is not None and mdl_nonthrow_side is not None:
        mdl_dbg = crit2_rotate_nonthrow_side_faces_target_details(
            mdl_df, a, b, target_sign=mdl_target_sign, nonthrow_side=mdl_nonthrow_side
        )

    items.append({
        "key": "c2",
        "name": "Rotate hip & shoulder so non-throwing side faces target",
        "model_range": [int(a), int(b)],
        "student_range": [int(sr[0]), int(sr[1])] if sr else None,
        "student": pack(stu_dbg),
        "model": pack(mdl_dbg),
    })

    # ---- c3
    a, b = model_windows["c3"]
    sr = model_range_to_student_range(ref_to_stu, a, b)

    stu_dbg = _fail(["no_student_range_or_meta"], valid_frames=0)
    if sr is not None and stu_target_sign is not None and stu_nonthrow_side is not None:
        stu_dbg = crit3_step_forward_with_nonthrow_foot_details(
            stu_df, sr[0], sr[1], target_sign=stu_target_sign, nonthrow_side=stu_nonthrow_side
        )

    mdl_dbg = _fail(["missing_model_meta"], valid_frames=0)
    if mdl_target_sign is not None and mdl_nonthrow_side is not None:
        mdl_dbg = crit3_step_forward_with_nonthrow_foot_details(
            mdl_df, a, b, target_sign=mdl_target_sign, nonthrow_side=mdl_nonthrow_side
        )

    items.append({
        "key": "c3",
        "name": "Step forward with non-throwing foot",
        "model_range": [int(a), int(b)],
        "student_range": [int(sr[0]), int(sr[1])] if sr else None,
        "student": pack(stu_dbg),
        "model": pack(mdl_dbg),
    })

    # ---- c4
    a, b = model_windows["c4"]
    sr = model_range_to_student_range(ref_to_stu, a, b)

    # 自动托底逻辑
    min_c4_frames = 10
    if sr is None:
        sr = (max(0, total_stu_frames - 15), total_stu_frames - 1)
    else:
        if (sr[1] - sr[0]) < min_c4_frames:
            sr = (sr[0], total_stu_frames - 1)
            if (sr[1] - sr[0]) < min_c4_frames:
                sr = (max(0, sr[1] - 15), sr[1])

    stu_dbg = _fail(["no_student_range_or_meta"], valid_frames=0)
    if sr is not None and stu_throwing_side is not None and stu_nonthrow_side is not None:
        stu_dbg = crit4_hand_across_body_towards_opposite_hip_details(
            stu_df, sr[0], sr[1], throwing_side=stu_throwing_side, nonthrow_side=stu_nonthrow_side
        )

    mdl_dbg = _fail(["missing_model_meta"], valid_frames=0)
    if mdl_throwing_side is not None and mdl_nonthrow_side is not None:
        mdl_dbg = crit4_hand_across_body_towards_opposite_hip_details(
            mdl_df, a, b, throwing_side=mdl_throwing_side, nonthrow_side=mdl_nonthrow_side
        )

    items.append({
        "key": "c4",
        "name": "Throwing hand reaches across body towards opposite hip",
        "model_range": [int(a), int(b)],
        "student_range": [int(sr[0]), int(sr[1])] if sr else None,
        "student": pack(stu_dbg),
        "model": pack(mdl_dbg),
    })
    
    
    # ---- extra signal: throw_higher
    # Do NOT put this into items yet, otherwise current frontend will render it.
    THROW_HIGHER_TAIL_FRAMES = 12

    stu_throw_higher_range = None
    mdl_throw_higher_range = None

    stu_throw_higher_dbg = _fail(["missing_student_meta"], valid_frames=0)
    mdl_throw_higher_dbg = _fail(["missing_model_meta"], valid_frames=0)

    if total_stu_frames > 0:
        stu_throw_higher_range = (
            max(0, total_stu_frames - THROW_HIGHER_TAIL_FRAMES),
            total_stu_frames - 1,
        )
    if len(mdl_df) > 0:
        mdl_throw_higher_range = (
            max(0, len(mdl_df) - THROW_HIGHER_TAIL_FRAMES),
            len(mdl_df) - 1,
        )

    if stu_throw_higher_range is not None and stu_target_sign is not None:
        stu_throw_higher_dbg = crit_throw_higher_details(
            stu_df,
            stu_throw_higher_range[0],
            stu_throw_higher_range[1],
            target_sign=stu_target_sign,
        )

    if mdl_throw_higher_range is not None and mdl_target_sign is not None:
        mdl_throw_higher_dbg = crit_throw_higher_details(
            mdl_df,
            mdl_throw_higher_range[0],
            mdl_throw_higher_range[1],
            target_sign=mdl_target_sign,
        )


    # ---- extra signal: throw_harder
    # Do NOT put this into items yet, otherwise current frontend will render it.
    THROW_HARDER_TAIL_FRAMES = 12
    HARDER_RATIO_THR = 0.5

    stu_throw_harder_range = None
    mdl_throw_harder_range = None

    stu_throw_harder_dbg = _fail(["missing_student_meta"], valid_frames=0)
    mdl_throw_harder_dbg = _fail(["missing_model_meta"], valid_frames=0)

    if total_stu_frames > 0:
        stu_throw_harder_range = (
            max(0, total_stu_frames - THROW_HARDER_TAIL_FRAMES),
            total_stu_frames - 1,
        )
    if len(mdl_df) > 0:
        mdl_throw_harder_range = (
            max(0, len(mdl_df) - THROW_HARDER_TAIL_FRAMES),
            len(mdl_df) - 1,
        )

    if stu_throw_harder_range is not None:
        stu_throw_harder_dbg = crit_throw_harder_details(
            stu_df,
            stu_throw_harder_range[0],
            stu_throw_harder_range[1],
        )

    if mdl_throw_harder_range is not None:
        mdl_throw_harder_dbg = crit_throw_harder_details(
            mdl_df,
            mdl_throw_harder_range[0],
            mdl_throw_harder_range[1],
        )

    # Final pass/fail is based on student speed relative to model speed
    stu_speed_norm = float(stu_throw_harder_dbg.metrics.get("release_speed_norm", np.nan)) \
        if isinstance(stu_throw_harder_dbg.metrics, dict) else np.nan
    mdl_speed_norm = float(mdl_throw_harder_dbg.metrics.get("release_speed_norm", np.nan)) \
        if isinstance(mdl_throw_harder_dbg.metrics, dict) else np.nan

    if np.isfinite(stu_speed_norm) and np.isfinite(mdl_speed_norm) and mdl_speed_norm > 1e-6:
        speed_ratio = float(stu_speed_norm / mdl_speed_norm)
        passed = bool(speed_ratio >= HARDER_RATIO_THR)

        stu_throw_harder_dbg.passed = passed
        stu_throw_harder_dbg.thresholds = {
            "HARDER_RATIO_THR": float(HARDER_RATIO_THR),
            "MODEL_RELEASE_SPEED_NORM": float(mdl_speed_norm),
        }
        stu_throw_harder_dbg.metrics["speed_ratio_vs_model"] = speed_ratio
        stu_throw_harder_dbg.notes = [] if passed else ["throw_harder"]

        # model side is just reference, always mark pass if valid
        mdl_throw_harder_dbg.passed = True
        mdl_throw_harder_dbg.thresholds = {
            "HARDER_RATIO_THR": float(HARDER_RATIO_THR),
        }
        mdl_throw_harder_dbg.notes = []
    else:
        if isinstance(stu_throw_harder_dbg.metrics, dict):
            stu_throw_harder_dbg.metrics["speed_ratio_vs_model"] = float("nan")
        stu_throw_harder_dbg.passed = False
        stu_throw_harder_dbg.thresholds = {
            "HARDER_RATIO_THR": float(HARDER_RATIO_THR),
            "MODEL_RELEASE_SPEED_NORM": float(mdl_speed_norm) if np.isfinite(mdl_speed_norm) else float("nan"),
        }
        stu_throw_harder_dbg.notes = ["throw_harder_reference_unavailable"]

        if isinstance(mdl_throw_harder_dbg.thresholds, dict):
            mdl_throw_harder_dbg.thresholds = {
                "HARDER_RATIO_THR": float(HARDER_RATIO_THR),
            }
            
            
            
    return {
        "meta": {
            "student": {
                "majority_facing": stu_facing,
                "throwing_side": stu_throwing_side,
                "nonthrow_side": stu_nonthrow_side,
                "target_sign": stu_target_sign,
                "num_frames": int(total_stu_frames),
            },
            "model": {
                "majority_facing": mdl_facing,
                "throwing_side": mdl_throwing_side,
                "nonthrow_side": mdl_nonthrow_side,
                "target_sign": mdl_target_sign,
                "num_frames": int(len(mdl_df)),
            },
        },
        "items": items,
        "extra_feedback_signals": {
            "throw_higher": {
                "name": "Throw higher",
                "student_range": [int(stu_throw_higher_range[0]), int(stu_throw_higher_range[1])] if stu_throw_higher_range else None,
                "model_range": [int(mdl_throw_higher_range[0]), int(mdl_throw_higher_range[1])] if mdl_throw_higher_range else None,
                "student": pack(stu_throw_higher_dbg),
                "model": pack(mdl_throw_higher_dbg),
            },
            "throw_harder": {
                "name": "Throw harder",
                "student_range": [int(stu_throw_harder_range[0]), int(stu_throw_harder_range[1])] if stu_throw_harder_range else None,
                "model_range": [int(mdl_throw_harder_range[0]), int(mdl_throw_harder_range[1])] if mdl_throw_harder_range else None,
                "student": pack(stu_throw_harder_dbg),
                "model": pack(mdl_throw_harder_dbg),
            }
        },
    }