#!/usr/bin/env python3
"""
Fixed Motion Comparison - Proper Normalization + Teacher Feature Set

Teacher-required pipeline (legacy mode):
Step 1) For each variable, compute DTW alignment and output ONE scalar per variable:
        -> average of aligned (DTW-warped) per-frame distances.
Step 2) Overall matching score = average of all variable scalars.

NEW (pose-align mode):
- Use ONE multi-dimensional DTW alignment path computed from normalized pose vectors
  (relative keypoint coordinates).
- Then re-compute ALL per-feature frame distances along this ONE alignment path.
- Keeps legacy mode unchanged by default.

IMPORTANT:
- Default behavior is unchanged: compare(..., use_pose_align=False) uses old per-feature DTW.
"""

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from typing import Dict, Tuple, List, Optional
import warnings
import math

warnings.filterwarnings("ignore")


class FixedMotionDTW:
    """
    Properly normalized DTW comparison
    - Legacy mode:
        - Origin: hip center (sequence mean)
        - Scale: body_height (shoulder-hip vertical distance)
        - Facing: mirror x if needed (based on body_facing mode)
        - DTW distance: relative difference (percentage-like, 0..1 capped)
    - Pose-align mode (NEW):
        - Build per-frame normalized pose vectors (translation/scale/rotation)
        - Compute ONE DTW path using multi-dim pose vectors
        - Recompute teacher features differences along that path
    """

    # Teacher-required feature names (generic / side-agnostic)
    FEATURE_NAMES = [
        "elbow_angle",                     # dominant side
        "shoulder_angle",                  # dominant side
        "hip_angle",                       # dominant side
        "feet_lr_dist",                    # left-right feet distance (normalized)
        "wrist_to_samehip_dist",           # dominant wrist to dominant hip
        "wrist_to_opphip_dist",            # dominant wrist to opposite hip
        "elbow_to_samehip_dist",           # dominant elbow to dominant hip
        "elbow_to_opphip_dist",            # dominant elbow to opposite hip
        "shoulder_to_wrist_y_dist",        # |y_shoulder - y_wrist|
        "shoulder_to_elbow_y_dist",        # |y_shoulder - y_elbow|
    ]

    # Angle features are compared in absolute degrees (raw).
    ANGLE_FEATURES = {"elbow_angle", "shoulder_angle", "hip_angle"}
    ANGLE_MAX_DEG = 180.0

    # Pose-align: which joints to use (12 points => 24 dims x,y)
    # Order matters: keep consistent for student and reference.
    POSE_JOINTS = [
        ("left",  "shoulder"), ("right", "shoulder"),
        ("left",  "hip"),      ("right", "hip"),
        ("left",  "elbow"),    ("right", "elbow"),
        ("left",  "wrist"),    ("right", "wrist"),
        ("left",  "knee"),     ("right", "knee"),
        ("left",  "ankle"),    ("right", "ankle"),
    ]
    
    OPPOSITE_ANKLE_DTW_FACTOR = math.sqrt(2.0)
    # 说明：
    # 因为 _pose_dist_l2 用的是 Euclidean norm:
    #   ||a-b|| = sqrt(sum_i d_i^2)
    # 如果你想让某一组维度的“权重翻倍”，数学上应该乘 sqrt(2)，
    # 这样该组维度对距离的平方贡献会变成原来的 2 倍。
    # 如果你直接乘 2.0，那么贡献会变成 4 倍，不是严格意义上的“翻倍”。

    def _pose_joint_flat_slice(self, side: str, joint: str) -> slice:
        idx = self.POSE_JOINTS.index((side, joint))
        return slice(2 * idx, 2 * idx + 2)

    def _apply_pose_joint_weight(
        self,
        pose_mat: np.ndarray,
        *,
        side: str,
        joint: str,
        factor: float,
    ) -> np.ndarray:
        if pose_mat is None or pose_mat.size == 0:
            return pose_mat
        out = np.array(pose_mat, dtype=float, copy=True)
        sl = self._pose_joint_flat_slice(side, joint)
        out[:, sl] *= float(factor)
        return out

    def _angle_distance_norm(self, x, y) -> float:
        """
        DTW distance for angles (legacy per-feature DTW):
        normalized absolute degree difference in [0, 1].
        """
        ref_val = float(x[0])
        stu_val = float(y[0])
        return min(abs(ref_val - stu_val) / self.ANGLE_MAX_DEG, 1.0)

    def __init__(self, reference_csv: str, reference_handedness: str = "auto"):
        self.reference_df = pd.read_csv(reference_csv)
        self.reference_facing = self._safe_mode(self.reference_df, "body_facing", default="unknown")

        # Decide reference dominant side
        if reference_handedness in ("left", "right"):
            self.reference_handedness = reference_handedness
        else:
            # AUTO: use body_facing if it is already 'left'/'right'
            if self.reference_facing in ("left", "right"):
                self.reference_handedness = self.reference_facing
            else:
                # fallback: heuristic
                df_norm = self._normalize_coordinates(self.reference_df, self.reference_facing)
                self.reference_handedness = self._infer_handedness(df_norm)

        self.reference_features = self._extract_features(
            self.reference_df,
            facing=self.reference_facing,
            dominant_side=self.reference_handedness,
        )

    # -----------------------------
    # Utilities
    # -----------------------------
    @staticmethod
    def _safe_mode(df: pd.DataFrame, col: str, default=None):
        if col in df.columns and df[col].notna().any():
            return df[col].mode().iloc[0]
        return default

    @staticmethod
    def _has_series(df: pd.DataFrame, col: str) -> bool:
        return col in df.columns and df[col].notna().any()

    def _pick_series(self, df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
        """Return first available series among candidates; else None."""
        for c in candidates:
            if self._has_series(df, c):
                return df[c]
        return None

    def _pick_value(self, row: pd.Series, candidates: List[str]) -> float:
        """Return first available scalar in row among candidates; else NaN."""
        for c in candidates:
            if c in row.index and pd.notna(row[c]):
                return float(row[c])
        return float("nan")

    def _relative_distance(self, x, y):
        """
        Relative diff metric (0..1):
            |ref - stu| / |ref|   (capped at 1.0)
        with protection when ref ~= 0.
        """
        ref_val = x[0]
        stu_val = y[0]

        thr = 1e-3
        if abs(ref_val) < thr:
            return min(abs(ref_val - stu_val), 1.0)

        return min(abs(ref_val - stu_val) / abs(ref_val), 1.0)

    def _aligned_relative_scalar(self, ref_val: float, stu_val: float) -> float:
        """
        Same as _relative_distance but scalar inputs.
        Returns in [0,1].
        """
        thr = 1e-3
        if np.isnan(ref_val) or np.isnan(stu_val):
            return np.nan
        if abs(ref_val) < thr:
            return min(abs(ref_val - stu_val), 1.0)
        return min(abs(ref_val - stu_val) / abs(ref_val), 1.0)

    def _interpolate_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Linear interpolate NaN; ffill/bfill at ends."""
        if seq.size == 0:
            return seq
        if np.all(np.isnan(seq)):
            return seq
        s = pd.Series(seq)
        s = s.interpolate(method="linear", limit_direction="both")
        s = s.ffill().bfill()
        return s.values

    def _interpolate_matrix(self, mat: np.ndarray) -> np.ndarray:
        """
        Interpolate NaNs in a 2D matrix column-wise.
        mat: (T, D)
        """
        if mat.size == 0:
            return mat
        out = mat.copy()
        for j in range(out.shape[1]):
            out[:, j] = self._interpolate_sequence(out[:, j])
        return out

    # -----------------------------
    # Normalization (legacy)
    # -----------------------------
    def _normalize_coordinates(self, df: pd.DataFrame, facing: str) -> pd.DataFrame:
        """
        Legacy normalization:
        1) body_height = |mean(shoulder_y) - mean(hip_y)|
        2) origin = hip center mean (sequence mean)
        3) normalize all *_x/*_y by subtracting origin and dividing by body_height
        4) mirror x if facing != reference_facing
        """
        df = df.copy()

        shoulder_y = self._pick_series(df, ["right_shoulder_y", "shoulder_y", "left_shoulder_y"])
        hip_y = self._pick_series(df, ["hip_y", "right_hip_y", "left_hip_y"])

        if shoulder_y is None or hip_y is None:
            y_cols = [c for c in df.columns if c.endswith("_y")]
            if len(y_cols) == 0:
                body_height = 100.0
            else:
                body_height = float(
                    abs(df[y_cols].mean(numeric_only=True).max() - df[y_cols].mean(numeric_only=True).min())
                )
        else:
            body_height = float(abs(hip_y.mean() - shoulder_y.mean()))

        if body_height < 10:
            body_height = 100.0

        # hip center
        if self._has_series(df, "right_hip_x") and self._has_series(df, "right_hip_y"):
            if self._has_series(df, "left_hip_x") and self._has_series(df, "left_hip_y"):
                hip_center_x = (df["right_hip_x"] + df["left_hip_x"]) / 2
                hip_center_y = (df["right_hip_y"] + df["left_hip_y"]) / 2
            else:
                hip_center_x = df["right_hip_x"]
                hip_center_y = df["right_hip_y"]
        else:
            x_cols = [c for c in df.columns if c.endswith("_x")]
            y_cols = [c for c in df.columns if c.endswith("_y")]
            hip_center_x = df[x_cols].mean(axis=1) if x_cols else pd.Series(np.zeros(len(df)))
            hip_center_y = df[y_cols].mean(axis=1) if y_cols else pd.Series(np.zeros(len(df)))

        origin_x = float(hip_center_x.mean())
        origin_y = float(hip_center_y.mean())

        coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
        for c in coord_cols:
            if c.endswith("_x"):
                df[c] = (df[c] - origin_x) / body_height
            else:
                df[c] = (df[c] - origin_y) / body_height

        # facing mirror
        if facing != self.reference_facing:
            x_cols = [c for c in df.columns if c.endswith("_x")]
            for c in x_cols:
                df[c] = -df[c]

        return df

    # -----------------------------
    # Geometry helpers
    # -----------------------------
    def _calculate_angle(self, p1, p2, p3) -> float:
        """Angle at p2 formed by p1-p2-p3, in degrees."""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
        cosang = float(np.dot(v1, v2) / denom)
        cosang = float(np.clip(cosang, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    # -----------------------------
    # Column candidates (robust to your dataset naming)
    # -----------------------------
    def _kp_candidates(self, side: str, joint: str) -> Tuple[List[str], List[str]]:
        """
        Returns candidate column names for x and y for a given joint and side.
        side: 'left' or 'right'
        joint: 'shoulder'/'elbow'/'wrist'/'hip'/'knee'/'ankle'
        """
        side = side.lower()
        assert side in ("left", "right")

        if joint == "wrist":
            x_cands = [f"{side}_wrist_x", f"{side}_hand_x", "hand_x"]
            y_cands = [f"{side}_wrist_y", f"{side}_hand_y", "hand_y"]
        elif joint == "elbow":
            x_cands = [f"{side}_elbow_x", "elbow_x"]
            y_cands = [f"{side}_elbow_y", "elbow_y"]
        elif joint == "shoulder":
            x_cands = [f"{side}_shoulder_x", "shoulder_x"]
            y_cands = [f"{side}_shoulder_y", "shoulder_y"]
        elif joint == "hip":
            x_cands = [f"{side}_hip_x", "hip_x"]
            y_cands = [f"{side}_hip_y", "hip_y"]
        elif joint == "knee":
            x_cands = [f"{side}_knee_x", "knee_x"]
            y_cands = [f"{side}_knee_y", "knee_y"]
        elif joint == "ankle":
            x_cands = [f"{side}_ankle_x", f"{side}_foot_x", f"{side}_feet_x"]
            y_cands = [f"{side}_ankle_y", f"{side}_foot_y", f"{side}_feet_y"]
        else:
            x_cands, y_cands = [], []

        return x_cands, y_cands

    def _get_point(self, row: pd.Series, side: str, joint: str) -> Tuple[float, float]:
        x_cands, y_cands = self._kp_candidates(side, joint)
        x = self._pick_value(row, x_cands)
        y = self._pick_value(row, y_cands)
        return x, y

    # -----------------------------
    # Pose-align helpers (NEW)
    # -----------------------------
    def _get_pose_points_row(self, row: pd.Series) -> np.ndarray:
        """
        Build 12x2 points for one frame using raw coordinates.
        Returns array shape (12,2) with NaNs if missing.
        """
        pts = []
        for side, joint in self.POSE_JOINTS:
            x, y = self._get_point(row, side, joint)
            pts.append([x, y])
        return np.array(pts, dtype=float)

    def _normalize_pose_frame(self, pts: np.ndarray, flip_x: bool = False) -> np.ndarray:
        """
        Normalize a single frame pose:
        - translation: origin at mid-hip
        - scale: shoulder width
        - optional mirror: flip x in local coords
        - rotation: align shoulder vector to x-axis

        pts: (12,2) raw coords
        returns: (12,2) normalized coords (NaN-safe)
        """
        pts_n = pts.copy()

        # indices for shoulders and hips based on POSE_JOINTS order
        # [L_sh, R_sh, L_hip, R_hip, ...]
        LSH, RSH = 0, 1
        LHIP, RHIP = 2, 3

        lhip = pts_n[LHIP]
        rhip = pts_n[RHIP]
        lsh = pts_n[LSH]
        rsh = pts_n[RSH]

        if np.any(np.isnan(lhip)) or np.any(np.isnan(rhip)) or np.any(np.isnan(lsh)) or np.any(np.isnan(rsh)):
            return np.full_like(pts_n, np.nan)

        mid_hip = (lhip + rhip) / 2.0
        pts_n = pts_n - mid_hip  # translation

        if flip_x:
            pts_n[:, 0] = -pts_n[:, 0]  # mirror in local coords

        shoulder_vec = (rsh - lsh)
        sw = float(np.linalg.norm(shoulder_vec))
        if sw < 1e-6 or np.isnan(sw):
            sw = 1.0

        pts_n = pts_n / sw  # scale

        # rotation: align shoulder vector to x-axis
        # after translation/flip/scale, recompute shoulder vector in normalized space
        lsh2 = pts_n[LSH]
        rsh2 = pts_n[RSH]
        v = rsh2 - lsh2
        if np.any(np.isnan(v)) or float(np.linalg.norm(v)) < 1e-6:
            return pts_n

        theta = float(np.arctan2(v[1], v[0]))
        c, s = float(np.cos(-theta)), float(np.sin(-theta))
        R = np.array([[c, -s], [s, c]], dtype=float)

        pts_rot = (R @ pts_n.T).T
        return pts_rot

    def _build_pose_matrix(self, df: pd.DataFrame, flip_x: bool = False) -> np.ndarray:
        """
        Build normalized pose matrix for a whole clip:
        returns shape (T, D) where D=24 (12 points x 2).
        """
        mats = []
        for _, row in df.iterrows():
            pts = self._get_pose_points_row(row)
            pts_n = self._normalize_pose_frame(pts, flip_x=flip_x)
            mats.append(pts_n.reshape(-1))  # flatten to (24,)
        mat = np.vstack(mats) if mats else np.zeros((0, 24), dtype=float)
        mat = self._interpolate_matrix(mat)
        return mat

    def _pose_dist_l2(self, a: np.ndarray, b: np.ndarray) -> float:
        """Distance between two pose vectors (D,)."""
        if a is None or b is None:
            return float("nan")
        # NaN guard: if any NaN remains, treat as large distance
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return 1e6
        return float(np.linalg.norm(a - b))

    def _compute_pose_dtw_path(self, ref_pose: np.ndarray, stu_pose: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Compute DTW path using multi-dimensional pose vectors.
        Returns (dtw_norm, path) where path is list of (ref_idx, stu_idx).
        """
        if ref_pose.shape[0] < 3 or stu_pose.shape[0] < 3:
            return np.nan, []

        dist, path = fastdtw(ref_pose, stu_pose, dist=self._pose_dist_l2)
        dtw_norm = float(dist) / max(len(path), 1)
        return dtw_norm, path

    def _path_to_map_s_to_m(self, path: List[Tuple[int, int]], n_student: int) -> np.ndarray:
        """
        Convert DTW path (ref_idx, stu_idx) to a stable mapping map_s_to_m[s]=m
        using median of matched reference indices.
        """
        if n_student <= 0:
            return np.zeros((0,), dtype=int)

        stu_to_ref: Dict[int, List[int]] = {}
        for ref_idx, stu_idx in path:
            if 0 <= stu_idx < n_student:
                stu_to_ref.setdefault(stu_idx, []).append(ref_idx)

        map_s_to_m = np.full(n_student, -1, dtype=int)
        last_valid = 0
        for s in range(n_student):
            refs = stu_to_ref.get(s, [])
            if refs:
                m = int(np.median(refs))
                map_s_to_m[s] = m
                last_valid = m
            else:
                map_s_to_m[s] = last_valid  # carry forward

        return map_s_to_m

    # -----------------------------
    # Handedness
    # -----------------------------
    def _infer_handedness(self, df_norm: pd.DataFrame) -> str:
        """
        Heuristic:
        choose side whose wrist has larger motion magnitude (std of distance to hip).
        If left/right wrist columns are unavailable, fall back to 'right'.
        """
        has_lw = (
            self._has_series(df_norm, "left_wrist_x")
            or self._has_series(df_norm, "left_hand_x")
            or self._has_series(df_norm, "left_wrist_y")
            or self._has_series(df_norm, "left_hand_y")
        )
        has_rw = (
            self._has_series(df_norm, "right_wrist_x")
            or self._has_series(df_norm, "right_hand_x")
            or self._has_series(df_norm, "right_wrist_y")
            or self._has_series(df_norm, "right_hand_y")
        )

        has_lhip = self._has_series(df_norm, "left_hip_x") and self._has_series(df_norm, "left_hip_y")
        has_rhip = self._has_series(df_norm, "right_hip_x") and self._has_series(df_norm, "right_hip_y")

        if not (has_lw and has_rw and has_lhip and has_rhip):
            return "right"

        lwx = self._pick_series(df_norm, ["left_wrist_x", "left_hand_x"])
        lwy = self._pick_series(df_norm, ["left_wrist_y", "left_hand_y"])
        rwx = self._pick_series(df_norm, ["right_wrist_x", "right_hand_x"])
        rwy = self._pick_series(df_norm, ["right_wrist_y", "right_hand_y"])

        lhipx, lhipy = df_norm["left_hip_x"], df_norm["left_hip_y"]
        rhipx, rhipy = df_norm["right_hip_x"], df_norm["right_hip_y"]

        ldist = np.sqrt((lwx - lhipx) ** 2 + (lwy - lhipy) ** 2)
        rdist = np.sqrt((rwx - rhipx) ** 2 + (rwy - rhipy) ** 2)

        lstd = float(np.nanstd(ldist.values))
        rstd = float(np.nanstd(rdist.values))

        return "left" if lstd > rstd else "right"

    # -----------------------------
    # Feature extraction (teacher set) - legacy normalized coords
    # -----------------------------
    def _extract_features(self, df: pd.DataFrame, facing: str, dominant_side: str) -> Dict[str, np.ndarray]:
        """
        Extract teacher-required sequences (legacy normalized coordinates).

        dominant_side: 'left' or 'right'
        """
        df_norm = self._normalize_coordinates(df, facing)
        n = len(df_norm)

        dom = dominant_side
        opp = "left" if dom == "right" else "right"

        feats: Dict[str, np.ndarray] = {}

        elbow_angles = []
        shoulder_angles = []
        hip_angles = []
        shoulder_wrist_y = []
        shoulder_elbow_y = []

        for _, row in df_norm.iterrows():
            sh = self._get_point(row, dom, "shoulder")
            el = self._get_point(row, dom, "elbow")
            wr = self._get_point(row, dom, "wrist")
            hp = self._get_point(row, dom, "hip")
            kn = self._get_point(row, dom, "knee")

            if not (np.isnan(sh[0]) or np.isnan(el[0]) or np.isnan(wr[0])):
                elbow_angles.append(self._calculate_angle(sh, el, wr))
            else:
                elbow_angles.append(np.nan)

            if not (np.isnan(el[0]) or np.isnan(sh[0]) or np.isnan(hp[0])):
                shoulder_angles.append(self._calculate_angle(el, sh, hp))
            else:
                shoulder_angles.append(np.nan)

            if not (np.isnan(sh[0]) or np.isnan(hp[0]) or np.isnan(kn[0])):
                hip_angles.append(self._calculate_angle(sh, hp, kn))
            else:
                hip_angles.append(np.nan)

            if not (np.isnan(sh[1]) or np.isnan(wr[1])):
                shoulder_wrist_y.append(abs(sh[1] - wr[1]))
            else:
                shoulder_wrist_y.append(np.nan)

            if not (np.isnan(sh[1]) or np.isnan(el[1])):
                shoulder_elbow_y.append(abs(sh[1] - el[1]))
            else:
                shoulder_elbow_y.append(np.nan)

        feats["elbow_angle"] = np.array(elbow_angles, dtype=float)
        feats["shoulder_angle"] = np.array(shoulder_angles, dtype=float)
        feats["hip_angle"] = np.array(hip_angles, dtype=float)

        # feet distance
        feet_dist = []
        for _, row in df_norm.iterrows():
            lf = self._get_point(row, "left", "ankle")
            rf = self._get_point(row, "right", "ankle")
            if not (np.isnan(lf[0]) or np.isnan(rf[0])):
                feet_dist.append(np.sqrt((lf[0] - rf[0]) ** 2 + (lf[1] - rf[1]) ** 2))
            else:
                feet_dist.append(np.nan)
        feats["feet_lr_dist"] = np.array(feet_dist, dtype=float)

        def dist_points(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            if np.isnan(a[0]) or np.isnan(b[0]):
                return np.nan
            return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

        wrist_samehip = []
        wrist_opphip = []
        elbow_samehip = []
        elbow_opphip = []

        for _, row in df_norm.iterrows():
            wr_dom = self._get_point(row, dom, "wrist")
            el_dom = self._get_point(row, dom, "elbow")
            hip_dom = self._get_point(row, dom, "hip")
            hip_opp = self._get_point(row, opp, "hip")

            wrist_samehip.append(dist_points(wr_dom, hip_dom))
            wrist_opphip.append(dist_points(wr_dom, hip_opp))
            elbow_samehip.append(dist_points(el_dom, hip_dom))
            elbow_opphip.append(dist_points(el_dom, hip_opp))

        feats["wrist_to_samehip_dist"] = np.array(wrist_samehip, dtype=float)
        feats["wrist_to_opphip_dist"] = np.array(wrist_opphip, dtype=float)
        feats["elbow_to_samehip_dist"] = np.array(elbow_samehip, dtype=float)
        feats["elbow_to_opphip_dist"] = np.array(elbow_opphip, dtype=float)

        feats["shoulder_to_wrist_y_dist"] = np.array(shoulder_wrist_y, dtype=float)
        feats["shoulder_to_elbow_y_dist"] = np.array(shoulder_elbow_y, dtype=float)

        for k in self.FEATURE_NAMES:
            if k not in feats:
                feats[k] = np.full(n, np.nan, dtype=float)

        return feats

    # -----------------------------
    # DTW + per-feature legacy alignment
    # -----------------------------
    def _compute_frame_distances_from_path(
        self,
        ref_seq: np.ndarray,
        stu_seq: np.ndarray,
        path: list,
        use_relative: bool = True,
    ) -> np.ndarray:
        """
        For each student frame index, average distances to all matched reference frames.
        Returns per-student-frame aligned distance series.
        """
        frame_distances = np.full(len(stu_seq), np.nan)
        stu_to_ref = {}

        for ref_idx, stu_idx in path:
            stu_to_ref.setdefault(stu_idx, []).append(ref_idx)

        for stu_idx, ref_indices in stu_to_ref.items():
            dists = []
            for ref_idx in ref_indices:
                ref_val = ref_seq[ref_idx]
                stu_val = stu_seq[stu_idx]

                if use_relative:
                    if abs(ref_val) < 1e-3:
                        dist = min(abs(ref_val - stu_val), 1.0)
                    else:
                        dist = min(abs(ref_val - stu_val) / abs(ref_val), 1.0)
                else:
                    dist = abs(ref_val - stu_val)

                dists.append(dist)

            frame_distances[stu_idx] = float(np.mean(dists)) if dists else np.nan

        return frame_distances

    def _compute_dtw(
        self,
        ref_seq: np.ndarray,
        stu_seq: np.ndarray,
        dist_fn=None,
        frame_use_relative: bool = True,
    ) -> Tuple[float, list, np.ndarray]:
        """
        Legacy per-feature DTW:
        Returns:
            dtw_norm: distance/len(path) (kept for debugging)
            path: DTW path
            frame_dists: aligned per-student-frame distances
        """
        if dist_fn is None:
            dist_fn = self._relative_distance

        ref_i = self._interpolate_sequence(ref_seq)
        stu_i = self._interpolate_sequence(stu_seq)

        if np.any(np.isnan(ref_i)) or np.any(np.isnan(stu_i)):
            return np.nan, [], np.full(len(stu_seq), np.nan)

        if len(ref_i) < 3 or len(stu_i) < 3:
            return np.nan, [], np.full(len(stu_seq), np.nan)

        ref_r = ref_i.reshape(-1, 1)
        stu_r = stu_i.reshape(-1, 1)

        dist, path = fastdtw(ref_r, stu_r, dist=dist_fn)
        dtw_norm = float(dist) / max(len(path), 1)

        frame_dists = self._compute_frame_distances_from_path(
            ref_i, stu_i, path, use_relative=frame_use_relative
        )
        return dtw_norm, path, frame_dists

    # -----------------------------
    # Compare
    # -----------------------------
    def compare(
        self,
        student_csv: str,
        handedness: str = "auto",
        save_frame_csv: bool = False,
        use_pose_align: bool = False,
        save_align_path: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, str]]:
        """
        Outputs:
        - per_feature_score: ONE scalar per variable + overall_matching_score
        - frame_distances_dict: per-frame distances for each feature (length = n_student_frames)
        - meta: used handedness & facing + align method

        handedness:
            'right' / 'left' / 'auto'
        use_pose_align:
            False => legacy per-feature DTW (default, unchanged)
            True  => NEW: pose-vector DTW path + aligned per-feature errors along that path
        """
        student_df = pd.read_csv(student_csv)
        student_facing = self._safe_mode(student_df, "body_facing", default="unknown")

        # Decide dominant side for student
        if handedness in ("left", "right"):
            dom = handedness
            handedness_source = "manual"
        else:
            # AUTO: in your dataset, body_facing == dominant side
            if student_facing in ("left", "right"):
                dom = student_facing
                handedness_source = "body_facing"
            else:
                # fallback: heuristic if body_facing is missing/unknown
                df_norm_for_infer = self._normalize_coordinates(student_df, student_facing)
                dom = self._infer_handedness(df_norm_for_infer)
                handedness_source = "heuristic"

        # Compute teacher features (legacy normalized coords)
        student_features = self._extract_features(student_df, facing=student_facing, dominant_side=dom)

        per_feature_score: Dict[str, float] = {}
        frame_distances_dict: Dict[str, np.ndarray] = {}

        # -----------------------------
        # Branch A: Legacy per-feature DTW (UNCHANGED)
        # -----------------------------
        if not use_pose_align:
            for feat in self.FEATURE_NAMES:
                ref_seq = self.reference_features.get(feat, np.full(len(self.reference_df), np.nan))
                stu_seq = student_features.get(feat, np.full(len(student_df), np.nan))

                if feat in self.ANGLE_FEATURES:
                    _, _, frame_dists_deg = self._compute_dtw(
                        ref_seq, stu_seq, dist_fn=self._angle_distance_norm, frame_use_relative=False
                    )
                    frame_distances_dict[feat] = frame_dists_deg
                    per_feature_score[feat] = float(np.nanmean(frame_dists_deg)) if np.any(~np.isnan(frame_dists_deg)) else np.nan
                else:
                    _, _, frame_dists = self._compute_dtw(ref_seq, stu_seq)
                    frame_distances_dict[feat] = frame_dists
                    per_feature_score[feat] = float(np.nanmean(frame_dists)) if np.any(~np.isnan(frame_dists)) else np.nan

            # overall
            vals_norm = []
            for feat in self.FEATURE_NAMES:
                v = per_feature_score.get(feat, np.nan)
                if np.isnan(v):
                    continue
                if feat in self.ANGLE_FEATURES:
                    vals_norm.append(min(v / self.ANGLE_MAX_DEG, 1.0))
                else:
                    vals_norm.append(v)
            per_feature_score["overall_matching_score"] = float(np.mean(vals_norm)) if vals_norm else np.nan

            if save_frame_csv:
                self._save_frame_distances_csv(student_csv, frame_distances_dict, len(student_df))
                self._save_summary_csv(student_csv, per_feature_score)

            meta = {
                "align_method": "legacy_per_feature_dtw",
                "student_facing": str(student_facing),
                "reference_facing": str(self.reference_facing),
                "dominant_side": str(dom),
                "handedness_source": str(handedness_source),
            }
            return per_feature_score, frame_distances_dict, meta

        # -----------------------------
        # Branch B: NEW Pose-Align DTW path + aligned per-feature differences
        # -----------------------------
        # Flip reference pose in local coords if dominant side differs (left vs right throw)
        flip_ref_pose = (dom != self.reference_handedness)

        ref_pose = self._build_pose_matrix(self.reference_df, flip_x=flip_ref_pose)
        stu_pose = self._build_pose_matrix(student_df, flip_x=False)

        opp_side = "left" if dom == "right" else "right"

        ref_pose = self._apply_pose_joint_weight(
            ref_pose,
            side=opp_side,
            joint="ankle",
            factor=self.OPPOSITE_ANKLE_DTW_FACTOR,
        )

        stu_pose = self._apply_pose_joint_weight(
            stu_pose,
            side=opp_side,
            joint="ankle",
            factor=self.OPPOSITE_ANKLE_DTW_FACTOR,
        )

        pose_dtw_norm, pose_path = self._compute_pose_dtw_path(ref_pose, stu_pose)
        map_s_to_m = self._path_to_map_s_to_m(pose_path, n_student=len(student_df))

        # Save align path (optional)
        if save_frame_csv and save_align_path:
            self._save_pose_align_path_csv(student_csv, pose_path)
            self._save_pose_align_map_csv(student_csv, map_s_to_m)

        # Recompute per-frame distances for each teacher feature along map_s_to_m
        n_stu = len(student_df)
        for feat in self.FEATURE_NAMES:
            ref_seq = self.reference_features.get(feat, np.full(len(self.reference_df), np.nan))
            stu_seq = student_features.get(feat, np.full(n_stu, np.nan))

            ref_seq_i = self._interpolate_sequence(np.array(ref_seq, dtype=float))
            stu_seq_i = self._interpolate_sequence(np.array(stu_seq, dtype=float))

            aligned = np.full(n_stu, np.nan, dtype=float)
            for s in range(n_stu):
                m = int(map_s_to_m[s])
                if m < 0 or m >= len(ref_seq_i):
                    continue
                rv = float(ref_seq_i[m])
                sv = float(stu_seq_i[s])
                if np.isnan(rv) or np.isnan(sv):
                    continue

                if feat in self.ANGLE_FEATURES:
                    aligned[s] = abs(rv - sv)  # degrees
                else:
                    aligned[s] = self._aligned_relative_scalar(rv, sv)  # 0..1

            frame_distances_dict[feat] = aligned
            per_feature_score[feat] = float(np.nanmean(aligned)) if np.any(~np.isnan(aligned)) else np.nan

        # Overall score (same rule): angles normalized by 180 only for overall.
        vals_norm = []
        for feat in self.FEATURE_NAMES:
            v = per_feature_score.get(feat, np.nan)
            if np.isnan(v):
                continue
            if feat in self.ANGLE_FEATURES:
                vals_norm.append(min(v / self.ANGLE_MAX_DEG, 1.0))
            else:
                vals_norm.append(v)
        per_feature_score["overall_matching_score"] = float(np.mean(vals_norm)) if vals_norm else np.nan

        # Save aligned per-frame distances CSV for debugging + future pause-point selection
        if save_frame_csv:
            self._save_pose_aligned_frame_csv(student_csv, frame_distances_dict, n_frames=n_stu)
            self._save_summary_csv(student_csv.replace(".csv", "_pose_aligned.csv"), per_feature_score)

        meta = {
            "align_method": "pose_vector_dtw",
            "pose_dtw_norm": str(pose_dtw_norm),
            "student_facing": str(student_facing),
            "reference_facing": str(self.reference_facing),
            "dominant_side": str(dom),
            "handedness_source": str(handedness_source),
            "reference_handedness": str(self.reference_handedness),
            "flip_reference_pose": str(flip_ref_pose),
        }
        return per_feature_score, frame_distances_dict, meta

    # -----------------------------
    # CSV saving
    # -----------------------------
    def _save_frame_distances_csv(self, student_csv: str, frame_distances_dict: Dict[str, np.ndarray], n_frames: int):
        """
        Legacy: Save per-frame aligned distances for debugging.
        """
        data = {"frame": range(n_frames)}
        for feat in self.FEATURE_NAMES:
            data[feat] = frame_distances_dict.get(feat, np.full(n_frames, np.nan))

        df = pd.DataFrame(data)
        out = student_csv.replace(".csv", "_dtw_frame.csv")
        df.to_csv(out, index=False, float_format="%.4f")

    def _save_pose_aligned_frame_csv(self, student_csv: str, frame_distances_dict: Dict[str, np.ndarray], n_frames: int):
        """
        Pose-align: Save per-frame aligned distances along ONE pose DTW mapping.
        """
        data = {"frame": range(n_frames)}
        for feat in self.FEATURE_NAMES:
            data[feat] = frame_distances_dict.get(feat, np.full(n_frames, np.nan))
        df = pd.DataFrame(data)
        out = student_csv.replace(".csv", "_pose_aligned_frame.csv")
        df.to_csv(out, index=False, float_format="%.4f")

    def _save_pose_align_path_csv(self, student_csv: str, path: List[Tuple[int, int]]):
        """Save raw DTW path pairs (ref_idx, stu_idx)."""
        if not path:
            return
        df = pd.DataFrame(path, columns=["ref_idx", "stu_idx"])
        out = student_csv.replace(".csv", "_pose_align_path.csv")
        df.to_csv(out, index=False)

    def _save_pose_align_map_csv(self, student_csv: str, map_s_to_m: np.ndarray):
        """Save stable mapping map_s_to_m[s]=ref_idx."""
        df = pd.DataFrame({"stu_idx": np.arange(len(map_s_to_m)), "ref_idx": map_s_to_m.astype(int)})
        out = student_csv.replace(".csv", "_pose_align_map.csv")
        df.to_csv(out, index=False)

    def _save_summary_csv(self, student_csv: str, per_feature_score: Dict[str, float]):
        """
        Save per-variable average scores + overall score to a separate CSV.
        """
        rows = []
        for feat in self.FEATURE_NAMES:
            rows.append({"variable": feat, "avg_dtw": per_feature_score.get(feat, np.nan)})
        rows.append({"variable": "overall_matching_score", "avg_dtw": per_feature_score.get("overall_matching_score", np.nan)})

        df = pd.DataFrame(rows)
        # If caller passes a modified student_csv, keep consistent naming
        if student_csv.endswith(".csv"):
            out = student_csv.replace(".csv", "_dtw_summary.csv")
        else:
            out = student_csv + "_dtw_summary.csv"
        df.to_csv(out, index=False, float_format="%.6f")

    # Convenience printer
    def print_results(self, student_csv: str, handedness: str = "auto", use_pose_align: bool = False):
        scores, _, meta = self.compare(
            student_csv,
            handedness=handedness,
            save_frame_csv=False,
            use_pose_align=use_pose_align,
        )
        print("\n" + "=" * 70)
        print("DTW Teacher Feature Set (1 scalar per variable + overall)")
        print("=" * 70)
        print(f"Align method:     {meta.get('align_method', 'unknown')}")
        print(f"Reference facing: {meta.get('reference_facing')}")
        print(f"Student facing:   {meta.get('student_facing')}")
        print(f"Dominant side:    {meta.get('dominant_side')} ({meta.get('handedness_source')})")
        print("-" * 70)
        for k in self.FEATURE_NAMES:
            v = scores.get(k, np.nan)
            if np.isnan(v):
                print(f"{k:28s}: NaN")
            else:
                if k in self.ANGLE_FEATURES:
                    print(f"{k:28s}: {v:.2f} deg")
                else:
                    print(f"{k:28s}: {v:.4f} ({v*100:.1f}%)")
        print("-" * 70)
        ov = scores.get("overall_matching_score", np.nan)
        if np.isnan(ov):
            print("overall_matching_score        : NaN")
        else:
            print(f"overall_matching_score        : {ov:.4f} ({ov*100:.1f}%)")
        print("=" * 70)


# =========================
# Utilities: signed delta + curve png
# Put this block ABOVE: if __name__ == "__main__":
# =========================

from pathlib import Path
import numpy as np
import pandas as pd

def export_signed_deltas_from_align_map(
    reference_csv: str,
    student_csv: str,
    align_map_csv: str,
    out_csv: str,
    handedness: str = "auto",
    theta0_deg: float = 20.0,
):
    """
    Produce per-frame signed deltas (student - reference) aligned by an existing map CSV.
    align_map_csv must have columns: stu_idx, ref_idx (like your clip_pose_align_map.csv).

    Output: dtw_signed_delta.csv with columns:
      frame, ref_idx,
      <feat>_delta (signed), <feat>_abs, <feat>_rel
    For angles: _rel equals degrees delta for convenience.
    """
    comparator = FixedMotionDTW(reference_csv, reference_handedness="auto")

    student_df = pd.read_csv(student_csv)
    ref_df = comparator.reference_df

    # infer facing/dominant like compare() does
    student_facing = comparator._safe_mode(student_df, "body_facing", default="unknown")
    if handedness in ("left", "right"):
        dom = handedness
    else:
        if student_facing in ("left", "right"):
            dom = student_facing
        else:
            df_norm_for_infer = comparator._normalize_coordinates(student_df, student_facing)
            dom = comparator._infer_handedness(df_norm_for_infer)

    # extract features
    ref_feats = comparator.reference_features
    stu_feats = comparator._extract_features(student_df, facing=student_facing, dominant_side=dom)

    # load mapping
    map_df = pd.read_csv(align_map_csv)
    stu_idx = map_df["stu_idx"].astype(int).values
    ref_idx = map_df["ref_idx"].astype(int).values

    n = len(student_df)
    out = {
        "frame": np.arange(n, dtype=int),
        "ref_idx": np.full(n, -1, dtype=int),
    }

    # build a fast dict: stu -> ref
    s2r = {int(s): int(r) for s, r in zip(stu_idx, ref_idx)}
    for s in range(n):
        out["ref_idx"][s] = s2r.get(s, -1)

    def _safe_rel_signed(stu_v: float, ref_v: float, eps: float = 1e-6) -> float:
        if np.isnan(stu_v) or np.isnan(ref_v):
            return np.nan
        denom = max(abs(ref_v), eps)
        return float((stu_v - ref_v) / denom)

    for feat in comparator.FEATURE_NAMES:
        ref_seq = np.array(ref_feats.get(feat, np.full(len(ref_df), np.nan)), dtype=float)
        stu_seq = np.array(stu_feats.get(feat, np.full(n, np.nan)), dtype=float)

        ref_seq = comparator._interpolate_sequence(ref_seq)
        stu_seq = comparator._interpolate_sequence(stu_seq)

        delta = np.full(n, np.nan, dtype=float)
        absd  = np.full(n, np.nan, dtype=float)
        rel   = np.full(n, np.nan, dtype=float)
        refv  = np.full(n, np.nan, dtype=float)  # NEW
        stuv  = np.full(n, np.nan, dtype=float)  # NEW (optional)

        for s in range(n):
            r = int(out["ref_idx"][s])
            if r < 0 or r >= len(ref_seq):
                continue
            rv = float(ref_seq[r])
            sv = float(stu_seq[s])
            if np.isnan(rv) or np.isnan(sv):
                continue

            d = sv - rv
            delta[s] = d
            absd[s]  = abs(d)
            refv[s]  = rv
            stuv[s]  = sv

            if feat in comparator.ANGLE_FEATURES:
                # signed relative vs model angle at that moment
                denom = max(abs(rv), float(theta0_deg))
                rel[s] = float(d / denom)
            else:
                # signed relative vs model distance at that moment
                rel[s] = _safe_rel_signed(sv, rv)

        out[f"{feat}_delta"] = delta
        out[f"{feat}_abs"]   = absd
        out[f"{feat}_rel"]   = rel
        out[f"{feat}_ref"]   = refv   # NEW
        out[f"{feat}_stu"]   = stuv   # NEW (optional)

    df_out = pd.DataFrame(out)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, float_format="%.6f")
    return str(out_path)


def render_curve_png_from_aligned_frame_csv(
    aligned_frame_csv: str,
    out_png: str,
    smooth_window: int = 5,
    normalize: bool = True,
    # NEW: if provided, we will draw ABS curves from this signed delta csv instead (recommended)
    signed_delta_csv: str | None = None,
    theta0_deg: float = 20.0,   # angle denominator floor for relative normalization
    cap_abs: float | None = 2.0 # optional cap for abs rel (e.g., 2.0 => 200%) to avoid huge spikes
):
    """
    Draw an ABS curve PNG.

    Priority:
      1) If signed_delta_csv is provided and exists:
         - For angles: abs( (stu - ref) / max(|ref|, theta0_deg) )
           computed from <feat>_delta and a <feat>_ref if exists, else fallback to abs(delta)/180.
         - For distances: abs(<feat>_rel) if exists, else abs(<feat>_delta)/(abs(ref)+eps) if ref exists.
         This matches your intention: normalize by the model value at that moment.
      2) Else fallback to plotting raw columns from aligned_frame_csv (legacy behavior).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    def smooth(y: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return y
        return pd.Series(y).rolling(w, min_periods=1, center=True).mean().values

    # Feature list should match your pipeline
    feats = [
        "elbow_angle",
        "shoulder_angle",
        "hip_angle",
        "feet_lr_dist",
        "wrist_to_samehip_dist",
        "wrist_to_opphip_dist",
        "elbow_to_samehip_dist",
        "elbow_to_opphip_dist",
        "shoulder_to_wrist_y_dist",
        "shoulder_to_elbow_y_dist",
    ]
    angle_feats = {"elbow_angle", "shoulder_angle", "hip_angle"}

    # ---------- Preferred path: use signed_delta_csv ----------
    if signed_delta_csv is not None and Path(signed_delta_csv).exists():
        df = pd.read_csv(signed_delta_csv)
        x = df["frame"].values if "frame" in df.columns else np.arange(len(df))

        plt.figure(figsize=(12, 4))

        for feat in feats:
            if feat in angle_feats:
                # We expect <feat>_delta in degrees, and ideally <feat>_ref (model angle) if you saved it.
                delta_col = f"{feat}_delta"
                ref_col = f"{feat}_ref"  # optional, if you have it
                if delta_col not in df.columns:
                    continue

                delta = df[delta_col].astype(float).values

                if normalize:
                    if ref_col in df.columns:
                        ref = np.abs(df[ref_col].astype(float).values)
                        denom = np.maximum(ref, float(theta0_deg))
                        y = np.abs(delta) / denom
                    else:
                        # fallback if ref not available
                        y = np.abs(delta) / 180.0
                else:
                    y = np.abs(delta)

            else:
                # distances: prefer <feat>_rel (already relative)
                rel_col = f"{feat}_rel"
                delta_col = f"{feat}_delta"
                ref_col = f"{feat}_ref"  # optional
                if normalize:
                    if rel_col in df.columns:
                        y = np.abs(df[rel_col].astype(float).values)
                    elif delta_col in df.columns and ref_col in df.columns:
                        delta = df[delta_col].astype(float).values
                        ref = np.abs(df[ref_col].astype(float).values)
                        denom = ref + 1e-6
                        y = np.abs(delta) / denom
                    elif delta_col in df.columns:
                        # last resort: absolute delta (no good normalization available)
                        y = np.abs(df[delta_col].astype(float).values)
                    else:
                        continue
                else:
                    if rel_col in df.columns:
                        y = np.abs(df[rel_col].astype(float).values)
                    elif delta_col in df.columns:
                        y = np.abs(df[delta_col].astype(float).values)
                    else:
                        continue

            if cap_abs is not None:
                y = np.clip(y, 0.0, float(cap_abs))

            y = smooth(y, smooth_window)
            plt.plot(x, y, label=feat)

        plt.xlabel("frame")
        plt.ylabel("abs relative error" if normalize else "abs error")
        plt.title("DTW per-frame ABS error curves (relative to model)")
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()

        out_path = Path(out_png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=160)
        plt.close()
        return str(out_path)

    # ---------- Fallback: legacy behavior ----------
    df = pd.read_csv(aligned_frame_csv)
    x = df["frame"].values if "frame" in df.columns else np.arange(len(df))

    cols = [c for c in df.columns if c != "frame"]
    plt.figure(figsize=(12, 4))
    for c in cols:
        y = df[c].values.astype(float)
        y = smooth(y, smooth_window)
        plt.plot(x, y, label=c)

    plt.xlabel("frame")
    plt.ylabel("aligned error")
    plt.title("DTW aligned per-frame error curves (legacy)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return str(out_path)


def render_signed_curve_png_from_signed_delta_csv(
    signed_delta_csv: str,
    out_png: str,
    use_rel_for_dist: bool = True,
    normalize_angles: bool = True,   # now means "use relative for angles if possible"
    theta0_deg: float = 20.0,
    cap_signed: float | None = 2.0,
    smooth_window: int = 5,
):
    """
    Draw a SIGNED curve PNG from dtw_signed_delta.csv.

    - Dist features:
        * if use_rel_for_dist=True: use <feat>_rel (preferred)
        * else: use <feat>_delta
    - Angle features (degrees):
        * preferred: use <feat>_rel if present (relative to model at that moment)
        * else if <feat>_ref exists: use (delta / max(|ref|, theta0_deg))
        * else fallback: use delta/180
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(signed_delta_csv)
    x = df["frame"].values if "frame" in df.columns else np.arange(len(df))

    def smooth(y: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return y
        return pd.Series(y).rolling(w, min_periods=1, center=True).mean().values

    angle_feats = {"elbow_angle", "shoulder_angle", "hip_angle"}

    feats = [
        "elbow_angle",
        "shoulder_angle",
        "hip_angle",
        "feet_lr_dist",
        "wrist_to_samehip_dist",
        "wrist_to_opphip_dist",
        "elbow_to_samehip_dist",
        "elbow_to_opphip_dist",
        "shoulder_to_wrist_y_dist",
        "shoulder_to_elbow_y_dist",
    ]

    plt.figure(figsize=(12, 4))

    for feat in feats:
        if feat in angle_feats:
            rel_col = f"{feat}_rel"
            delta_col = f"{feat}_delta"
            ref_col = f"{feat}_ref"  # optional

            if normalize_angles and rel_col in df.columns:
                y = df[rel_col].astype(float).values
            elif delta_col in df.columns and ref_col in df.columns:
                delta = df[delta_col].astype(float).values
                ref = np.abs(df[ref_col].astype(float).values)
                denom = np.maximum(ref, float(theta0_deg))
                y = delta / denom
            elif delta_col in df.columns:
                # fallback
                y = df[delta_col].astype(float).values / 180.0 if normalize_angles else df[delta_col].astype(float).values
            else:
                continue

        else:
            col = f"{feat}_rel" if use_rel_for_dist else f"{feat}_delta"
            if col not in df.columns:
                continue
            y = df[col].astype(float).values

        if cap_signed is not None:
            y = np.clip(y, -float(cap_signed), float(cap_signed))

        y = smooth(y, smooth_window)
        plt.plot(x, y, label=feat)

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("frame")
    plt.ylabel("signed relative error" if normalize_angles else "signed delta")
    plt.title("DTW signed per-frame deltas (student vs model)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return out_png



if __name__ == "__main__":

    # Example usage (project-root relative)
    ref_csv = "reference/model.csv"
    stu_csv = "output/Eddie Throwing iPad recording 1/throws/throw_001/clip.csv"

    comparator = FixedMotionDTW(ref_csv, reference_handedness="auto")

    # Legacy (unchanged)
    comparator.print_results(stu_csv, handedness="auto", use_pose_align=False)

    # NEW pose-align mode
    comparator.print_results(stu_csv, handedness="auto", use_pose_align=True)
