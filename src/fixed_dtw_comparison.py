#!/usr/bin/env python3
"""
Fixed Motion Comparison - Proper Normalization + Teacher Feature Set

Teacher-required pipeline:
Step 1) For each variable, compute DTW alignment and output ONE scalar per variable:
        -> average of aligned (DTW-warped) per-frame relative distances.
Step 2) Overall matching score = average of all variable scalars.

Also supports left-handed students:
- Distances/angles that are "right side" are computed on dominant side instead.
"""

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")


class FixedMotionDTW:
    """
    Properly normalized DTW comparison
    - Origin: hip center (sequence mean)
    - Scale: body_height (shoulder-hip vertical distance)
    - Facing: mirror x if needed (based on body_facing mode)
    - DTW distance: relative difference (percentage-like, 0..1 capped)
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
        "shoulder_to_wrist_y_dist",         # |y_shoulder - y_wrist|
        "shoulder_to_elbow_y_dist",         # |y_shoulder - y_elbow|
    ]

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

    # -----------------------------
    # Normalization
    # -----------------------------
    def _normalize_coordinates(self, df: pd.DataFrame, facing: str) -> pd.DataFrame:
        """
        1) Compute body_height = |mean(shoulder_y) - mean(hip_y)|
        2) Compute hip center (mean) as origin
        3) Normalize all *_x/*_y by subtracting origin and dividing by body_height
        4) Mirror x if facing != reference_facing
        """
        df = df.copy()

        # shoulder_y candidates
        shoulder_y = self._pick_series(df, ["right_shoulder_y", "shoulder_y", "left_shoulder_y"])
        # hip_y candidates
        hip_y = self._pick_series(df, ["hip_y", "right_hip_y", "left_hip_y"])

        if shoulder_y is None or hip_y is None:
            # last resort: use any *_y columns mean span
            y_cols = [c for c in df.columns if c.endswith("_y")]
            if len(y_cols) == 0:
                body_height = 100.0
            else:
                body_height = float(abs(df[y_cols].mean(numeric_only=True).max() - df[y_cols].mean(numeric_only=True).min()))
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
            # If hip is missing, fall back to mean of all coords (not ideal but prevents crash)
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

        # Common naming variants
        if joint == "wrist":
            # in your current code, wrist may be stored as hand_x/hand_y
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
    # Handedness
    # -----------------------------
    def _infer_handedness(self, df_norm: pd.DataFrame) -> str:
        """
        Heuristic:
        choose side whose wrist has larger motion magnitude (std of distance to hip).
        If left/right wrist columns are unavailable, fall back to 'right'.
        """
        # need both wrists to infer
        has_lw = self._has_series(df_norm, "left_wrist_x") or self._has_series(df_norm, "left_hand_x") or self._has_series(df_norm, "left_wrist_y") or self._has_series(df_norm, "left_hand_y")
        has_rw = self._has_series(df_norm, "right_wrist_x") or self._has_series(df_norm, "right_hand_x") or self._has_series(df_norm, "right_wrist_y") or self._has_series(df_norm, "right_hand_y")

        has_lhip = self._has_series(df_norm, "left_hip_x") and self._has_series(df_norm, "left_hip_y")
        has_rhip = self._has_series(df_norm, "right_hip_x") and self._has_series(df_norm, "right_hip_y")

        if not (has_lw and has_rw and has_lhip and has_rhip):
            return "right"

        # build distances
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
    # Feature extraction (teacher set)
    # -----------------------------
    def _extract_features(self, df: pd.DataFrame, facing: str, dominant_side: str) -> Dict[str, np.ndarray]:
        """
        Extract teacher-required sequences (all already normalized).

        dominant_side: 'left' or 'right'
        """
        df_norm = self._normalize_coordinates(df, facing)
        n = len(df_norm)

        dom = dominant_side
        opp = "left" if dom == "right" else "right"

        feats: Dict[str, np.ndarray] = {}

        # 1) elbow_angle (shoulder - elbow - wrist) on dominant side
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

            # elbow angle
            if not (np.isnan(sh[0]) or np.isnan(el[0]) or np.isnan(wr[0])):
                elbow_angles.append(self._calculate_angle(sh, el, wr))
            else:
                elbow_angles.append(np.nan)

            # shoulder angle (elbow - shoulder - hip)
            if not (np.isnan(el[0]) or np.isnan(sh[0]) or np.isnan(hp[0])):
                shoulder_angles.append(self._calculate_angle(el, sh, hp))
            else:
                shoulder_angles.append(np.nan)

            # hip angle (shoulder - hip - knee)
            if not (np.isnan(sh[0]) or np.isnan(hp[0]) or np.isnan(kn[0])):
                hip_angles.append(self._calculate_angle(sh, hp, kn))
            else:
                hip_angles.append(np.nan)

            # vertical-only distances
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

        # 2) distance between left & right feet (normalized)
        # Use ankles/feet if available
        feet_dist = []
        for _, row in df_norm.iterrows():
            lf = self._get_point(row, "left", "ankle")
            rf = self._get_point(row, "right", "ankle")
            if not (np.isnan(lf[0]) or np.isnan(rf[0])):
                feet_dist.append(np.sqrt((lf[0] - rf[0]) ** 2 + (lf[1] - rf[1]) ** 2))
            else:
                feet_dist.append(np.nan)
        feats["feet_lr_dist"] = np.array(feet_dist, dtype=float)

        # 3) wrist/elbow distances to hips (dominant vs opposite)
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

        # Ensure all required keys exist (even if NaN)
        for k in self.FEATURE_NAMES:
            if k not in feats:
                feats[k] = np.full(n, np.nan, dtype=float)

        return feats

    # -----------------------------
    # DTW + "average across frames"
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

    def _compute_dtw(self, ref_seq: np.ndarray, stu_seq: np.ndarray) -> Tuple[float, list, np.ndarray]:
        """
        Returns:
            dtw_norm: distance/len(path) (kept for debugging)
            path: DTW path
            frame_dists: aligned per-student-frame distances (relative)
        """
        ref_i = self._interpolate_sequence(ref_seq)
        stu_i = self._interpolate_sequence(stu_seq)

        if np.any(np.isnan(ref_i)) or np.any(np.isnan(stu_i)):
            return np.nan, [], np.full(len(stu_seq), np.nan)

        if len(ref_i) < 3 or len(stu_i) < 3:
            return np.nan, [], np.full(len(stu_seq), np.nan)

        ref_r = ref_i.reshape(-1, 1)
        stu_r = stu_i.reshape(-1, 1)

        dist, path = fastdtw(ref_r, stu_r, dist=self._relative_distance)
        dtw_norm = float(dist) / max(len(path), 1)

        frame_dists = self._compute_frame_distances_from_path(ref_i, stu_i, path, use_relative=True)
        return dtw_norm, path, frame_dists

    def compare(
        self,
        student_csv: str,
        handedness: str = "auto",
        save_frame_csv: bool = False,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, str]]:
        """
        Teacher-required outputs:
        - per_feature_score: ONE scalar per variable = mean(aligned frame distances)
        - frame_distances_dict: (optional) for debugging/visualization
        - meta: used handedness & facing

        handedness:
            'right' / 'left' / 'auto' (auto tries to infer from motion)
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


        student_features = self._extract_features(student_df, facing=student_facing, dominant_side=dom)

        per_feature_score: Dict[str, float] = {}
        frame_distances_dict: Dict[str, np.ndarray] = {}

        for feat in self.FEATURE_NAMES:
            ref_seq = self.reference_features.get(feat, np.full(len(self.reference_df), np.nan))
            stu_seq = student_features.get(feat, np.full(len(student_df), np.nan))

            _, _, frame_dists = self._compute_dtw(ref_seq, stu_seq)
            frame_distances_dict[feat] = frame_dists

            # Teacher Step 1: 1 scalar per variable = average across frames
            per_feature_score[feat] = float(np.nanmean(frame_dists)) if np.any(~np.isnan(frame_dists)) else np.nan

        # Teacher Step 2: overall matching score = average across variable scalars
        vals = [v for v in per_feature_score.values() if not np.isnan(v)]
        per_feature_score["overall_matching_score"] = float(np.mean(vals)) if vals else np.nan

        # Optional: save per-frame distances CSV (debug)
        if save_frame_csv:
            self._save_frame_distances_csv(student_csv, frame_distances_dict, len(student_df))
            self._save_summary_csv(student_csv, per_feature_score)


        meta = {
            "student_facing": str(student_facing),
            "reference_facing": str(self.reference_facing),
            "dominant_side": str(dom),
            "handedness_source": str(handedness_source),
        }
        return per_feature_score, frame_distances_dict, meta

    def _save_frame_distances_csv(self, student_csv: str, frame_distances_dict: Dict[str, np.ndarray], n_frames: int):
        """
        Save per-frame aligned distances for debugging.
        """
        data = {"frame": range(n_frames)}
        for feat in self.FEATURE_NAMES:
            data[feat] = frame_distances_dict.get(feat, np.full(n_frames, np.nan))

        df = pd.DataFrame(data)
        out = student_csv.replace(".csv", "_dtw_frame.csv")
        df.to_csv(out, index=False, float_format="%.4f")
        
    def _save_summary_csv(self, student_csv: str, per_feature_score: Dict[str, float]):
        """
        Save per-variable average scores + overall score to a separate CSV.
        """
        rows = []
        for feat in self.FEATURE_NAMES:
            rows.append({"variable": feat, "avg_dtw": per_feature_score.get(feat, np.nan)})
        rows.append({"variable": "overall_matching_score", "avg_dtw": per_feature_score.get("overall_matching_score", np.nan)})

        df = pd.DataFrame(rows)
        out = student_csv.replace(".csv", "_dtw_summary.csv")
        df.to_csv(out, index=False, float_format="%.6f")


    # Convenience printer
    def print_results(self, student_csv: str, handedness: str = "auto"):
        scores, _, meta = self.compare(student_csv, handedness=handedness, save_frame_csv=False)
        print("\n" + "=" * 70)
        print("DTW Teacher Feature Set (1 scalar per variable + overall)")
        print("=" * 70)
        print(f"Reference facing: {meta['reference_facing']}")
        print(f"Student facing:   {meta['student_facing']}")
        print(f"Dominant side:    {meta['dominant_side']} ({meta['handedness_source']})")
        print("-" * 70)
        for k in self.FEATURE_NAMES:
            v = scores.get(k, np.nan)
            if np.isnan(v):
                print(f"{k:28s}: NaN")
            else:
                print(f"{k:28s}: {v:.4f} ({v*100:.1f}%)")
        print("-" * 70)
        ov = scores.get("overall_matching_score", np.nan)
        if np.isnan(ov):
            print("overall_matching_score        : NaN")
        else:
            print(f"overall_matching_score        : {ov:.4f} ({ov*100:.1f}%)")
        print("=" * 70)




if __name__ == "__main__":
    # Example usage (adjust paths yourself)
    ref_csv = "model/standard_throw.csv"
    stu_csv = "output/video1/clips/throw_01_frame_114_to_182.csv"

    comparator = FixedMotionDTW(ref_csv, reference_handedness="auto")
    comparator.print_results(stu_csv, handedness="auto")

