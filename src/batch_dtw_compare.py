#!/usr/bin/env python3
"""
One-Click Batch DTW Comparison (Teacher Feature Set)

Outputs:
- One scalar per variable (avg across aligned frames)
- overall_matching_score = average of variable scalars
- (optional) per-frame DTW CSV for debugging (enabled)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Import DTW engine
sys.path.append(os.path.dirname(__file__))
try:
    from fixed_dtw_comparison import FixedMotionDTW
except ImportError:
    print("Error: fixed_dtw_comparison.py not found!")
    sys.exit(1)


class BatchDTWComparer:
    def __init__(self, model_dir: str = "model", output_dir: str = "output"):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)

        if not self.model_dir.exists():
            raise ValueError(f"Model directory not found: {self.model_dir}")
        if not self.output_dir.exists():
            raise ValueError(f"Output directory not found: {self.output_dir}")

        self.model_csvs = sorted(list(self.model_dir.glob("*.csv")))
        if len(self.model_csvs) == 0:
            raise ValueError(f"No CSV files found in {self.model_dir}")

        self.reference_csv = self.model_csvs[0]
        print(f"✓ Using reference model: {self.reference_csv.name}")

        ref_df = pd.read_csv(self.reference_csv)
        self.reference_facing = ref_df["body_facing"].mode().iloc[0] if "body_facing" in ref_df.columns else "unknown"
        print(f"✓ Reference facing: {self.reference_facing}")

        # Initialize DTW comparator (assume reference model is right-handed)
        self.comparator = FixedMotionDTW(str(self.reference_csv), reference_handedness="auto")


        # Teacher feature list
        self.feature_names = list(self.comparator.FEATURE_NAMES)

        self._scan_clips()

    def _scan_clips(self):
        self.all_clips = []
        video_folders = [d for d in self.output_dir.iterdir() if d.is_dir() and (d / "clips").exists()]
        if len(video_folders) == 0:
            raise ValueError(f"No video folders with clips/ found in {self.output_dir}")

        print(f"\n✓ Found {len(video_folders)} video folders:")
        for video_folder in sorted(video_folders):
            video_name = video_folder.name
            clips_dir = video_folder / "clips"
            csv_files = sorted([p for p in clips_dir.glob("*.csv") if "_dtw_frame" not in p.stem])

            print(f"  - {video_name}: {len(csv_files)} clips")

            for csv_file in csv_files:
                self.all_clips.append(
                    {
                        "video_name": video_name,
                        "csv_path": csv_file,
                        "clip_name": csv_file.stem,
                    }
                )

        print(f"\n✓ Total clips to process: {len(self.all_clips)}")

    def _extract_clip_info(self, clip_name: str) -> dict:
        parts = clip_name.split("_")
        info = {"throw_num": None, "start_frame": None, "end_frame": None}

        try:
            if "throw" in parts:
                throw_idx = parts.index("throw")
                info["throw_num"] = int(parts[throw_idx + 1])

            if "frame" in parts:
                frame_idx = parts.index("frame")
                info["start_frame"] = int(parts[frame_idx + 1])
                info["end_frame"] = int(parts[frame_idx + 3])
        except Exception:
            pass

        return info

    def compare_all(self) -> pd.DataFrame:
        results = []

        print("\n" + "=" * 70)
        print("COMPARING ALL CLIPS (Teacher Feature Set)")
        print("=" * 70)

        for i, clip in enumerate(self.all_clips, 1):
            print(f"[{i}/{len(self.all_clips)}] {clip['video_name']}/{clip['clip_name']}")

            try:
                clip_df = pd.read_csv(clip["csv_path"])
                clip_facing = clip_df["body_facing"].mode().iloc[0] if "body_facing" in clip_df.columns else "unknown"

                clip_info = self._extract_clip_info(clip["clip_name"])

                # Compare (auto handedness) + save per-frame CSV for debugging
                scores, _, meta = self.comparator.compare(
                    str(clip["csv_path"]),
                    handedness="auto",
                    save_frame_csv=True,
                )

                row = {
                    "video_name": clip["video_name"],
                    "clip_name": clip["clip_name"],
                    "throw_num": clip_info["throw_num"],
                    "start_frame": clip_info["start_frame"],
                    "end_frame": clip_info["end_frame"],
                    "num_frames": len(clip_df),
                    "body_facing": clip_facing,
                    "facing_match": (clip_facing == self.reference_facing),
                    "dominant_side": meta.get("dominant_side", "unknown"),
                    "handedness_source": meta.get("handedness_source", "unknown"),
                }

                # Per-variable scalar scores
                for feat in self.feature_names:
                    row[feat] = scores.get(feat, np.nan)

                # Overall
                row["overall_matching_score"] = scores.get("overall_matching_score", np.nan)

                results.append(row)

                print(
                    f"  Facing: {clip_facing} {'✓' if row['facing_match'] else '(mirrored)'}; "
                    f"Side: {row['dominant_side']} ({row['handedness_source']}); "
                    f"Overall: {row['overall_matching_score']:.4f}"
                )

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        print("=" * 70)
        print(f"✓ Done: {len(results)}/{len(self.all_clips)} clips processed")

        return pd.DataFrame(results)

    def save_outputs(self, df: pd.DataFrame, results_dir: str = "results"):
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        out_csv = results_dir / "dtw_teacher_features_all_clips.csv"
        df.to_csv(out_csv, index=False, float_format="%.6f")

        # Simple summary
        out_txt = results_dir / "summary_teacher_features.txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("BATCH SUMMARY (Teacher Feature Set)\n")
            f.write("=" * 70 + "\n")
            f.write(f"Reference model: {self.reference_csv.name}\n")
            f.write(f"Reference facing: {self.reference_facing}\n")
            f.write(f"Total clips: {len(df)}\n\n")

            if "overall_matching_score" in df.columns and len(df) > 0:
                f.write("OVERALL MATCHING SCORE:\n")
                f.write(f"  mean: {df['overall_matching_score'].mean():.6f}\n")
                f.write(f"  min : {df['overall_matching_score'].min():.6f}\n")
                f.write(f"  max : {df['overall_matching_score'].max():.6f}\n\n")

            f.write("TOP-10 BEST (lowest overall):\n")
            best = df.sort_values("overall_matching_score").head(10)
            for _, r in best.iterrows():
                f.write(f"  {r['video_name']}/{r['clip_name']}: {r['overall_matching_score']:.6f}\n")

            f.write("\nTOP-10 WORST (highest overall):\n")
            worst = df.sort_values("overall_matching_score", ascending=False).head(10)
            for _, r in worst.iterrows():
                f.write(f"  {r['video_name']}/{r['clip_name']}: {r['overall_matching_score']:.6f}\n")

        print(f"\n✓ Saved: {out_csv}")
        print(f"✓ Saved: {out_txt}")

    def run(self):
        df = self.compare_all()
        self.save_outputs(df)


if __name__ == "__main__":
    comparer = BatchDTWComparer(model_dir="model", output_dir="output")
    comparer.run()
