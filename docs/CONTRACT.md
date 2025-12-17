# Throwing Analysis Tool – Engineering Contract

This project provides an automated pipeline for analyzing throwing motions.
Users upload one or multiple videos; the system automatically segments throws
(clips) and produces per-throw quantitative scores and annotated videos.


## Core hierarchy

The project follows a strict three-level hierarchy:

Session → Video → Throw

- Session: one user-triggered analysis run (e.g. one UI submission)
- Video: one input video within a session
- Throw: one segmented throwing action (one clip)


## Directory structure contract

All generated outputs MUST follow this structure:

runs/
  <session_id>/
    videos/
      <video_id>/
        input/
          original.mp4
        intermediate/
          raw_pose_data.csv
        events.json
        throws/
          throw_000/
            clip.mp4
            clip.csv
            summary.json
            dtw_summary.csv
            dtw_frame.csv
            annotated.mp4
            plots/



## Script responsibilities

fsm_new.py
- Responsible for:
  - Video preprocessing
  - Pose and ball detection
  - Throw segmentation
  - Writing clip.mp4, clip.csv, and events.json
- MUST NOT:
  - Compute DTW scores
  - Access reference templates

fixed_dtw_comparison.py
- Responsible for:
  - Per-throw DTW comparison
  - Automatic facing alignment (mirroring to reference)
  - Writing summary.json and annotated.mp4
- MUST NOT:
  - Access other throws
  - Assume session or video context

batch_dtw_compare.py
- Responsible for:
  - Reading summary.json files
  - Aggregating and ranking results
- MUST NOT:
  - Run pose detection or DTW


## Failure handling policy

- Failures are handled at the throw level.
- A failed throw does NOT invalidate the entire video or session.
- Each failed throw MUST still produce a summary.json with:
  - status = "failed"
  - fail_reason describing the error.


## Reference and facing rules

- A single fixed reference template is used (CSV only).
- Student motion is always mirrored to match the reference facing.
- Facing decision and mirroring status MUST be recorded in summary.json.
