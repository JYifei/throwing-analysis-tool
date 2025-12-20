"""
Tennis Detection using YOLOv8 + Mediapipe Pose
- Forearm length for scaling across different resolution(elbow-wrist)
- Interpolation (<=3 frames) for ball_x, ball_y, distance
- Delta ratio + absolute ratio gating for release detection
- Standing gate: must stand ~1s before allowing a release
- Cooldown: after a release, require standing again to arm next detection
- Marked plot with release lines
- Fixed: Frame overlay drawn AFTER detection to avoid interference
- NEW: Caching mechanism to skip AI inference on repeat runs
- NEW: Full body chain analysis (Shoulder/Hip/Knee) for accurate start detection
- NEW: Robust Adaptive algorithm for clip start detection
- FIX: Realtime preview and Video Saving restored
- FEATURE: Auto-extract clips after analysis
"""
from __future__ import annotations
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque
import matplotlib.pyplot as plt
from math import ceil
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys

_THIS_DIR = Path(__file__).resolve().parent  # .../src
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))




class TennisDetector:
    def __init__(self, model_path='yolov8n.pt', confidence=0.08, iou=0.45):
        from pathlib import Path
        # --- resolve model path relative to this script (not CWD) ---
        script_dir = Path(__file__).resolve().parent
        model_path = (script_dir / model_path).resolve()
        print(f"[INFO] Using model: {model_path}")

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.iou = iou
        self.ball_class_id = 32
        # allow similar small round objects as candidates (optional but robust)
        self.classes = [32]

        # --- initialize Mediapipe Pose once ---
        self.mp_pose = mp.solutions.pose
        # Original code (in __init__)
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,            # Current setting (medium model)
            enable_segmentation=False,
            min_detection_confidence=0.5,  # Current setting (too loose)
            min_tracking_confidence=0.5    # Current setting (too loose)
        )
        
        
    # --- metadata-based rotation utilities ---
    def _get_rotation_metadata(self, video_path: str) -> int | None:
        """
        Return rotation degrees (-90, 90, 180) from metadata, or None if unavailable.
        """
        import shutil, json, subprocess, re, math
        from pathlib import Path

        exe = shutil.which("ffprobe")
        if not exe:
            print("[ROT] ffprobe not found; skip metadata rotation.")
            return None

        # ---- JSON: streams + format (most robust)
        cmd_json = [
            exe, "-v", "error", "-select_streams", "v:0",
            "-show_streams", "-show_format",
            "-show_entries",
            "stream=index,tags,side_data_list:format=format_name,format_long_name,tags",
            "-print_format", "json", video_path
        ]
        try:
            out = subprocess.check_output(cmd_json, stderr=subprocess.STDOUT)
            info = json.loads(out.decode("utf-8", errors="ignore"))
            streams = info.get("streams") or []
            if streams:
                s0 = streams[0]
                # 1) tags.rotate
                rot = (s0.get("tags") or {}).get("rotate")
                if rot is not None:
                    return int(rot)
                # 2) side_data_list.rotation
                for sd in s0.get("side_data_list") or []:
                    if sd.get("side_data_type") == "Display Matrix" and "rotation" in sd:
                        return int(sd["rotation"])
            # 3) format.tags.rotate
            fmt = info.get("format") or {}
            rot_fmt = (fmt.get("tags") or {}).get("rotate")
            if rot_fmt is not None:
                return int(rot_fmt)
        except Exception:
            pass

        # ---- Text: displaymatrix (look for explicit rotation line)
        cmd_txt = [
            exe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream_side_data=displaymatrix",
            "-of", "default=nw=1", video_path
        ]
        try:
            text = subprocess.check_output(cmd_txt, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
            m = re.search(r"rotation:\s*(-?\d+)", text)
            if m:
                return int(m.group(1))

            # ---- Fallback: parse 3x3 display matrix
            rows = re.findall(r"displaymatrix.*?\n\s*([^\n]+)\n\s*([^\n]+)\n\s*([^\n]+)", text, re.IGNORECASE | re.DOTALL)
            if rows:
                def row_to_nums(s):
                    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", s)]
                r1 = row_to_nums(rows[0][0]); r2 = row_to_nums(rows[0][1]); r3 = row_to_nums(rows[0][2])
                if len(r1) >= 2 and len(r2) >= 2:
                    a, b = r1[0], r1[1]
                    c, d = r2[0], r2[1]
                    def near(x, y, eps=1e-3): return abs(x - y) < eps
                    if near(a, 0) and near(b, 1) and near(c, -1) and near(d, 0): return -90
                    if near(a, 0) and near(b, -1) and near(c, 1) and near(d, 0): return 90
                    if near(a, -1) and near(b, 0) and near(c, 0) and near(d, -1): return 180
                    angle = math.degrees(math.atan2(c, a))
                    for cand in (-90, 90, 180, 0):
                        if abs((angle - cand + 180) % 360 - 180) < 10: return cand
        except Exception:
            pass
        return None

    def _rotation_to_code(self, rot_degrees: int | None):
        if rot_degrees in (-90, 270): return cv2.ROTATE_90_CLOCKWISE
        if rot_degrees in (90, -270): return cv2.ROTATE_90_COUNTERCLOCKWISE
        if rot_degrees in (180, -180): return cv2.ROTATE_180
        return None

    def detect_video(self, video_path, output_path='output_detection.mp4',
                  show_realtime=True, use_tracker=False, img_size=640, video_dir=None):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- metadata-based rotation ---
        rot_meta = self._get_rotation_metadata(video_path)
        rotate_code = self._rotation_to_code(rot_meta)
        
        # Adjust output dimensions (if rotated 90 degrees, swap width and height)
        out_width, out_height = width, height
        if rotate_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            out_width, out_height = height, width

        angle_map = {
            None: "0° (no-op)", cv2.ROTATE_90_CLOCKWISE: "90° CW",
            cv2.ROTATE_180: "180°", cv2.ROTATE_90_COUNTERCLOCKWISE: "90° CCW"
        }
        print(f"[INFO] Orientation by metadata: {rot_meta}°, apply: {angle_map.get(rotate_code, '0°')}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Fix: VideoWriter uses rotated width and height
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_width, out_height))

        # --- preview window setup ---
        WIN = "Detection with Pose"
        PREVIEW_W = 960
        if show_realtime:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            # Calculate preview window height, maintain aspect ratio
            PREVIEW_H0 = max(1, int(PREVIEW_W * out_height / out_width))
            cv2.resizeWindow(WIN, PREVIEW_W, PREVIEW_H0)

        frame_count = 0
        all_records = []
        
        standing_frames_rt = []
        release_frames_rt = []

        # === State machine variables ===
        detection_state = "waiting_for_standing"
        yolo_call_count = 0
        stand_frames = max(1, int(1.0 * fps))
        v_thresh = 0.08
        speed_norm_buffer = deque(maxlen=stand_frames)
        buffer_frames = int(0.5 * fps)
        cooldown_frames_remaining = 0
        ball_frames_in_session = []
        release_detected_by_ratio = False
        
        # Log
        from pathlib import Path

        if video_dir is not None:
            log_dir = Path(video_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            state_log_path = str(log_dir / "state_log.txt")
        else:
            state_log_path = os.path.join(os.path.dirname(output_path), "state_log.txt")

        state_log = open(state_log_path, 'w', encoding='utf-8')
        state_log.write(f"State Machine Log\n{'='*60}\n")


        # === Cache check logic ===
        from pathlib import Path
        # Cache path
        if video_dir is not None:
            intermediate_dir = Path(video_dir) / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            cache_file_path = str(intermediate_dir / "raw_pose_data.csv")
        else:
            cache_file_path = "raw_pose_data.csv"  # fallback to old behavior

        df_cache = None
        if os.path.exists(cache_file_path):
            print(f"\n[CACHE] Found {cache_file_path}. Loading data to SKIP AI inference...")
            df_cache = pd.read_csv(cache_file_path)
            print(f"[CACHE] Loaded {len(df_cache)} frames.")
        else:
            print("\n[CACHE] No cache found. Running full YOLO + MediaPipe inference...")


        print(f"Processing video: {video_path}")
        print(f"Resolution (Raw): {width}x{height}, FPS: {fps}, Frames: {total_frames}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- Rotate frame ---
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
                # Now frame w,h has changed, use frame.shape below to get actual dimensions
                # We use frame.shape here to get current frame's actual width and height to avoid confusion
                curr_h, curr_w = frame.shape[:2]
            else:
                curr_h, curr_w = height, width
                
            frame_count += 1

            # Initialize variables
            shoulder_mid_x_pixel = None
            right_wrist, right_elbow, forearm_length = None, None, None
            hip_y = None
            body_facing = None
            closest_ball, distance = None, None
            right_shoulder, right_hip, right_knee = None, None, None
            feet_dist_pixel = None

            # =========================================================
            # Branch: Use cached data OR run AI models
            # =========================================================
            cache_hit = False
            if df_cache is not None and (frame_count - 1) < len(df_cache):
                try:
                    row = df_cache.iloc[frame_count - 1]
                    def get_pt(r, x_col, y_col):
                        if pd.notna(r.get(x_col)) and pd.notna(r.get(y_col)):
                            return (int(r[x_col]), int(r[y_col]))
                        return None

                    # Restore data
                    right_wrist = get_pt(row, 'hand_x', 'hand_y')
                    right_elbow = get_pt(row, 'elbow_x', 'elbow_y')
                    closest_ball = get_pt(row, 'ball_x', 'ball_y')
                    
                    right_shoulder = get_pt(row, 'right_shoulder_x', 'right_shoulder_y')
                    right_hip = get_pt(row, 'right_hip_x', 'right_hip_y')
                    right_knee = get_pt(row, 'right_knee_x', 'right_knee_y')
                    
                    forearm_length = float(row['forearm_length']) if pd.notna(row.get('forearm_length')) else None
                    distance = float(row['distance']) if pd.notna(row.get('distance')) else None
                    hip_y = float(row['hip_y']) if pd.notna(row.get('hip_y')) else None
                    
                    if pd.notna(row.get('shoulder_mid_x')):
                        shoulder_mid_x_pixel = float(row['shoulder_mid_x'])
                    body_facing = row['body_facing'] if 'body_facing' in row else None
                    
                    cache_hit = True
                except Exception as e:
                    pass

            # Define a dictionary to store all data for the current frame
            frame_data = {'frame': frame_count, 'fps': fps}

            if not cache_hit:
                # --- Step 1: Mediapipe Pose detection ---
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(rgb)

                if pose_results.pose_landmarks:
                    lm = pose_results.pose_landmarks.landmark
                    
                    # === [NEW] Loop to extract all 33 keypoints ===
                    for i, landmark in enumerate(lm):
                        name = self.mp_pose.PoseLandmark(i).name.lower() # Get joint name (nose, left_shoulder...)
                        frame_data[f"{name}_x"] = int(landmark.x * curr_w)
                        frame_data[f"{name}_y"] = int(landmark.y * curr_h)
                        frame_data[f"{name}_v"] = landmark.visibility

                    # === [KEEP] To maintain original logic, reassign variables ===
                    # Get from just extracted data, or get from lm again
                    rw = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    re = lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                    
                    wrist_pt_raw = np.array([rw.x * curr_w, rw.y * curr_h])
                    elbow_pt_raw = np.array([re.x * curr_w, re.y * curr_h])
                                        
                    if rw.visibility > 0.4:
                        right_wrist = (int(wrist_pt_raw[0]), int(wrist_pt_raw[1]))
                    if re.visibility > 0.4:
                        right_elbow = (int(elbow_pt_raw[0]), int(elbow_pt_raw[1]))
                    if rw.visibility > 0.4 and re.visibility > 0.4:
                        forearm_length = float(np.linalg.norm(wrist_pt_raw - elbow_pt_raw))

                    # Left hand detection (for left-handed players)
                    lw = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    le = lm[self.mp_pose.PoseLandmark.LEFT_ELBOW]
                    
                    left_wrist = None
                    left_elbow = None
                    left_forearm_length = None
                    
                    lw_pt_raw = np.array([lw.x * curr_w, lw.y * curr_h])
                    le_pt_raw = np.array([le.x * curr_w, le.y * curr_h])
                    
                    if lw.visibility > 0.4:
                        left_wrist = (int(lw_pt_raw[0]), int(lw_pt_raw[1]))
                    if le.visibility > 0.4:
                        left_elbow = (int(le_pt_raw[0]), int(le_pt_raw[1]))
                    if lw.visibility > 0.4 and le.visibility > 0.4:
                        left_forearm_length = float(np.linalg.norm(lw_pt_raw - le_pt_raw))

                    # 2. Right shoulder
                    rs = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    if rs.visibility > 0.4:
                        right_shoulder = (int(rs.x * curr_w), int(rs.y * curr_h))
                        ls = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                        if ls.visibility > 0.4:
                             shoulder_mid_x_pixel = ((rs.x + ls.x)/2) * curr_w

                    # 3. Right hip
                    rh = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
                    if rh.visibility > 0.4:
                        right_hip = (int(rh.x * curr_w), int(rh.y * curr_h))
                        hip_y = right_hip[1]

                    # 4. Right knee
                    rk = lm[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                    if rk.visibility > 0.4:
                        right_knee = (int(rk.x * curr_w), int(rk.y * curr_h))

                    # 5. Both feet
                    rankle = lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
                    lankle = lm[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                    if rankle.visibility > 0.4 and lankle.visibility > 0.4:
                        ra = np.array([rankle.x * curr_w, rankle.y * curr_h])
                        la = np.array([lankle.x * curr_w, lankle.y * curr_h])
                        feet_dist_pixel = float(np.linalg.norm(ra - la))

                    # 6. Body facing direction
                    nose = lm[self.mp_pose.PoseLandmark.NOSE]
                    ls = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    if ls.visibility > 0.4 and rs.visibility > 0.4:
                          shoulder_mid_x = (ls.x + rs.x) / 2
                          if nose.visibility > 0.3:
                              body_facing = "right" if nose.x > shoulder_mid_x else "left"
                          else:
                              body_facing = "right" if rs.x > ls.x else "left"

                    # === AUTO-SELECT THROWING HAND based on body_facing ===
                    # Determine which hand to track based on facing direction
                    if body_facing == "left":
                        # Person facing left -> use LEFT hand
                        throwing_hand = "left"
                        active_wrist = left_wrist
                        active_elbow = left_elbow
                        active_forearm = left_forearm_length
                    else:
                        # Person facing right or uncertain -> use RIGHT hand (default)
                        throwing_hand = "right"
                        active_wrist = right_wrist
                        active_elbow = right_elbow
                        active_forearm = forearm_length
                    
                    # Update the tracking variables to use the selected hand
                    if active_wrist:
                        right_wrist = active_wrist  # Keep variable name for compatibility
                    if active_elbow:
                        right_elbow = active_elbow  # Keep variable name for compatibility
                    if active_forearm:
                        forearm_length = active_forearm  # Keep variable name for compatibility

                # --- Step 2: YOLO detection ---
                run_yolo = (detection_state == "detecting_throw")
                balls_in_frame = []
                if run_yolo:
                    yolo_call_count += 1
                    if use_tracker:
                        results = self.model.track(frame, conf=self.confidence, iou=self.iou, imgsz=img_size, persist=True, verbose=False, classes=self.classes)
                    else:
                        results = self.model(frame, conf=self.confidence, iou=self.iou, imgsz=img_size, verbose=False, augment=True, classes=self.classes)
                    
                    for result in results:
                        for box in result.boxes:
                            if int(box.cls[0]) == self.ball_class_id:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0])
                                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                                if hip_y is not None:
                                    waist_margin = int(0.03 * curr_h)
                                    if cy > hip_y + waist_margin: continue
                                balls_in_frame.append({'center': (cx, cy), 'conf': conf})

                # --- Step 3: Distance ---
                if right_wrist and balls_in_frame:
                    y_tol = int(0.1 * curr_h)
                    candidates = []
                    for ball in balls_in_frame:
                        bx, by = ball['center']
                        if by > right_wrist[1] + y_tol: continue
                        d = float(np.linalg.norm(np.array([bx, by]) - np.array(right_wrist)))
                        candidates.append((d, bx, by))
                    if candidates:
                        d_min, bx_min, by_min = min(candidates, key=lambda x: x[0])
                        closest_ball, distance = (bx_min, by_min), d_min

            # === State machine logic ===
            if len(all_records) >= 1 and forearm_length and forearm_length > 0:
                prev_record = all_records[-1]
                if prev_record['right_wrist'] and right_wrist:
                    prev_x, prev_y = prev_record['right_wrist']
                    curr_x, curr_y = right_wrist
                    disp = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    speed_norm = disp / forearm_length
                    speed_norm_buffer.append(speed_norm)
                else:
                    speed_norm_buffer.append(np.nan)
            else:
                speed_norm_buffer.append(np.nan)
            
            if detection_state == "waiting_for_standing":
                if len(speed_norm_buffer) >= stand_frames:
                    recent_speeds = list(speed_norm_buffer)[-stand_frames:]
                    valid_speeds = [s for s in recent_speeds if not np.isnan(s)]
                    if valid_speeds and (len(valid_speeds) / stand_frames) >= 0.8:
                        if all(s < v_thresh for s in valid_speeds):
                            detection_state = "detecting_throw"
                            standing_frames_rt.append(frame_count)
                            cooldown_frames_remaining = 0
                            ball_frames_in_session = []
                            release_detected_by_ratio = False
                            print(f"[Frame {frame_count:5d}] STANDING DETECTED")
            
            elif detection_state == "detecting_throw":
                if cooldown_frames_remaining > 0:
                    cooldown_frames_remaining -= 1
                    if cooldown_frames_remaining == 0:
                        detection_state = "waiting_for_standing"
                
                # Person lost fallback
                elif (not cache_hit and not pose_results.pose_landmarks) or (cache_hit and not right_wrist):
                     if not release_detected_by_ratio and len(ball_frames_in_session) > 0:
                        best_frame = ball_frames_in_session[-1]
                        for rec in all_records:
                            if rec['frame'] == best_frame['frame']:
                                rec['is_fallback_release'] = True
                                break
                        release_frames_rt.append(best_frame['frame'])
                        print(f"[Frame {best_frame['frame']:5d}] Fallback release")
                     detection_state = "waiting_for_standing"

            # Drawing (HUD)
            if right_wrist: cv2.circle(frame, right_wrist, 6, (255, 0, 0), -1)
            if right_elbow: cv2.circle(frame, right_elbow, 6, (0, 255, 255), -1)
            if closest_ball: cv2.line(frame, right_wrist, closest_ball, (255, 255, 0), 2)
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (360, 55), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames} | FPS: {fps}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # === [Fix 1] Write video frame ===
            out.write(frame)
            
            # === [Fix 2] Realtime preview ===
            if show_realtime:
                cv2.imshow(WIN, frame)
                if cv2.waitKey(1) & 0xFF == 27: # Press ESC to exit
                    break

            # === [Modified] Update logically calculated variables into frame_data ===
            # frame_data already contains 33 joint data from the loop above (nose_x, nose_y, ...)
            # Here we only need to supplement the logical variables calculated for this frame
            frame_data.update({
                'right_wrist': right_wrist, # Tuple used for logic judgment
                'right_elbow': right_elbow,
                'forearm_length': forearm_length,
                'closest_ball': closest_ball,
                'distance': distance,
                
                # Keep original key coordinate columns for later logic code to read directly
                'hand_x': right_wrist[0] if right_wrist else np.nan,
                'hand_y': right_wrist[1] if right_wrist else np.nan,
                'elbow_x': right_elbow[0] if right_elbow else np.nan,
                'elbow_y': right_elbow[1] if right_elbow else np.nan,
                'right_shoulder_x': right_shoulder[0] if right_shoulder else np.nan,
                'right_shoulder_y': right_shoulder[1] if right_shoulder else np.nan,
                'right_hip_x': right_hip[0] if right_hip else np.nan,
                'right_hip_y': right_hip[1] if right_hip else np.nan,
                'right_knee_x': right_knee[0] if right_knee else np.nan,
                'right_knee_y': right_knee[1] if right_knee else np.nan,
                
                'ball_x': closest_ball[0] if closest_ball else np.nan,
                'ball_y': closest_ball[1] if closest_ball else np.nan,
                'hip_y': hip_y if hip_y is not None else np.nan,
                'body_facing': body_facing,
                'throwing_hand': throwing_hand if 'throwing_hand' in locals() else 'right',
                'is_fallback_release': False,
                'shoulder_mid_x': shoulder_mid_x_pixel,
                'feet_dist': feet_dist_pixel
            })

            # Finally store this large dictionary containing "all joints + computed data" into the list
            all_records.append(frame_data)

            # Record ball frames
            if detection_state == "detecting_throw" and closest_ball and right_wrist and forearm_length:
                 ratio = distance / forearm_length
                 ball_frames_in_session.append({'frame': frame_count, 'ball_pos': closest_ball, 'ratio': ratio})
            
            # === Detect release ===
            if detection_state == "detecting_throw":
                diff_thresh = 0.06
                abs_thresh = 2.0
                if len(all_records) >= 2 and distance and forearm_length and right_wrist and closest_ball:
                    ratio = distance / forearm_length
                    prev_record = all_records[-2]
                    prev_dist = prev_record.get('distance')
                    prev_fore = prev_record.get('forearm_length')
                    
                    if prev_dist is not None and prev_fore is not None and prev_fore > 0:
                        prev_ratio = prev_dist / prev_fore
                        ratio_diff = ratio - prev_ratio
                        cond_forward_rt = True
                        if shoulder_mid_x_pixel is not None:
                            if body_facing == "right":
                                cond_forward_rt = closest_ball[0] > shoulder_mid_x_pixel
                            elif body_facing == "left":
                                cond_forward_rt = closest_ball[0] < shoulder_mid_x_pixel

                        if ratio_diff > diff_thresh and ratio > abs_thresh and cond_forward_rt:
                            if cooldown_frames_remaining == 0:
                                cooldown_frames_remaining = buffer_frames
                                release_detected_by_ratio = True
                                release_frames_rt.append(frame_count)
                                all_records[-1]["is_rt_release"] = True
                                print(f"[Frame {frame_count:5d}] Release detected, buffer started")

        state_log.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.pose.close()
        
        df = pd.DataFrame(all_records)
        print("\nDetection complete.")

        if df.empty:
            return df, []
        
        # === Cache saving ===
        if df_cache is None:
            print(f"[CACHE] Saving detection data to {cache_file_path}")
            df.to_csv(cache_file_path, index=False)

        # Interpolation & smoothing
        df["is_rt_release"] = False
        if release_frames_rt: df.loc[df["frame"].isin(release_frames_rt), "is_rt_release"] = True
        for col in ["ball_x", "ball_y", "distance"]:
            df[col] = df[col].interpolate(limit=3, limit_direction='both')
        df["forearm_length"] = df["forearm_length"].ffill().bfill()
        df["ratio"] = df["distance"] / df["forearm_length"]
        df["ratio_smooth"] = df["ratio"].rolling(window=3, min_periods=1).mean()
        df["ratio_diff"] = df["ratio_smooth"].diff()

        scale = df["forearm_length"].replace(0, np.nan)
        def calc_speed(x_col, y_col):
            if x_col not in df.columns or y_col not in df.columns: return pd.Series([0]*len(df))
            dx = df[x_col].diff(); dy = df[y_col].diff()
            return np.sqrt(dx**2 + dy**2) / scale

        s_wrist = calc_speed("hand_x", "hand_y")
        s_elbow = calc_speed("elbow_x", "elbow_y")
        s_shoulder = calc_speed("right_shoulder_x", "right_shoulder_y")
        s_hip = calc_speed("right_hip_x", "right_hip_y")
        s_knee = calc_speed("right_knee_x", "right_knee_y")

        df["speed_norm"] = s_wrist
        df["speed_norm_smooth"] = s_wrist.rolling(window=5, min_periods=1).mean()
        activity_stack = np.vstack([s_wrist.fillna(0).values, s_elbow.fillna(0).values, s_shoulder.fillna(0).values, s_hip.fillna(0).values, s_knee.fillna(0).values])
        df["body_activity"] = np.max(activity_stack, axis=0)
        df["upper_speed_smooth"] = pd.Series(df["body_activity"]).rolling(window=5, min_periods=1).median()
        
        if "feet_dist" in df: 
            df["feet_spread_ratio"] = df["feet_dist"] / scale
        else: 
            df["feet_spread_ratio"] = np.nan
        dx_hh = df["hand_x"] - df["right_hip_x"]
        dy_hh = df["hand_y"] - df["right_hip_y"]
        df["hand_hip_ratio"] = np.sqrt(dx_hh**2 + dy_hh**2) / scale

        # === Pairing logic ===
        standing_frames_rt = sorted(list(set(standing_frames_rt)))
        release_frames_rt = sorted(list(set(release_frames_rt)))
        final_standing, final_release = [], []
        max_duration_frames = 20 * fps 

        for r_frame in release_frames_rt:
            valid_standings = [s for s in standing_frames_rt if s < r_frame]
            if not valid_standings: continue
            best_s_frame = valid_standings[-1]
            if (r_frame - best_s_frame) > max_duration_frames: continue
            
            if final_standing and best_s_frame == final_standing[-1]:
                final_standing[-1] = best_s_frame
                final_release[-1] = r_frame 
            else:
                final_standing.append(best_s_frame)
                final_release.append(r_frame)
        
        standing_frames, release_frames = final_standing, final_release
        
        # ========== Detect Clip Start ==========
        print("\n[INFO] Detecting Clip Start (Strict Hand Reset + Direction Check)...")
        throw_start_frames = []
        
        cols_to_check = ["hand_hip_ratio", "speed_norm_smooth", "upper_speed_smooth"]
        for c in cols_to_check:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        for idx, (standing_frame, release_frame) in enumerate(zip(standing_frames, release_frames)):
            search_end = max(standing_frame, release_frame - int(0.5 * fps))
            search_start = standing_frame
            if search_start >= search_end:
                throw_start_frames.append(standing_frame)
                continue
            
            back_range = range(search_end, search_start, -1)
            t_hand_reset = None
            hand_speed_lim = 0.05  
            hand_hip_lim = 1     
            
            for t in back_range:
                is_stable = True
                for offset in range(3): 
                    check_t = t - offset
                    if check_t < search_start: is_stable = False; break
                    h_speed = df.loc[check_t, "speed_norm_smooth"]
                    h_dist = df.loc[check_t, "hand_hip_ratio"]
                    scale = df.loc[check_t, "forearm_length"]
                    hx = df.loc[check_t, "hand_x"]
                    rx = df.loc[check_t, "right_hip_x"]
                    facing = df.loc[check_t, "body_facing"]
                    
                    if pd.isna(h_speed) or pd.isna(h_dist): continue 
                    if h_speed > hand_speed_lim or h_dist > hand_hip_lim: is_stable = False; break
                    
                    allowance = 0.2 * scale if (pd.notna(scale) and scale > 0) else 20
                    is_behind = False
                    if facing == "right":
                        if hx < (rx - allowance): is_behind = True
                    elif facing == "left":
                        if hx > (rx + allowance): is_behind = True
                    if is_behind: is_stable = False; break
                
                if is_stable: t_hand_reset = t; break 
            
            best_start = standing_frame
            if t_hand_reset is not None:
                check_win_start = max(standing_frame, t_hand_reset - 15) 
                check_win_end = min(search_end, t_hand_reset + 5)
                local_segment = df.loc[check_win_start:check_win_end, "upper_speed_smooth"]
                if not local_segment.empty:
                    rev_segment = local_segment[::-1]
                    found_last_quiet = False
                    body_quiet_thresh = 0.06 
                    for t, val in rev_segment.items():
                        if val < body_quiet_thresh:
                            best_start = t; found_last_quiet = True; break
                    if not found_last_quiet: best_start = t_hand_reset
                else: best_start = t_hand_reset
            else:
                segment = df.loc[search_start:search_end, "upper_speed_smooth"]
                trig_idx = None
                for t in segment.index[:-3]:
                    if segment[t]>0.25 and segment[t+1]>0.25: trig_idx = t; break
                if trig_idx:
                    sub = df.loc[standing_frame:trig_idx].iloc[::-1]
                    for i, row in sub.iterrows():
                        if row["upper_speed_smooth"] < 0.08: best_start = i; break
                else: best_start = max(standing_frame, release_frame - int(1.5*fps))

            final_start = max(standing_frame, best_start - 10)
            throw_start_frames.append(final_start)
        
        # Plot (Skipping for brevity, keeping original logic effectively)
        frames_idx = df["frame"]
        hand_y = df["hand_y"]
        ball_y = df["ball_y"]
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(frames_idx, hand_y, label="Hand Y", color="blue")
        ax1.scatter(frames_idx, ball_y, label="Ball Y", color="green", s=10)
        for rf in release_frames: ax1.axvline(x=df.loc[rf, "frame"], color="red", linestyle="--")
        for tsf in throw_start_frames: ax1.axvline(x=df.loc[tsf, "frame"], color="purple", linestyle="--")
        plots_dir = Path(video_dir) / "plots" if video_dir is not None else Path(os.path.dirname(output_path)) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(plots_dir / "hand_ball_y_plot_marked.png"))
        plt.close()

        # =========================================================
        # === NEW FEATURE: Extract Clips (Start -> Release) ===
        # =========================================================
        if len(throw_start_frames) > 0:
            events = []
            if video_dir is not None:
                events_path = Path(video_dir) / "events.json"
            else:
                events_path = Path(os.path.dirname(output_path)) / "events.json"

            print(f"\n[CLIPS] Extracting {len(throw_start_frames)} clips...")
            
            # Re-open video for reading (to jump around)
            cap_clip = cv2.VideoCapture(video_path)
            
            # Output directory root (per-throw subfolders)
            from pathlib import Path
            if video_dir is not None:
                throws_root = Path(video_dir) / "throws"
            else:
                throws_root = Path(os.path.dirname(output_path)) / "throws"
            throws_root.mkdir(parents=True, exist_ok=True)

            
            # Buffer setting: frames after release to keep (e.g., 30 frames ~ 1 sec)
            post_release_buffer = 10 
            
            for i, (ts_idx, rel_idx) in enumerate(zip(throw_start_frames, release_frames)):
                # Get actual frame numbers from DataFrame
                start_f = int(df.loc[ts_idx, "frame"])
                end_f = int(df.loc[rel_idx, "frame"]) + post_release_buffer
                end_f = min(end_f, total_frames - 1)
                
                if start_f >= end_f:
                    throw_id = i + 1
                    # 这里假设你现在已经有 throw_dir 变量；如果你叫别的名字，就把 throw_dir 换成你的变量
                    events.append({
                        "throw_id": throw_id,
                        "start_frame": int(start_f),
                        "release_frame": int(df.loc[rel_idx, "frame"]),
                        "end_frame": int(end_f),
                        "status": "failed",
                        "reason": "start_frame >= end_frame"
                    })
                    continue
                                
                throw_id = i + 1
                throw_dir = throws_root / f"throw_{throw_id:03d}"
                throw_dir.mkdir(parents=True, exist_ok=True)

                clip_name = "clip.mp4"
                clip_path = str(throw_dir / clip_name)
                # --- record OK event (index for UI) ---
                throw_dir_rel = str(Path("throws") / f"throw_{throw_id:03d}")
                events.append({
                    "throw_id": throw_id,
                    "start_frame": int(start_f),
                    "release_frame": int(df.loc[rel_idx, "frame"]),
                    "end_frame": int(end_f),
                    "status": "ok",
                    "throw_dir": throw_dir_rel,
                    "clip_video": str(Path(throw_dir_rel) / "clip.mp4"),
                    "clip_csv": str(Path(throw_dir_rel) / "clip.csv")
                })
                
                # Clip writer setup
                clip_w = out_width  # Defined earlier (rotated)
                clip_h = out_height # Defined earlier (rotated)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_clip = cv2.VideoWriter(clip_path, fourcc, fps, (clip_w, clip_h))
                
                print(f"  -> Saving {clip_name} ({end_f - start_f} frames)")
                
                # === [NEW] Save corresponding CSV data segment ===
                # Filter out DataFrame data for the current segment
                # Prefer cached full-pose dataframe for exporting clip CSV
                source_df = df_cache if df_cache is not None else df
                clip_df = source_df[(source_df['frame'] >= start_f) & (source_df['frame'] <= end_f)].copy()

                
                # Define body joints to save (all body parts, exclude face only)
                body_joints = [
                    # Main joints (12)
                    'left_shoulder', 'right_shoulder',
                    'left_elbow', 'right_elbow', 
                    'left_wrist', 'right_wrist',
                    'left_hip', 'right_hip',
                    'left_knee', 'right_knee',
                    'left_ankle', 'right_ankle',
                    # Hand details (6)
                    'left_pinky', 'right_pinky',
                    'left_index', 'right_index',
                    'left_thumb', 'right_thumb',
                    # Foot details (4)
                    'left_heel', 'right_heel',
                    'left_foot_index', 'right_foot_index'
                ]
                
                # Build columns to save
                columns_to_save = ['frame', 'fps']
                
                # Add body joint coordinates (x, y only, no visibility)
                for joint in body_joints:
                    columns_to_save.extend([f'{joint}_x', f'{joint}_y'])
                
                # Add computed/analysis columns
                computed_cols = [
                    'hand_x', 'hand_y', 'elbow_x', 'elbow_y',
                    'right_shoulder_x', 'right_shoulder_y',
                    'right_hip_x', 'right_hip_y', 
                    'right_knee_x', 'right_knee_y',
                    'ball_x', 'ball_y',
                    'forearm_length', 'distance', 
                    'hip_y', 'body_facing', 
                    'is_fallback_release', 'shoulder_mid_x', 'feet_dist'
                ]
                columns_to_save.extend(computed_cols)
                
                # Filter to only existing columns
                columns_to_save = [col for col in columns_to_save if col in clip_df.columns]
                # De-duplicate columns to avoid repeated headers in CSV (which later become .1, .2 when read)
                seen = set()
                dedup_cols = []
                for c in columns_to_save:
                    if c not in seen:
                        dedup_cols.append(c)
                        seen.add(c)
                columns_to_save = dedup_cols

                
                # Generate CSV filename (same name as video, but with .csv extension)
                csv_name = "clip.csv"
                csv_path = str(throw_dir / csv_name)

                
                # Save filtered DataFrame
                clip_df[columns_to_save].to_csv(csv_path, index=False)
                print(f"     -> Also saved data to {csv_name} ({len(columns_to_save)} columns)")
                
                # Jump to start frame
                cap_clip.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                
                for f_num in range(start_f, end_f + 1):
                    ret_c, frame_c = cap_clip.read()
                    if not ret_c: break
                    
                    # Apply the same rotation as main detection!
                    if rotate_code is not None:
                        frame_c = cv2.rotate(frame_c, rotate_code)
                    
                    out_clip.write(frame_c)
                
                out_clip.release()
            
            cap_clip.release()
            # --- write events.json (video-level index) ---
            with open(events_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "video_path": str(video_path),
                        "fps": int(fps),
                        "total_frames": int(total_frames),
                        "num_throws": int(len(events)),
                        "events": events
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            print(f"[EVENTS] Saved events to {events_path}")

            print(f"[CLIPS] All clips saved to {throws_root}")

        return df, release_frames

def run_once(
    video_path: str,
    out_dir: str,
    reference_csv: str,
    thresholds_json: str,
    *,
    enable_annotation: bool = True,
    enable_dtw: bool = True,
    show_realtime: bool = False,
) -> Dict[str, Any]:
    """
    Run the full pipeline for a single input video.

    Returns a dict containing key output paths and summary info for UI.
    """
    video_path_p = Path(video_path)
    out_dir_p = Path(out_dir)

    # 1) Prepare output folders
    out_dir_p.mkdir(parents=True, exist_ok=True)
    (out_dir_p / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir_p / "intermediate").mkdir(parents=True, exist_ok=True)
    (out_dir_p / "throws").mkdir(parents=True, exist_ok=True)

    # 2) Run detection & FSM segmentation
    # NOTE: call your existing detect_video() here.
    # IMPORTANT: Do NOT chdir. Make detect_video write into out_dir_p.
    events_path = out_dir_p / "events.json"
    detector = TennisDetector(model_path="yolov8m.pt", confidence=0.15, iou=0.45)

    # If your current detect_video signature is different, keep it as-is,
    # but pass show_realtime=show_realtime and make sure all outputs go to out_dir_p.
    detector.detect_video(
    video_path=str(video_path_p),
    output_path=str(out_dir_p / "output_detection_with_pose.mp4"),
    show_realtime=False,
    use_tracker=False,
    img_size=1280,
    video_dir=str(out_dir_p),)


    if not events_path.exists():
        raise FileNotFoundError(f"events.json not found at {events_path}")

    # 3) Optional: annotation from events
    annotated_video_path: Optional[str] = None
    if enable_annotation:
        from annotate_basic import annotate_from_events
        annotated_video_path = annotate_from_events(out_dir_p)

    # 4) Optional: DTW scoring for each throw
    summary_scores_path: Optional[str] = None
    if enable_dtw:
        summary_scores_path = _run_dtw_and_write_summary(
            out_dir_p=out_dir_p,
            reference_csv=Path(reference_csv),
            thresholds_json=Path(thresholds_json),
        )

    # 5) Collect known plot outputs (optional)
    plots_dir = out_dir_p / "plots"
    plots: List[str] = []
    if plots_dir.exists():
        plots = [str(p) for p in sorted(plots_dir.glob("*.png"))]

    # 6) Return to UI
    return {
        "video": str(video_path_p),
        "out_dir": str(out_dir_p),
        "events_json": str(events_path),
        "annotated_video": annotated_video_path,
        "summary_scores": summary_scores_path,
        "plots": plots,
    }

def run_single_throw_dtw(
    sample_csv: str,
    reference_csv: str,
    thresholds: dict,
    out_json: str,
    *,
    save_frame_csv: bool = False,
):
    from fixed_dtw_comparison import FixedMotionDTW
    import json
    from pathlib import Path

    comparator = FixedMotionDTW(reference_csv, reference_handedness="auto")
    scores, frame_dists, meta = comparator.compare(
        student_csv=sample_csv,
        handedness="auto",
        save_frame_csv=save_frame_csv
    )

    features_out = {}
    for k in comparator.FEATURE_NAMES:
        if k not in scores:
            continue

        v = float(scores[k])
        cfg = thresholds.get(k)

        if cfg is None:
            level = "unknown"
            warn = bad = None
        else:
            warn = float(cfg.get("warn", 0.0))
            bad = float(cfg.get("bad", 1e9))
            level = "ok" if v < warn else "warn" if v < bad else "bad"

        features_out[k] = {
            "value": v,
            "level": level,
            "warn": warn,
            "bad": bad,
        }

    score_obj = {
        "meta": meta,
        "dtw": {
            "features": features_out,
            "overall_matching_score": scores.get("overall_matching_score"),
        }
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(score_obj, f, indent=2)

    return score_obj




def _run_dtw_and_write_summary(
    out_dir_p: Path,
    reference_csv: Path,
    thresholds_json: Path,
) -> str:
    """
    Reads events.json, runs DTW per ok throw, writes:
    - throws/throw_xxx/score.json (existing behavior)
    - outputs/scores_summary.json (new; for UI table)
    Returns path to the summary json.
    """
    events_path = out_dir_p / "events.json"
    with events_path.open("r", encoding="utf-8") as f:
        events_obj = json.load(f)

    thresholds = {}
    if thresholds_json.exists():
        with thresholds_json.open("r", encoding="utf-8") as f:
            thresholds = json.load(f)

    throws = events_obj.get("events", [])
    summary_rows = []

    for t in throws:
        if t.get("status") != "ok":
            continue

        throw_id = t.get("throw_id")
        ev = t
        throw_dir = out_dir_p / ev["throw_dir"]
        clip_csv = out_dir_p / ev["clip_csv"]
        score_path = throw_dir / "score.json"
        if not clip_csv.exists():
            # record as missing but do not crash the whole job
            summary_rows.append({
                "throw_id": throw_id,
                "status": "failed",
                "reason": f"missing clip_csv: {clip_csv}",
            })
            continue

        # ---- Call your existing DTW compare logic here ----
        # Example:
        # dtw = FixedMotionDTW(reference_csv=str(reference_csv), thresholds=thresholds)
        # score_obj = dtw.compare(sample_csv=str(clip_csv), out_json=str(score_path))
        #
        # IMPORTANT: keep the original logic, just ensure paths are absolute.

        score_obj = run_single_throw_dtw(
            sample_csv=str(clip_csv),
            reference_csv=str(reference_csv),
            thresholds=thresholds,
            out_json=str(score_path),
        )

        # Build UI-friendly row
        row = {
            "throw_id": throw_id,
            "start_frame": t.get("start_frame"),
            "release_frame": t.get("release_frame"),
            "end_frame": t.get("end_frame"),
            "score_json": str(score_path),
            "status": "ok",
        }
        # if score_obj contains an overall score, attach it
        if isinstance(score_obj, dict):
            row["overall_matching_score"] = (score_obj.get("dtw") or {}).get("overall_matching_score")
        summary_rows.append(row)

    outputs_dir = out_dir_p / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = outputs_dir / "scores_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": summary_rows}, f, ensure_ascii=False, indent=2)

    return str(summary_path)


if __name__ == "__main__":
    import json

    video = "samples/your_test_video.mp4"
    out_dir = "runs/local_test"

    result = run_once(
        video_path=video,
        out_dir=out_dir,
        reference_csv="reference/model.csv",
        thresholds_json="config/dtw_thresholds.json",
        enable_annotation=True,
        enable_dtw=True,
    )

    print(json.dumps(result, indent=2))
