import cv2

# ---------- config ----------
in_path = "input.mp4"
out_path = "output_with_frame_idx.mp4"

# 左上角文字位置与样式
pos = (20, 40)                 # (x, y)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
thickness = 2
color = (255, 255, 255)        # 白色(BGR)
# ---------------------------

cap = cv2.VideoCapture(in_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {in_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 也可改成 "avc1" / "XVID" 视系统而定
writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
if not writer.isOpened():
    raise RuntimeError(f"Cannot open writer: {out_path}")

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    text = f"Frame: {frame_idx}"

    # 可选：给文字加黑色描边，增强可读性
    cv2.putText(frame, text, pos, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
print("Done:", out_path)