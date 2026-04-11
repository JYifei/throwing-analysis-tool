import cv2
from pathlib import Path


def mirror_video_left_right(input_path: str, output_path: str) -> None:
    """
    Mirror a video horizontally (left-right flip).

    Args:
        input_path: Path to the input video.
        output_path: Path to the output mirrored video.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input video not found: {input_file}")

    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Try mp4 first
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {output_file}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            mirrored = cv2.flip(frame, 1)  # 1 = horizontal flip
            writer.write(mirrored)
    finally:
        cap.release()
        writer.release()


if __name__ == "__main__":
    input_video = r"G:\Throwing_Project_v3\input.mp4"
    output_video = r"G:\Throwing_Project_v3\input_mirrored.mp4"

    mirror_video_left_right(input_video, output_video)
    print(f"Done: {output_video}")