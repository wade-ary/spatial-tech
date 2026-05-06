"""
Single-label image collector using Mac webcam.

Keys:
  c  toggle continuous click collection on/off
  q  quit
"""

from __future__ import annotations

import time
from urllib.request import urlretrieve
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

_REPO_ROOT = Path(__file__).resolve().parent
_MODEL_PATH = _REPO_ROOT / "models" / "hand_landmarker.task"
_CLICK_IMAGES_DIR = _REPO_ROOT / "data" / "images" / "click"
_JPEG_QUALITY = 92
_TARGET_FPS = 50.0
_CAPTURE_INTERVAL_S = 1.0 / _TARGET_FPS
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def put_status(frame, text: str) -> None:
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def save_click_image(frame, seq: int) -> Path:
    _CLICK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    t_ms = int(time.time() * 1000)
    out_path = _CLICK_IMAGES_DIR / f"{seq:06d}_{t_ms}.jpg"
    ok = cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")
    return out_path


def ensure_model_exists() -> None:
    if _MODEL_PATH.exists():
        return
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {_MODEL_PATH} ...")
    urlretrieve(_MODEL_URL, _MODEL_PATH)


def hand_bbox_int_pixels(landmarks, h: int, w: int, pad: int = 10) -> tuple[int, int, int, int]:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x = int(min(xs) * w)
    max_x = int(max(xs) * w)
    min_y = int(min(ys) * h)
    max_y = int(max(ys) * h)
    x1 = max(0, min_x - pad)
    y1 = max(0, min_y - pad)
    x2 = min(w - 1, max_x + pad)
    y2 = min(h - 1, max_y + pad)
    return x1, y1, x2, y2


def main() -> None:
    ensure_model_exists()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open Mac webcam (camera index 0).")

    print("Camera started. Press 'c' to toggle collecting at 50 fps, 'q' to quit.")
    sample_seq = 1
    last_saved = "none"
    collecting = False
    next_capture_at = 0.0

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    with vision.HandLandmarker.create_from_options(options) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            result = hands.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                x1, y1, x2, y2 = hand_bbox_int_pixels(hand_landmarks, h, w)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                now = time.perf_counter()
                if collecting and now >= next_capture_at:
                    crop = frame[y1 : y2 + 1, x1 : x2 + 1]
                    if crop.size > 0:
                        out_path = save_click_image(crop, sample_seq)
                        last_saved = out_path.name
                        sample_seq += 1
                    next_capture_at = now + _CAPTURE_INTERVAL_S

            status = "ON" if collecting else "OFF"
            put_status(frame, f"Collecting: {status} | target: 50 fps | saved: {sample_seq - 1}")
            cv2.putText(
                frame,
                f"Last: {last_saved}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press c to start/stop, q to quit",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Click data collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                collecting = not collecting
                if collecting:
                    next_capture_at = time.perf_counter()
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
