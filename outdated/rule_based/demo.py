from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from rules import GestureRules


BASE_DIR = Path(__file__).resolve().parent
HAND_LANDMARKER_PATH = BASE_DIR / "hand_landmarker.task"


def classify_handedness(label_obj) -> str:
    try:
        return label_obj[0].category_name.lower()
    except (TypeError, IndexError, AttributeError):
        return "unknown"


def draw_landmark_point(frame, x_norm: float, y_norm: float, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    x = max(0, min(w - 1, int(x_norm * w)))
    y = max(0, min(h - 1, int(y_norm * h)))
    cv2.circle(frame, (x, y), 7, color, -1)


def main() -> None:
    rules = GestureRules(
        smoothing_alpha=0.25,
        pinch_distance_threshold=0.045,
        pinch_hold_s=0.2,
        click_cooldown_s=1.0,
        active_x_min=0.15,
        active_x_max=0.85,
        active_y_min=0.10,
        active_y_max=0.70,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera 0.")

    if not HAND_LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"MediaPipe hand model not found at {HAND_LANDMARKER_PATH}.")

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH.resolve())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    print("Press 'q' to quit.")
    print("Left index finger controls pointer.")
    print("Right thumb-index pinch (<threshold) held for 0.2s triggers one left click.")
    print("Right index-middle close activates scroll mode (up/down).")
    print("Cooldown: 1.0s between clicks.")

    try:
        with vision.HandLandmarker.create_from_options(options) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(time.time() * 1000)
                result = hands.detect_for_video(mp_image, timestamp_ms)

                pointer_text = "left hand: not detected"
                pinch_text = "right hand: not detected"
                scroll_text = "scroll: inactive"
                action_text = "action: none"

                if result.hand_landmarks:
                    handedness = result.handedness or []
                    for i, hand_landmarks in enumerate(result.hand_landmarks):
                        hand_side = classify_handedness(handedness[i]) if i < len(handedness) else "unknown"

                        if hand_side == "left":
                            index_tip = hand_landmarks[8]
                            smooth_x, smooth_y = rules.update_pointer(
                                index_tip_x_norm=index_tip.x,
                                index_tip_y_norm=index_tip.y,
                            )
                            draw_landmark_point(frame, index_tip.x, index_tip.y, (255, 0, 255))
                            pointer_text = f"left pointer: ({smooth_x}, {smooth_y})"

                        elif hand_side == "right":
                            pinch_status = rules.process_right_pinch(
                                right_hand_landmarks=hand_landmarks,
                            )
                            scroll_status = rules.process_scroll(hand_landmarks=hand_landmarks)
                            draw_landmark_point(frame, hand_landmarks[4].x, hand_landmarks[4].y, (0, 255, 255))
                            draw_landmark_point(frame, hand_landmarks[8].x, hand_landmarks[8].y, (0, 255, 0))
                            draw_landmark_point(frame, hand_landmarks[12].x, hand_landmarks[12].y, (255, 255, 0))
                            pinch_text = (
                                f"right pinch: d={pinch_status['pinch_distance']:.3f} "
                                f"held={pinch_status['held_for_s']:.2f}s "
                                f"cooldown={pinch_status['cooldown_remaining_s']:.2f}s"
                            )
                            scroll_text = (
                                f"scroll: active={scroll_status['scroll_active']} "
                                f"d={scroll_status['fingers_distance']:.3f} "
                                f"move={scroll_status['action']}"
                            )
                            action = "none"
                            if pinch_status["action"] != "none":
                                action = pinch_status["action"]
                            elif scroll_status["action"] != "none":
                                action = scroll_status["action"]
                            action_text = f"action: {action}"

                cv2.putText(
                    frame,
                    pointer_text,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    pinch_text,
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    scroll_text,
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    action_text,
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Rule-Based MediaPipe Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
