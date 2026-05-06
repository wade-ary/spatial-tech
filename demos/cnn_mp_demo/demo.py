from pathlib import Path
import time

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from PIL import Image
from torchvision import transforms

from rules import GestureRules


MODEL_TYPE = "cnn"
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
HAND_LANDMARKER_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
PREDICTION_CONFIDENCE_THRESHOLD = 0.9
ACTIVE_X_MIN = 0.15
ACTIVE_X_MAX = 0.85
ACTIVE_Y_MIN = 0.10
ACTIVE_Y_MAX = 0.70
CLICK_COOLDOWN_S = 3.0


class BasicCNN(nn.Module):
    def __init__(self, n_classes: int, img_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (img_size // 8) * (img_size // 8), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def resolve_model_paths():
    model_candidates = [
        PROJECT_ROOT / "models" / "basic_cnn" / "basic_cnn.pth",
        PROJECT_ROOT / "basic_cnn.pth",
    ]
    meta_candidates = [
        PROJECT_ROOT / "models" / "basic_cnn" / "basic_cnn_meta.pth",
        PROJECT_ROOT / "basic_cnn_meta.pth",
    ]

    model_path = next((p.resolve() for p in model_candidates if p.exists()), None)
    meta_path = next((p.resolve() for p in meta_candidates if p.exists()), None)

    if model_path is None:
        raise FileNotFoundError("Model weights not found.")
    if meta_path is None:
        raise FileNotFoundError("Model metadata not found.")

    return model_path, meta_path


def load_cnn(device):
    model_path, meta_path = resolve_model_paths()
    metadata = torch.load(meta_path, map_location="cpu")
    class_names = metadata["class_names"]
    img_size = int(metadata.get("img_size", 128))
    num_classes = int(metadata["num_classes"])

    model = BasicCNN(n_classes=num_classes, img_size=img_size).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return model, class_names, transform


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def hand_bbox_int_pixels(landmarks, h: int, w: int, pad: int = 10):
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


def classify_handedness(label_obj) -> str:
    try:
        return label_obj[0].category_name.lower()
    except (TypeError, IndexError, AttributeError):
        return "unknown"


def main():
    if MODEL_TYPE.lower() != "cnn":
        raise ValueError(f"Unsupported MODEL_TYPE='{MODEL_TYPE}'. Use 'cnn'.")

    device = get_device()
    model, class_names, transform = load_cnn(device)
    softmax = nn.Softmax(dim=1)
    rules = GestureRules(
        prediction_conf_threshold=PREDICTION_CONFIDENCE_THRESHOLD,
        click_window_s=1.5,
        click_majority_threshold=0.9,
        smoothing_alpha=0.25,
        click_cooldown_s=CLICK_COOLDOWN_S,
        active_x_min=ACTIVE_X_MIN,
        active_x_max=ACTIVE_X_MAX,
        active_y_min=ACTIVE_Y_MIN,
        active_y_max=ACTIVE_Y_MAX,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera 0.")

    if not HAND_LANDMARKER_PATH.exists():
        raise FileNotFoundError("MediaPipe hand model not found.")

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH.resolve())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    print("Press 'q' to quit.")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print("Right hand index finger controls pointer (smoothed alpha=0.25).")
    print(f"Active control box: x={ACTIVE_X_MIN:.2f}-{ACTIVE_X_MAX:.2f}, y={ACTIVE_Y_MIN:.2f}-{ACTIVE_Y_MAX:.2f}")
    print(f"Click cooldown: {CLICK_COOLDOWN_S:.1f}s after either click action.")
    print("Left hand CNN gestures: palm -> right click, thumb -> left click.")

    try:
        with vision.HandLandmarker.create_from_options(options) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(time.time() * 1000)
                result = hands.detect_for_video(mp_image, timestamp_ms)

                pointer_text = "right hand: not detected"
                left_text = "left hand: no gesture"
                action_text = "action: none"

                if result.hand_landmarks:
                    handedness = result.handedness or []
                    for i, hand_landmarks in enumerate(result.hand_landmarks):
                        hand_side = classify_handedness(handedness[i]) if i < len(handedness) else "unknown"

                        if hand_side == "right":
                            index_tip = hand_landmarks[8]
                            smooth_x, smooth_y = rules.update_pointer(
                                index_tip_x_norm=index_tip.x,
                                index_tip_y_norm=index_tip.y,
                            )
                            tip_x = int(index_tip.x * w)
                            tip_y = int(index_tip.y * h)
                            tip_x = max(0, min(w - 1, tip_x))
                            tip_y = max(0, min(h - 1, tip_y))
                            cv2.circle(frame, (tip_x, tip_y), 7, (255, 0, 255), -1)
                            pointer_text = f"right pointer: ({smooth_x}, {smooth_y})"

                        elif hand_side == "left":
                            x1, y1, x2, y2 = hand_bbox_int_pixels(hand_landmarks, h, w, pad=14)
                            if x2 > x1 and y2 > y1:
                                hand_crop = frame_rgb[y1 : y2 + 1, x1 : x2 + 1]
                                pil_img = Image.fromarray(hand_crop)
                                x = transform(pil_img).unsqueeze(0).to(device)

                                with torch.no_grad():
                                    logits = model(x)
                                    probs = softmax(logits)
                                    conf, pred_idx = torch.max(probs, dim=1)

                                pred_idx = int(pred_idx.item())
                                conf = float(conf.item())
                                pred_label = class_names[pred_idx]
                                action = rules.process_left_prediction(
                                    predicted_label=pred_label,
                                    confidence=conf,
                                )

                                left_text = f"left gesture: {pred_label} p={conf:.3f}"
                                action_text = f"action: {action['action']}"

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(
                                    frame,
                                    left_text,
                                    (x1, max(30, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )

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
                    left_text,
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    action_text,
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Live Prediction (CNN + MediaPipe Rules)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
