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
HAND_LANDMARKER_PATH = Path("../models/hand_landmarker.task")
PREDICTION_CONFIDENCE_THRESHOLD = 0.9


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
        Path("../models/basic_cnn/basic_cnn.pth"),
        Path("../basic_cnn.pth"),
    ]
    meta_candidates = [
        Path("../models/basic_cnn/basic_cnn_meta.pth"),
        Path("../basic_cnn_meta.pth"),
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


def main():
    if MODEL_TYPE.lower() != "cnn":
        raise ValueError(f"Unsupported MODEL_TYPE='{MODEL_TYPE}'. Use 'cnn'.")

    device = get_device()
    model, class_names, transform = load_cnn(device)
    softmax = nn.Softmax(dim=1)
    rules = GestureRules(prediction_conf_threshold=PREDICTION_CONFIDENCE_THRESHOLD)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera 0.")

    if not HAND_LANDMARKER_PATH.exists():
        raise FileNotFoundError("MediaPipe hand model not found.")

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH.resolve())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    print("Press 'q' to quit.")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")

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

                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]
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
                        is_confident = conf >= PREDICTION_CONFIDENCE_THRESHOLD

                        action = rules.process_frame(
                            predicted_label=pred_label,
                            confidence=conf,
                            index_tip_x_norm=hand_landmarks[8].x,
                            index_tip_y_norm=hand_landmarks[8].y,
                        )

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = (
                            f"{pred_label} | p={conf:.3f}"
                            if is_confident
                            else f"uncertain | p={conf:.3f}"
                        )
                        cv2.putText(
                            frame,
                            text,
                            (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                        index_tip = hand_landmarks[8]
                        tip_x = int(index_tip.x * w)
                        tip_y = int(index_tip.y * h)
                        tip_x = max(0, min(w - 1, tip_x))
                        tip_y = max(0, min(h - 1, tip_y))
                        cv2.circle(frame, (tip_x, tip_y), 7, (255, 0, 255), -1)

                        status = f"move={action['mouse_moved']} click={action['click_fired']} hist={action['history_size']}"
                        cv2.putText(
                            frame,
                            status,
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                else:
                    cv2.putText(
                        frame,
                        "No hand detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Live Prediction (CNN + Rules)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
