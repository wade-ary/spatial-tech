from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional

import pyautogui


pyautogui.PAUSE = 0


@dataclass
class PredictionEvent:
    timestamp: float
    label: str


class GestureRules:
    """
    Rule set for mixed MediaPipe + CNN control:
    - Right-hand index fingertip controls mouse pointer with exponential smoothing.
    - Left-hand CNN predictions trigger clicks only when one class reaches >=90%
      majority in the latest 1.5 seconds.
    - Only "palm" and "thumb" are considered; other labels are ignored.
    """

    def __init__(
        self,
        *,
        prediction_conf_threshold: float = 0.9,
        click_window_s: float = 1.5,
        click_majority_threshold: float = 0.9,
        smoothing_alpha: float = 0.25,
        click_cooldown_s: float = 3.0,
        active_x_min: float = 0.15,
        active_x_max: float = 0.85,
        active_y_min: float = 0.10,
        active_y_max: float = 0.70,
    ) -> None:
        self.prediction_conf_threshold = prediction_conf_threshold
        self.click_window_s = click_window_s
        self.click_majority_threshold = click_majority_threshold
        self.smoothing_alpha = smoothing_alpha
        self.click_cooldown_s = click_cooldown_s
        self.active_x_min = active_x_min
        self.active_x_max = active_x_max
        self.active_y_min = active_y_min
        self.active_y_max = active_y_max

        self.screen_w, self.screen_h = pyautogui.size()
        self._smoothed_mouse: Optional[tuple[float, float]] = None
        self._history: Deque[PredictionEvent] = deque()
        self._last_click_ts: float = 0.0

    def update_pointer(
        self,
        *,
        index_tip_x_norm: float,
        index_tip_y_norm: float,
    ) -> tuple[int, int]:
        target_x, target_y = self._normalized_to_screen(index_tip_x_norm, index_tip_y_norm)

        if self._smoothed_mouse is None:
            smoothed_x, smoothed_y = float(target_x), float(target_y)
        else:
            old_x, old_y = self._smoothed_mouse
            alpha = self.smoothing_alpha
            smoothed_x = (1.0 - alpha) * old_x + alpha * target_x
            smoothed_y = (1.0 - alpha) * old_y + alpha * target_y

        self._smoothed_mouse = (smoothed_x, smoothed_y)
        pyautogui.moveTo(int(smoothed_x), int(smoothed_y))
        return int(smoothed_x), int(smoothed_y)

    def process_left_prediction(
        self,
        *,
        predicted_label: str,
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> dict:
        now = timestamp if timestamp is not None else time.time()
        normalized_label = predicted_label.strip().lower().replace("_", " ")

        if confidence >= self.prediction_conf_threshold:
            if normalized_label in {"palm", "thumb", "thumb click"}:
                canonical = "thumb" if "thumb" in normalized_label else "palm"
                self._history.append(PredictionEvent(timestamp=now, label=canonical))

        self._trim_history(now)

        action = "none"
        if self._click_allowed(now):
            majority_label = self._majority_action()
            if majority_label == "palm":
                pyautogui.rightClick()
                self._last_click_ts = now
                action = "right_click"
            elif majority_label == "thumb":
                pyautogui.click()
                self._last_click_ts = now
                action = "left_click"

        return {
            "action": action,
            "history_size": len(self._history),
            "majority": self._majority_action(),
        }

    def _majority_action(self) -> Optional[str]:
        if not self._history:
            return None

        counts = Counter(event.label for event in self._history)
        label, count = counts.most_common(1)[0]
        ratio = count / len(self._history)
        if ratio >= self.click_majority_threshold:
            return label
        return None

    def _trim_history(self, now: float) -> None:
        cutoff = now - self.click_window_s
        while self._history and self._history[0].timestamp < cutoff:
            self._history.popleft()

    def _click_allowed(self, now: float) -> bool:
        return (now - self._last_click_ts) >= self.click_cooldown_s

    def _normalized_to_screen(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        x_norm = self._remap_from_active_region(
            value=x_norm,
            region_min=self.active_x_min,
            region_max=self.active_x_max,
        )
        y_norm = self._remap_from_active_region(
            value=y_norm,
            region_min=self.active_y_min,
            region_max=self.active_y_max,
        )
        screen_x = int(x_norm * self.screen_w)
        screen_y = int(y_norm * self.screen_h)
        return screen_x, screen_y

    def _remap_from_active_region(
        self,
        *,
        value: float,
        region_min: float,
        region_max: float,
    ) -> float:
        if region_max <= region_min:
            return min(max(value, 0.0), 1.0)

        clamped = min(max(value, region_min), region_max)
        normalized = (clamped - region_min) / (region_max - region_min)
        return min(max(normalized, 0.0), 1.0)
