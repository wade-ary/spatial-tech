from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional

import pyautogui


# Prevent pyautogui from adding an implicit pause after each action.
pyautogui.PAUSE = 0


@dataclass
class PredictionEvent:
    timestamp: float
    label: str
    confidence: float


class GestureRules:
    """
    Stateful gesture policy:
    - Maintains a rolling 3-second history of confident predictions.
    - Moves mouse only on confident "mouse" frames.
    - Double-clicks when "thumb" has >=90% majority in the last 1 second.
    """

    def __init__(
        self,
        *,
        prediction_conf_threshold: float = 0.9,
        history_window_s: float = 3.0,
        click_window_s: float = 1.0,
        click_majority_threshold: float = 0.9,
        click_cooldown_s: float = 0.4,
    ) -> None:
        self.prediction_conf_threshold = prediction_conf_threshold
        self.history_window_s = history_window_s
        self.click_window_s = click_window_s
        self.click_majority_threshold = click_majority_threshold
        self.click_cooldown_s = click_cooldown_s

        self._history: Deque[PredictionEvent] = deque()
        self._last_click_ts: float = 0.0

        self.screen_w, self.screen_h = pyautogui.size()

    def process_frame(
        self,
        *,
        predicted_label: str,
        confidence: float,
        index_tip_x_norm: Optional[float] = None,
        index_tip_y_norm: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> dict:
        """
        Process one frame's model output and execute actions.
        Returns action flags for easy debugging/UI overlays.
        """
        now = timestamp if timestamp is not None else time.time()
        frame_is_confident = confidence >= self.prediction_conf_threshold

        if frame_is_confident:
            self._history.append(
                PredictionEvent(
                    timestamp=now,
                    label=predicted_label,
                    confidence=confidence,
                )
            )
        self._trim_history(now)

        mouse_moved = False
        click_fired = False

        if (
            frame_is_confident
            and predicted_label == "mouse"
            and index_tip_x_norm is not None
            and index_tip_y_norm is not None
        ):
            screen_x, screen_y = self._normalized_to_screen(
                index_tip_x_norm, index_tip_y_norm
            )
            pyautogui.moveTo(screen_x, screen_y)
            mouse_moved = True

        if self._click_majority_reached(now) and self._click_allowed(now):
            pyautogui.doubleClick()
            self._last_click_ts = now
            click_fired = True

        return {
            "mouse_moved": mouse_moved,
            "click_fired": click_fired,
            "history_size": len(self._history),
        }

    def _normalized_to_screen(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        x_norm = min(max(x_norm, 0.0), 1.0)
        y_norm = min(max(y_norm, 0.0), 1.0)
        screen_x = int(x_norm * self.screen_w)
        screen_y = int(y_norm * self.screen_h)
        return screen_x, screen_y

    def _trim_history(self, now: float) -> None:
        cutoff = now - self.history_window_s
        while self._history and self._history[0].timestamp < cutoff:
            self._history.popleft()

    def _click_majority_reached(self, now: float) -> bool:
        click_cutoff = now - self.click_window_s
        recent = [event for event in self._history if event.timestamp >= click_cutoff]
        if not recent:
            return False

        counts = Counter(event.label for event in recent)
        click_ratio = counts.get("thumb", 0) / len(recent)
        return click_ratio >= self.click_majority_threshold

    def _click_allowed(self, now: float) -> bool:
        return (now - self._last_click_ts) >= self.click_cooldown_s
