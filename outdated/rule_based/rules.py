from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import pyautogui


pyautogui.PAUSE = 0


@dataclass
class PinchState:
    currently_pinched: bool = False
    pinch_start_ts: Optional[float] = None
    click_sent_for_current_pinch: bool = False
    last_click_ts: float = 0.0


@dataclass
class ScrollState:
    scroll_active: bool = False
    anchor_y: Optional[float] = None


class GestureRules:
    """
    Simple hand rules:
    - Left index fingertip controls mouse pointer.
    - Right pinch (thumb tip to index tip) triggers one click after hold time.
    - Clicks are rate-limited with cooldown.
    """

    def __init__(
        self,
        *,
        smoothing_alpha: float = 0.25,
        pinch_distance_threshold: float = 0.045,
        pinch_hold_s: float = 0.5,
        click_cooldown_s: float = 1.0,
        scroll_finger_close_threshold: float = 0.04,
        scroll_min_delta_y: float = 0.25,
        scroll_step: int = 120,
        active_x_min: float = 0.15,
        active_x_max: float = 0.85,
        active_y_min: float = 0.10,
        active_y_max: float = 0.70,
    ) -> None:
        self.smoothing_alpha = smoothing_alpha
        self.pinch_distance_threshold = pinch_distance_threshold
        self.pinch_hold_s = pinch_hold_s
        self.click_cooldown_s = click_cooldown_s
        self.scroll_finger_close_threshold = scroll_finger_close_threshold
        self.scroll_min_delta_y = scroll_min_delta_y
        self.scroll_step = scroll_step
        self.active_x_min = active_x_min
        self.active_x_max = active_x_max
        self.active_y_min = active_y_min
        self.active_y_max = active_y_max

        self.screen_w, self.screen_h = pyautogui.size()
        self._smoothed_mouse: Optional[tuple[float, float]] = None
        self.pinch_state = PinchState()
        self.scroll_state = ScrollState()

    def update_pointer(self, *, index_tip_x_norm: float, index_tip_y_norm: float) -> tuple[int, int]:
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

    def process_right_pinch(
        self,
        *,
        right_hand_landmarks: Sequence,
        timestamp: Optional[float] = None,
    ) -> dict:
        now = timestamp if timestamp is not None else time.time()
        thumb_tip = right_hand_landmarks[4]
        index_tip = right_hand_landmarks[8]

        pinch_distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        now_pinched = pinch_distance < self.pinch_distance_threshold
        state = self.pinch_state

        click_fired = False
        action = "none"

        if not state.currently_pinched and now_pinched:
            state.currently_pinched = True
            state.pinch_start_ts = now
            state.click_sent_for_current_pinch = False

        elif state.currently_pinched and now_pinched:
            held_for_s = now - (state.pinch_start_ts or now)
            click_allowed = (now - state.last_click_ts) >= self.click_cooldown_s
            if (
                not state.click_sent_for_current_pinch
                and held_for_s >= self.pinch_hold_s
                and click_allowed
            ):
                pyautogui.click()
                state.last_click_ts = now
                state.click_sent_for_current_pinch = True
                click_fired = True
                action = "left_click"
        else:
            state.currently_pinched = False
            state.pinch_start_ts = None
            state.click_sent_for_current_pinch = False

        held_for_s = 0.0
        if state.currently_pinched and state.pinch_start_ts is not None:
            held_for_s = max(0.0, now - state.pinch_start_ts)

        return {
            "action": action,
            "click_fired": click_fired,
            "pinch_distance": pinch_distance,
            "is_pinched": state.currently_pinched,
            "held_for_s": held_for_s,
            "cooldown_remaining_s": max(0.0, self.click_cooldown_s - (now - state.last_click_ts)),
        }

    def process_scroll(self, *, hand_landmarks: Sequence) -> dict:
        """
        Scroll rule:
        - If index tip and middle tip are close, scroll mode is active.
        - While active, vertical movement determines scroll direction.
        - If fingers separate, scroll mode stops and resets.
        """
        index_tip = hand_landmarks[8]
        middle_tip = hand_landmarks[12]
        close_distance = math.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)
        fingers_close = close_distance < self.scroll_finger_close_threshold

        state = self.scroll_state
        action = "none"

        if fingers_close:
            if not state.scroll_active:
                state.scroll_active = True
                state.anchor_y = index_tip.y
            else:
                if state.anchor_y is None:
                    state.anchor_y = index_tip.y
                delta_y = index_tip.y - state.anchor_y
                if delta_y >= self.scroll_min_delta_y:
                    pyautogui.scroll(-self.scroll_step)
                    action = "scroll_down"
                    state.anchor_y = index_tip.y
                elif delta_y <= -self.scroll_min_delta_y:
                    pyautogui.scroll(self.scroll_step)
                    action = "scroll_up"
                    state.anchor_y = index_tip.y
        else:
            state.scroll_active = False
            state.anchor_y = None

        return {
            "action": action,
            "scroll_active": state.scroll_active,
            "fingers_close": fingers_close,
            "fingers_distance": close_distance,
        }

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
