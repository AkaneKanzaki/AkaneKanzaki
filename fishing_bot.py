"""
Fishing bot prototype that automates a three-step Q/W/E minigame using:
- Screen capture via mss
- HSV masking to locate a rotating needle/arrow and its target
- Template-matching OCR to read dynamic Q/W/E prompts
- pydirectinput to send keyboard inputs when alignment is detected

This script is intentionally modular so the color ranges, capture regions,
and template assets can be tuned to a specific game layout.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import mss
import numpy as np
import pydirectinput


Point = Tuple[int, int]
MOMENT_EPSILON = 1e-3


@dataclass
class HSVRange:
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]


@dataclass
class BotConfig:
    monitor_index: int = 1  # 1-based index used by mss
    capture_region: Optional[Dict[str, int]] = None  # left, top, width, height
    pointer_hsv: HSVRange = field(
        default_factory=lambda: HSVRange((40, 60, 60), (80, 255, 255))
    )
    # Default pointer hue range is green-ish; adjust to match in-game arrow/needle color.
    target_hsv: HSVRange = field(
        default_factory=lambda: HSVRange((0, 150, 150), (10, 255, 255))
    )
    # Default target hue range is red-ish; adjust to the in-game target highlight.
    min_contour_area: int = 50
    angle_tolerance_deg: float = 6.0
    # Screen region (x, y, w, h) where Q/W/E prompts appear; tune for your layout.
    letter_region: Tuple[int, int, int, int] = (0, 0, 320, 160)
    template_dir: str = "templates"
    template_threshold: float = 0.55
    restart_delay_seconds: float = 4.0
    key_cooldown_seconds: float = 0.35
    restart_key: str = "2"
    loop_sleep_seconds: float = 0.01


class FishingBot:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.sct = mss.mss()
        self.templates: Dict[str, np.ndarray] = self._load_templates()
        self.stage_index = 0
        self.last_press_time = 0.0
        self.restart_at: Optional[float] = None

    def _load_templates(self) -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        for letter in ("q", "w", "e"):
            path = os.path.join(self.config.template_dir, f"{letter}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[letter.upper()] = img
                else:
                    print(f"[WARN] Template unreadable or corrupt: {path}")
            else:
                print(f"[WARN] Template not found: {path}")
        if not templates:
            print("[WARN] No letter templates loaded; OCR will fail.")
        return templates

    def grab_frame(self) -> np.ndarray:
        region = (
            self.config.capture_region or self.sct.monitors[self.config.monitor_index]
        )
        shot = self.sct.grab(region)
        frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
        return frame

    def _find_centroid(self, mask: np.ndarray) -> Optional[Point]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.config.min_contour_area:
            return None
        moments = cv2.moments(contour)
        if moments["m00"] < MOMENT_EPSILON:
            return None
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)

    def detect_pointer_and_target(self, frame: np.ndarray) -> Tuple[Optional[Point], Optional[Point]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pointer_mask = cv2.inRange(
            hsv, np.array(self.config.pointer_hsv.lower), np.array(self.config.pointer_hsv.upper)
        )
        target_mask = cv2.inRange(
            hsv, np.array(self.config.target_hsv.lower), np.array(self.config.target_hsv.upper)
        )
        return self._find_centroid(pointer_mask), self._find_centroid(target_mask)

    @staticmethod
    def _angle_from_center(center: Point, point: Point) -> float:
        # Convert screen coords (y increases downward) to math coords (y upward) for atan2.
        dx = point[0] - center[0]
        dy = center[1] - point[1]
        return math.degrees(math.atan2(dy, dx))

    def relative_angle(self, frame: np.ndarray, pointer: Point, target: Point) -> float:
        h, w, _ = frame.shape
        center = (w // 2, h // 2)
        pointer_angle = self._angle_from_center(center, pointer)
        target_angle = self._angle_from_center(center, target)
        diff = (pointer_angle - target_angle + 180) % 360 - 180
        return diff

    def detect_letter_prompt(self, frame: np.ndarray) -> Optional[str]:
        x, y, w, h = self.config.letter_region
        cropped = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        best_letter = None
        best_score = -1.0
        for letter, template in self.templates.items():
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_letter = letter
        if best_score >= self.config.template_threshold:
            return best_letter
        return None

    def press_key(self, key: str) -> None:
        now = time.time()
        if now - self.last_press_time < self.config.key_cooldown_seconds:
            return
        pydirectinput.press(key)
        self.last_press_time = now
        if self.stage_index < 3:
            self.stage_index += 1
            if self.stage_index == 3:
                self.restart_at = now + self.config.restart_delay_seconds

    def maybe_restart_cycle(self) -> None:
        if self.restart_at is None:
            return
        if time.time() >= self.restart_at:
            pydirectinput.press(self.config.restart_key)
            self.stage_index = 0
            self.restart_at = None

    def run(self) -> None:
        print("Starting fishing bot. Press Ctrl+C to exit.")
        while True:
            try:
                frame = self.grab_frame()
                pointer, target = self.detect_pointer_and_target(frame)
                letter = self.detect_letter_prompt(frame) if self.stage_index < 3 else None
                if pointer and target and letter:
                    angle_diff = self.relative_angle(frame, pointer, target)
                    if abs(angle_diff) <= self.config.angle_tolerance_deg:
                        self.press_key(letter.lower())
                self.maybe_restart_cycle()
                time.sleep(self.config.loop_sleep_seconds)
            except KeyboardInterrupt:
                print("Stopping bot.")
                break
            except (mss.exception.ScreenShotError, cv2.error, OSError) as exc:
                print(f"[WARN] Loop error: {exc}")
                time.sleep(0.1)


if __name__ == "__main__":
    bot = FishingBot(BotConfig())
    bot.run()
