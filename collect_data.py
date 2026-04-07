"""
collect_data.py
Behavioral Cloning - Data Collection Script
Hardware: UBTECH RoboGo on ARM64 AI Box (Linux)

Controls:
  W - Drive straight
  A - Steer left  (smooth, incremental)
  D - Steer right (smooth, incremental)
  Q - Quit safely
  No key - Stop
"""

import cv2
import csv
import os
import time
import datetime
import numpy as np
from typing import Tuple

from oneai.robogo.robogo_device_solver import RoboGoDeviceSolver

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MAX_ANGLE   = 20        # degrees, max steering lock
SPEED       = 30        # motor speed unit
STEER_STEP  = 4         # degrees incremented per frame while key held
DEBOUNCE_S  = 0.20      # seconds: ignore key-release gaps shorter than this
FRAME_W     = 320
FRAME_H     = 240
DATASET_DIR = "my_dataset"
IMG_DIR     = os.path.join(DATASET_DIR, "images")
LOG_PATH    = os.path.join(DATASET_DIR, "driving_log.csv")

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
current_angle   = 0.0   # degrees, negative = left, positive = right
last_active_key = None  # 'W', 'A', 'D', or None
last_key_time   = 0.0   # epoch seconds of last recognised key press
is_moving       = False # True when the robot is currently driving
frame_count     = 0


def setup_dirs():
    """Create dataset directories and CSV header if needed."""
    os.makedirs(IMG_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_filename", "steering_angle", "speed"])


def normalize_angle(angle_deg: float) -> float:
    """Map [-MAX_ANGLE, MAX_ANGLE] → [-1.0, 1.0]."""
    return float(np.clip(angle_deg / MAX_ANGLE, -1.0, 1.0))


def save_frame(frame, angle_deg: float):
    """Resize, save image, and append to CSV."""
    resized = cv2.resize(frame, (FRAME_W, FRAME_H))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"frame_{ts}.jpg"
    img_path = os.path.join(IMG_DIR, filename)
    cv2.imwrite(img_path, resized)

    norm = normalize_angle(angle_deg)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([filename, f"{norm:.4f}", SPEED])

    return filename, norm


def draw_hud(frame, angle_deg: float, norm_angle: float,
             is_moving: bool, frame_count: int, active_key: str):
    """
    Overlay a HUD on the given frame (in-place).
    Layout:
      ┌─────────────────────────────────┐
      │ STATUS  ●RECORDING / ■ STOPPED  │
      │ KEY     W / A / D / —           │
      │ ANGLE   +12.0 °  [====|    ]    │
      │ NORM    +0.60                   │
      │ SPEED   30                      │
      │ FRAMES  000042                  │
      └─────────────────────────────────┘
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (w - 8, 148), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Status indicator
    if is_moving:
        status_text  = "RECORDING"
        status_color = (0, 220, 80)   # green
        status_dot   = (0, 220, 80)
    else:
        status_text  = "STOPPED"
        status_color = (60, 60, 220)  # blue-ish
        status_dot   = (80, 80, 220)

    cv2.circle(frame, (22, 24), 6, status_dot, -1)
    cv2.putText(frame, status_text, (34, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)

    # Key indicator
    key_label = active_key if active_key else "—"
    cv2.putText(frame, f"KEY    {key_label}", (16, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Steering bar  ──────[||]──────
    bar_x, bar_y, bar_w, bar_h = 16, 62, w - 24, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    mid_x = bar_x + bar_w // 2
    # Fill from center
    fill_w = int(abs(norm_angle) * (bar_w // 2))
    if norm_angle < 0:  # left
        fill_color = (220, 140, 0)   # orange-left
        cv2.rectangle(frame,
                      (mid_x - fill_w, bar_y),
                      (mid_x, bar_y + bar_h),
                      fill_color, -1)
    else:               # right
        fill_color = (0, 160, 220)   # blue-right
        cv2.rectangle(frame,
                      (mid_x, bar_y),
                      (mid_x + fill_w, bar_y + bar_h),
                      fill_color, -1)
    # Center tick
    cv2.line(frame, (mid_x, bar_y), (mid_x, bar_y + bar_h), (255, 255, 255), 1)

    # Angle text
    angle_sign = "+" if angle_deg >= 0 else ""
    cv2.putText(frame, f"ANGLE  {angle_sign}{angle_deg:.1f} deg",
                (16, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"NORM   {norm_angle:+.3f}",
                (16, 113), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"SPEED  {SPEED}",
                (16, 131), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FRAMES {frame_count:06d}",
                (16, 149), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Controls reminder (bottom bar)
    cv2.rectangle(frame, (8, h - 22), (w - 8, h - 4), (20, 20, 20), -1)
    cv2.putText(frame, "W:fwd  A:left  D:right  ESC:quit",
                (14, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (140, 140, 140), 1, cv2.LINE_AA)

    return frame


def apply_drive(robot: RoboGoDeviceSolver,
                key_char: str,
                prev_angle: float) -> Tuple[float, bool]:
    """
    Given the effective key press, update angle and send commands to robot.
    Returns (new_angle, is_moving).
    """
    if key_char == 'W':
        new_angle = 0.0
        robot.servo_comeback_center()
        robot.drive_forward(SPEED)
        return new_angle, True

    elif key_char == 'A':
        new_angle = max(prev_angle - STEER_STEP, -MAX_ANGLE)
        robot.drive_left(int(abs(new_angle)))
        robot.drive_forward(SPEED)
        return new_angle, True

    elif key_char == 'D':
        new_angle = min(prev_angle + STEER_STEP, MAX_ANGLE)
        robot.drive_right(int(abs(new_angle)))
        robot.drive_forward(SPEED)
        return new_angle, True

    else:  # None / stop
        robot.drive_stop()
        return prev_angle, False


def main():
    global frame_count

    setup_dirs()

    # ── Init robot ────────────────────────────────────────
    print("[INFO] Initializing RoboGo...")
    robot = RoboGoDeviceSolver()
    robot.load()
    print("[INFO] Robot loaded. Starting camera...")

    # ── Init camera ───────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera. Exiting.")
        robot.unload()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("RoboGo Data Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RoboGo Data Collector", 640, 480)

    print("[INFO] Ready. Use W/A/D to drive, Q to quit.")
    print(f"[INFO] Dataset will be saved to: {os.path.abspath(DATASET_DIR)}")

    current_angle  = 0.0
    last_active_key = None
    last_key_time   = 0.0
    is_moving       = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Empty camera frame, skipping.")
                continue

            # ── Key input (30 ms poll → ~33 fps) ──────────
            raw_key = cv2.waitKey(30) & 0xFF
            now = time.time()

            if raw_key == 27:  # ESC
                print("[INFO] Quit requested.")
                break

            # Map raw key → character
            pressed_now = None
            if   raw_key == ord('w') or raw_key == ord('W'):
                pressed_now = 'W'
            elif raw_key == ord('a') or raw_key == ord('A'):
                pressed_now = 'A'
            elif raw_key == ord('d') or raw_key == ord('D'):
                pressed_now = 'D'
            # raw_key == 255 means no key (cv2 timeout)

            # ── Debounce: ignore momentary "no-key" gaps ──
            # If we had an active key and suddenly see no key,
            # keep the last active key alive for DEBOUNCE_S seconds.
            if pressed_now is not None:
                last_active_key = pressed_now
                last_key_time   = now
                effective_key   = pressed_now
            else:
                # No key pressed this frame
                if last_active_key is not None:
                    gap = now - last_key_time
                    if gap < DEBOUNCE_S:
                        # Within debounce window → treat as still held
                        effective_key = last_active_key
                    else:
                        # Gap exceeded → genuinely released
                        effective_key   = None
                        last_active_key = None
                else:
                    effective_key = None

            # ── Apply drive command ────────────────────────
            current_angle, is_moving = apply_drive(
                robot, effective_key, current_angle
            )

            # ── Data collection ───────────────────────────
            if is_moving:
                filename, norm = save_frame(frame, current_angle)
                frame_count += 1
                # Lightweight console feedback every 50 frames
                if frame_count % 50 == 0:
                    print(f"[DATA] {frame_count} frames saved. "
                          f"Last: {filename}  angle={norm:+.3f}")

            # ── HUD overlay ───────────────────────────────
            norm_angle = normalize_angle(current_angle)
            display = draw_hud(
                frame.copy(),
                current_angle,
                norm_angle,
                is_moving,
                frame_count,
                effective_key
            )
            cv2.imshow("RoboGo Data Collector", display)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by Ctrl+C.")

    finally:
        print("[INFO] Stopping robot and releasing resources...")
        robot.drive_stop()
        robot.unload()
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Collection complete. Total frames saved: {frame_count}")
        print(f"[INFO] Dataset location: {os.path.abspath(DATASET_DIR)}")


if __name__ == "__main__":
    main()