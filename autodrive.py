"""
autodrive.py
============
Behavioral Cloning – Autonomous Inference Script
Hardware  : UBTECH RoboGo on NVIDIA Jetson Nano (ARM64, JetPack 4.6)
Model     : NVIDIA PilotNet (trained via train_model.py)
Weights   : robogo_pilotnet.pth

Controls (OpenCV window must be in focus):
  Q / ESC  – Safe quit
  Any key  – (display only, car is fully autonomous)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import sys
import time

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

# Proprietary RoboGo control API
from oneai.robogo.robogo_device_solver import RoboGoDeviceSolver

# PilotNet architecture (defined in same directory)
from train_model import PilotNet


# ─────────────────────────────────────────────────────────────────────────────
# Constants  (must match collect_data.py and train_model.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
WEIGHTS_PATH  = "robogo_pilotnet.pth"   # path to trained checkpoint
MAX_ANGLE     = 20                       # degrees – matches training normalisation
SPEED         = 30                       # motor speed unit
DEADZONE_DEG  = 1.0                      # ± degrees; go straight within this band

DISPLAY_W     = 320                      # HUD preview width  (pixels)
DISPLAY_H     = 240                      # HUD preview height (pixels)

# PilotNet canonical input size – MUST match train_model.py  IMG_H × IMG_W
MODEL_H       = 66
MODEL_W       = 200

# ImageNet statistics – identical to training pipeline
MEAN          = [0.485, 0.456, 0.406]
STD           = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing transform  (mirrors DrivingDataset._base_tf in train_model.py)
# ─────────────────────────────────────────────────────────────────────────────
preprocess = T.Compose([
    T.Resize((MODEL_H, MODEL_W)),   # torchvision uses (H, W) — correct order
    T.ToTensor(),                    # uint8 [0,255] → float32 [0,1]
    T.Normalize(mean=MEAN, std=STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    """
    Instantiate PilotNet, load weights from the 'model_state' key inside
    the checkpoint dict, move to device, and set to eval mode.
    """
    print(f"[INFO] Loading checkpoint: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)

    if "model_state" not in checkpoint:
        raise KeyError(
            f"[ERROR] Key 'model_state' not found in checkpoint. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    model = PilotNet(dropout_p=0.1)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    saved_epoch = checkpoint.get("epoch", "?")
    saved_loss  = checkpoint.get("val_loss", float("nan"))
    print(f"[INFO] Weights loaded  — saved epoch: {saved_epoch}  "
          f"| best val MSE: {saved_loss:.6f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Helper: frame → tensor on device
# ─────────────────────────────────────────────────────────────────────────────
def frame_to_tensor(bgr_frame: np.ndarray,
                    device: torch.device) -> torch.Tensor:
    """
    Convert an OpenCV BGR frame to a normalised (1, 3, 66, 200) CUDA tensor.

    Pipeline:
      1. BGR → RGB  (OpenCV reads BGR; the model was trained on RGB PIL images)
      2. numpy → PIL Image
      3. Resize to (MODEL_H, MODEL_W) = (66, 200)
      4. ToTensor  → [0, 1] float32
      5. Normalise with ImageNet stats
      6. unsqueeze(0)  → add batch dimension
      7. Move to device
    """
    rgb   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil   = Image.fromarray(rgb)
    tensor = preprocess(pil)          # shape: (3, 66, 200)
    tensor = tensor.unsqueeze(0)      # shape: (1, 3, 66, 200)
    return tensor.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: apply steering prediction to robot
# ─────────────────────────────────────────────────────────────────────────────
def apply_steering(robot: RoboGoDeviceSolver,
                   angle_deg: float) -> str:
    """
    Translate a steering angle (degrees, signed) into robot API commands.

    Returns a short label string for the HUD.
    """
    if abs(angle_deg) <= DEADZONE_DEG:
        # Deadzone → drive perfectly straight, suppress servo jitter
        robot.servo_comeback_center()
        robot.drive_forward(SPEED)
        return "STRAIGHT"

    elif angle_deg < 0:
        # Negative = left
        robot.drive_left(int(abs(angle_deg)))
        robot.drive_forward(SPEED)
        return f"LEFT  {abs(angle_deg):.1f} deg"

    else:
        # Positive = right
        robot.drive_right(int(angle_deg))
        robot.drive_forward(SPEED)
        return f"RIGHT {angle_deg:.1f} deg"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: draw HUD overlay on display frame
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray,
             angle_deg: float,
             direction_label: str,
             fps: float,
             frame_idx: int) -> np.ndarray:
    """
    Draws a semi-transparent HUD panel with:
      • AUTOPILOT: ON  (green)
      • Steering bar   (orange = left, blue = right)
      • Numeric angle + direction label
      • FPS counter
      • Frame counter
      • Bottom controls reminder
    Modifies `frame` in-place and returns it.
    """
    h, w = frame.shape[:2]
    norm_angle = float(np.clip(angle_deg / MAX_ANGLE, -1.0, 1.0))

    # ── Semi-transparent dark info panel (top) ────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (w - 8, 162), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # ── AUTOPILOT status (green) ──────────────────────────────────────────
    cv2.circle(frame, (22, 26), 7, (0, 220, 70), -1)
    cv2.putText(frame, "AUTOPILOT: ON",
                (36, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (0, 230, 80), 1, cv2.LINE_AA)

    # ── Steering bar ──────────────────────────────────────────────────────
    bar_x, bar_y = 16, 42
    bar_w, bar_h = w - 24, 14
    cv2.rectangle(frame,
                  (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  (55, 55, 55), -1)
    mid_x = bar_x + bar_w // 2
    fill_w = int(abs(norm_angle) * (bar_w // 2))

    if norm_angle < 0:                          # left → orange
        cv2.rectangle(frame,
                      (mid_x - fill_w, bar_y),
                      (mid_x, bar_y + bar_h),
                      (0, 140, 255), -1)
    else:                                       # right → blue
        cv2.rectangle(frame,
                      (mid_x, bar_y),
                      (mid_x + fill_w, bar_y + bar_h),
                      (220, 120, 0), -1)

    # Centre tick
    cv2.line(frame, (mid_x, bar_y), (mid_x, bar_y + bar_h),
             (255, 255, 255), 1)

    # ── Text rows ─────────────────────────────────────────────────────────
    angle_sign = "+" if angle_deg >= 0 else ""
    rows = [
        f"ANGLE   {angle_sign}{angle_deg:.2f} deg",
        f"ACTION  {direction_label}",
        f"FPS     {fps:.1f}",
        f"FRAMES  {frame_idx:06d}",
    ]
    for i, text in enumerate(rows):
        y = 76 + i * 20
        cv2.putText(frame, text,
                    (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (195, 195, 195), 1, cv2.LINE_AA)

    # ── Bottom controls reminder ──────────────────────────────────────────
    cv2.rectangle(frame, (8, h - 22), (w - 8, h - 4), (15, 15, 15), -1)
    cv2.putText(frame, "Q / ESC : quit  |  window must be focused",
                (14, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                (130, 130, 130), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Device selection ──────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Inference device : {device}")
    if device.type == 'cuda':
        print(f"       GPU  : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"       VRAM : {vram_gb:.1f} GB")
    else:
        print("[WARN] CUDA not available – running on CPU (will be slow).")

    # ── Load model ────────────────────────────────────────────────────────
    try:
        model = load_model(WEIGHTS_PATH, device)
    except (FileNotFoundError, KeyError) as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)

    # ── Enable cuDNN benchmark for fixed-size inference throughput ────────
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # ── Warm up GPU (avoids first-frame latency spike) ────────────────────
    if device.type == 'cuda':
        print("[INFO] Warming up GPU inference pipeline...")
        dummy = torch.zeros(1, 3, MODEL_H, MODEL_W, device=device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy)
        torch.cuda.synchronize()
        print("[INFO] GPU warm-up complete.")

    # ── Init robot ────────────────────────────────────────────────────────
    print("[INFO] Initializing RoboGo robot...")
    robot = RoboGoDeviceSolver()
    robot.load()
    print("[INFO] Robot loaded and ready.")

    # ── Init camera ───────────────────────────────────────────────────────
    print("[INFO] Opening camera (index 0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FATAL] Could not open camera. Shutting down.")
        robot.drive_stop()
        robot.unload()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("RoboGo AutoDrive", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RoboGo AutoDrive", DISPLAY_W, DISPLAY_H)

    print("[INFO] Autopilot active.  Press Q or ESC in the window to quit.")
    print("=" * 55)

    # ── Runtime state ─────────────────────────────────────────────────────
    frame_idx  = 0
    fps        = 0.0
    fps_timer  = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────
    try:
        while True:
            # ── Grab frame ────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Empty camera frame – skipping.")
                continue

            # ── Inference ─────────────────────────────────────────────────
            tensor = frame_to_tensor(frame, device)

            with torch.no_grad():
                prediction = model(tensor)           # shape: (1, 1)

            # Denormalise: [-1, 1] → [-MAX_ANGLE, MAX_ANGLE] degrees
            norm_angle = float(prediction.item())
            norm_angle = float(np.clip(norm_angle, -1.0, 1.0))   # safety clamp
            angle_deg  = norm_angle * MAX_ANGLE

            # ── Robot control with deadzone ────────────────────────────────
            direction_label = apply_steering(robot, angle_deg)

            # ── FPS calculation (rolling 30-frame average) ─────────────────
            frame_idx += 1
            if frame_idx % 30 == 0:
                now      = time.time()
                fps      = 30.0 / max(now - fps_timer, 1e-6)
                fps_timer = now
                print(f"[DRIVE] frame={frame_idx:06d}  "
                      f"angle={angle_deg:+6.2f} deg  "
                      f"({direction_label:20s})  "
                      f"fps={fps:.1f}")

            # ── HUD overlay ───────────────────────────────────────────────
            display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            display = draw_hud(display, angle_deg, direction_label,
                                fps, frame_idx)
            cv2.imshow("RoboGo AutoDrive", display)

            # ── Key input (1 ms poll – keep GUI responsive) ────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):   # Q or ESC
                print("[INFO] Quit requested by user.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by Ctrl+C.")

    finally:
        # ── CRITICAL: always stop motors before exit ───────────────────────
        print("[INFO] Stopping robot...")
        robot.drive_stop()
        print("[INFO] Unloading robot...")
        robot.unload()
        print("[INFO] Releasing camera...")
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Autopilot session ended. Total frames processed: {frame_idx}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()