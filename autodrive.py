"""
autodrive.py
============
Real-time Autonomous Driving – Behavioral Cloning Inference
Hardware  : UBTECH RoboGo RC Car · NVIDIA Jetson Nano (JetPack 4.6)
Model     : NVIDIA PilotNet (trained via train_model.py)

Usage
-----
    python3 autodrive.py

Press 'Q' on the OpenCV preview window to safely stop the car.
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

# ── Project model architecture ────────────────────────────────────────────────
from train_model import PilotNet

# ── Proprietary RoboGo motor/servo API ────────────────────────────────────────
from oneai.robogo.robogo_device_solver import RoboGoDeviceSolver

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "robogo_pilotnet.pth"          # trained checkpoint

MAX_ANGLE      = 20       # max steering angle in degrees (hardware limit)
SPEED          = 25       # base forward speed (reduced from 30 for stability)
MIN_SPEED      = 15       # minimum speed during sharp turns
DEADZONE       = 2.0      # ±degrees – within this band, drive straight
STEERING_SCALE = 0.7      # dampen model output (< 1.0 = less aggressive turns)
EMA_ALPHA      = 0.4      # smoothing factor: 0.0 = full smoothing, 1.0 = no smoothing

CAMERA_ID  = 0          # /dev/video0
DISPLAY_W  = 320        # HUD preview width
DISPLAY_H  = 240        # HUD preview height

# PilotNet canonical input (must match training pipeline in train_model.py)
IMG_W, IMG_H = 200, 66

# ImageNet normalisation (same as training)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
# Device selection – prefer GPU (CUDA) on Jetson Nano
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing transform (mirrors training _base_tf exactly)
# ─────────────────────────────────────────────────────────────────────────────
preprocess = T.Compose([
    T.Resize((IMG_H, IMG_W)),   # (Height, Width) – torchvision convention
    T.ToTensor(),               # HWC uint8 [0,255] → CHW float32 [0,1]
    T.Normalize(mean=MEAN, std=STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Helper – load model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(path: str) -> PilotNet:
    """
    Instantiate PilotNet, load trained weights from checkpoint, move to
    GPU and switch to eval mode (disables dropout / batchnorm running stats).
    """
    model = PilotNet()

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    model.to(device)
    model.eval()

    print(f"[INFO] Model loaded from '{path}'")
    print(f"[INFO] Checkpoint epoch  : {checkpoint.get('epoch', '?')}")
    print(f"[INFO] Checkpoint val MSE: {checkpoint.get('val_loss', '?')}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Helper – preprocess a single BGR frame from the camera
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_steering(model: PilotNet, frame_bgr: np.ndarray) -> float:
    """
    Convert a raw BGR frame to the tensor format expected by PilotNet,
    run a forward pass on the GPU, and return the predicted steering
    angle in degrees (range ≈ [-MAX_ANGLE, +MAX_ANGLE]).
    """
    # BGR → RGB (OpenCV reads BGR, PyTorch/PIL expect RGB)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # NumPy → PIL Image (required by torchvision transforms)
    pil_img = Image.fromarray(frame_rgb)

    # Apply the exact same transforms used during training
    tensor = preprocess(pil_img)            # shape: (3, 66, 200)
    tensor = tensor.unsqueeze(0)            # add batch dim → (1, 3, 66, 200)
    tensor = tensor.to(device)              # move to GPU

    # Forward pass (inference – no gradient computation)
    output = model(tensor)                  # shape: (1, 1)
    normalised_angle = output.item()        # scalar in ≈ [-1, 1]

    # Scale to real steering angle in degrees, dampened by STEERING_SCALE
    steering_angle = normalised_angle * MAX_ANGLE * STEERING_SCALE
    return steering_angle


# ─────────────────────────────────────────────────────────────────────────────
# Helper – send motor commands based on predicted angle
# ─────────────────────────────────────────────────────────────────────────────
def compute_adaptive_speed(angle: float) -> int:
    """
    Reduce speed proportionally to steering angle.
    Straight → SPEED,  sharp turn → MIN_SPEED.
    """
    ratio = abs(angle) / MAX_ANGLE          # 0.0 (straight) → 1.0 (max turn)
    ratio = min(ratio, 1.0)
    speed = int(SPEED - ratio * (SPEED - MIN_SPEED))
    return max(MIN_SPEED, speed)


def execute_steering(robot: RoboGoDeviceSolver, angle: float) -> tuple:
    """
    Translate a steering angle (degrees) into RoboGo API calls.

    Returns (direction_label, actual_speed) for the HUD overlay.
    """
    speed = compute_adaptive_speed(angle)

    if abs(angle) <= DEADZONE:
        # ── Straight ──────────────────────────────────────────────────────
        robot.servo_comeback_center()
        robot.drive_forward(speed)
        return "STRAIGHT", speed
    elif angle < 0:
        # ── Left turn ─────────────────────────────────────────────────────
        robot.drive_left(abs(angle))        # API accepts positive values only
        robot.drive_forward(speed)
        return "LEFT", speed
    else:
        # ── Right turn ────────────────────────────────────────────────────
        robot.drive_right(abs(angle))       # API accepts positive values only
        robot.drive_forward(speed)
        return "RIGHT", speed


# ─────────────────────────────────────────────────────────────────────────────
# Helper – draw HUD overlay on the display frame
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, angle: float, direction: str,
             speed: int, fps: float):
    """
    Overlay steering info and status text onto the preview frame.
    """
    # ── Background bar at the top for readability ─────────────────────────
    cv2.rectangle(frame, (0, 0), (DISPLAY_W, 70), (0, 0, 0), cv2.FILLED)

    # ── "AUTOPILOT: ON" in green ──────────────────────────────────────────
    cv2.putText(
        frame, "AUTOPILOT: ON",
        (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 255, 0), 2, cv2.LINE_AA,
    )

    # ── Steering angle + direction ────────────────────────────────────────
    angle_text = f"Angle: {angle:+6.2f} deg  [{direction}]"
    cv2.putText(
        frame, angle_text,
        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (255, 255, 255), 1, cv2.LINE_AA,
    )

    # ── Speed indicator ───────────────────────────────────────────────────
    speed_text = f"Speed: {speed}"
    cv2.putText(
        frame, speed_text,
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (0, 200, 255), 1, cv2.LINE_AA,
    )

    # ── FPS counter (bottom-right) ────────────────────────────────────────
    fps_text = f"{fps:.1f} FPS"
    cv2.putText(
        frame, fps_text,
        (DISPLAY_W - 95, DISPLAY_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (0, 255, 255), 1, cv2.LINE_AA,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RoboGo AutoPilot – PilotNet Inference")
    print("=" * 60)
    print(f"[INFO] Device    : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU       : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Max angle : {MAX_ANGLE}°")
    print(f"[INFO] Speed     : {SPEED}")
    print(f"[INFO] Deadzone  : ±{DEADZONE}°")
    print()

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(MODEL_PATH)

    # ── Initialise robot ──────────────────────────────────────────────────
    print("[INFO] Initialising RoboGo hardware…")
    robot = RoboGoDeviceSolver()
    robot.load()
    print("[INFO] RoboGo loaded and ready.")

    # ── Open camera ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera! Exiting.")
        robot.drive_stop()
        robot.unload()
        sys.exit(1)

    print(f"[INFO] Camera {CAMERA_ID} opened.")
    print("[INFO] Press 'Q' on the preview window to stop.\n")

    # ── Inference loop ────────────────────────────────────────────────────
    try:
        frame_count = 0
        fps = 0.0
        t_start = time.time()
        smoothed_angle = 0.0         # EMA state

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame – retrying…")
                continue

            # ── Predict steering angle ────────────────────────────────────
            raw_angle = predict_steering(model, frame_bgr)

            # ── EMA smoothing (reduces jitter between frames) ─────────────
            smoothed_angle = EMA_ALPHA * raw_angle + (1 - EMA_ALPHA) * smoothed_angle

            # Clamp to hardware limits (safety)
            angle = max(-MAX_ANGLE, min(MAX_ANGLE, smoothed_angle))

            # ── Send commands to motors ───────────────────────────────────
            direction, speed = execute_steering(robot, angle)

            # ── FPS calculation ───────────────────────────────────────────
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            # ── HUD display ───────────────────────────────────────────────
            display_frame = cv2.resize(frame_bgr, (DISPLAY_W, DISPLAY_H))
            draw_hud(display_frame, angle, direction, speed, fps)

            cv2.imshow("RoboGo AutoPilot", display_frame)

            # ── Quit on 'Q' ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("\n[INFO] 'Q' pressed – stopping autopilot…")
                break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt – stopping autopilot…")

    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}")

    finally:
        # ── CRITICAL CLEANUP ──────────────────────────────────────────────
        # Always stop the car, release hardware, and close windows
        # regardless of how the loop terminated (normal, crash, Ctrl-C).
        print("[INFO] Cleaning up…")

        try:
            robot.drive_stop()
            print("[INFO] Motors stopped.")
        except Exception as e:
            print(f"[WARN] Failed to stop motors: {e}")

        try:
            robot.unload()
            print("[INFO] RoboGo unloaded.")
        except Exception as e:
            print(f"[WARN] Failed to unload robot: {e}")

        try:
            cap.release()
            print("[INFO] Camera released.")
        except Exception as e:
            print(f"[WARN] Failed to release camera: {e}")

        cv2.destroyAllWindows()
        print("[INFO] Autopilot terminated safely. Goodbye!")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
