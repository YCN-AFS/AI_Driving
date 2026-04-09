"""
server_inference.py
===================
Remote Inference Server – runs on Pop!_OS workstation (RTX 4050)

Receives JPEG frames from the RoboGo car over TCP, runs PilotNet
inference on the GPU, and sends back the predicted steering angle.

Usage
-----
    python3 server_inference.py              # listen on 0.0.0.0:5555
    python3 server_inference.py --port 6000  # custom port

Network Info
------------
    Server (Pop!_OS) : 192.168.100.148
    Client (Jetson)  : 192.168.100.172
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import socket
import struct
import time

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from train_model import PilotNet

# ─────────────────────────────────────────────────────────────────────────────
# Constants (mirror autodrive.py tuning)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH     = "robogo_pilotnet.pth"

MAX_ANGLE      = 20
STEERING_SCALE = 0.85
EMA_ALPHA      = 0.5

# PilotNet input size
IMG_W, IMG_H = 200, 66
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Network
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5555

# HUD display
DISPLAY_W = 640
DISPLAY_H = 480

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (identical to training pipeline)
# ─────────────────────────────────────────────────────────────────────────────
preprocess = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(path: str) -> PilotNet:
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
# Inference
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_steering(model: PilotNet, frame_bgr: np.ndarray) -> float:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    output = model(tensor)
    steering_angle = output.item() * MAX_ANGLE * STEERING_SCALE
    return steering_angle


# ─────────────────────────────────────────────────────────────────────────────
# Network helpers
# ─────────────────────────────────────────────────────────────────────────────
def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket, or raise ConnectionError."""
    chunks = []
    received = 0
    while received < n:
        chunk = sock.recv(min(n - received, 65536))
        if not chunk:
            raise ConnectionError("Client disconnected")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, angle: float, fps: float, latency_ms: float):
    display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

    # Background bar
    cv2.rectangle(display, (0, 0), (DISPLAY_W, 80), (0, 0, 0), cv2.FILLED)

    # Mode label
    cv2.putText(
        display, "REMOTE AUTOPILOT: ON",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 0), 2, cv2.LINE_AA,
    )

    # Direction
    if abs(angle) <= 1.5:
        direction = "STRAIGHT"
    elif angle < 0:
        direction = "LEFT"
    else:
        direction = "RIGHT"

    angle_text = f"Angle: {angle:+6.2f} deg  [{direction}]"
    cv2.putText(
        display, angle_text,
        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1, cv2.LINE_AA,
    )

    # Stats
    stats_text = f"FPS: {fps:.1f}  |  Latency: {latency_ms:.1f}ms"
    cv2.putText(
        display, stats_text,
        (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (0, 255, 255), 1, cv2.LINE_AA,
    )

    return display


# ─────────────────────────────────────────────────────────────────────────────
# Protocol:
#   Client → Server:  [4 bytes: frame_size (uint32 BE)] [frame_size bytes: JPEG]
#   Server → Client:  [8 bytes: steering_angle (float64 BE)]
# ─────────────────────────────────────────────────────────────────────────────
def handle_client(conn: socket.socket, addr, model: PilotNet):
    """Handle a single client connection (blocking loop)."""
    print(f"[INFO] Client connected: {addr}")

    smoothed_angle = 0.0
    frame_count = 0
    fps = 0.0
    t_fps = time.time()

    try:
        while True:
            # ── Receive frame size (4 bytes, big-endian uint32) ───────────
            header = recv_exact(conn, 4)
            frame_size = struct.unpack("!I", header)[0]

            if frame_size == 0 or frame_size > 10_000_000:  # sanity check
                print(f"[WARN] Invalid frame size: {frame_size}, skipping")
                continue

            # ── Receive JPEG frame ────────────────────────────────────────
            jpeg_data = recv_exact(conn, frame_size)

            t_start = time.time()

            # ── Decode JPEG ───────────────────────────────────────────────
            np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                print("[WARN] Failed to decode JPEG, skipping")
                # Send zero angle as fallback
                conn.sendall(struct.pack("!d", 0.0))
                continue

            # ── Run inference ─────────────────────────────────────────────
            raw_angle = predict_steering(model, frame_bgr)

            # ── EMA smoothing ─────────────────────────────────────────────
            smoothed_angle = EMA_ALPHA * raw_angle + (1 - EMA_ALPHA) * smoothed_angle
            angle = max(-MAX_ANGLE, min(MAX_ANGLE, smoothed_angle))

            latency_ms = (time.time() - t_start) * 1000

            # ── Send angle back (8 bytes, big-endian float64) ─────────────
            conn.sendall(struct.pack("!d", angle))

            # ── FPS ───────────────────────────────────────────────────────
            frame_count += 1
            elapsed = time.time() - t_fps
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_fps = time.time()

            # ── HUD ───────────────────────────────────────────────────────
            display = draw_hud(frame_bgr, angle, fps, latency_ms)
            cv2.imshow("RoboGo Remote Server", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("[INFO] 'Q' pressed on server – closing connection")
                break

    except ConnectionError:
        print(f"[INFO] Client {addr} disconnected.")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        conn.close()
        cv2.destroyAllWindows()
        print(f"[INFO] Connection to {addr} closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RoboGo Remote Inference Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port")
    args = parser.parse_args()

    print("=" * 60)
    print("  RoboGo Remote Inference Server")
    print("=" * 60)
    print(f"[INFO] Device         : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU            : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Steering scale : {STEERING_SCALE}")
    print(f"[INFO] EMA alpha      : {EMA_ALPHA}")
    print()

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(MODEL_PATH)

    # ── Start TCP server ──────────────────────────────────────────────────
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(1)

    print(f"[INFO] Listening on {args.host}:{args.port}")
    print("[INFO] Waiting for car to connect...\n")

    try:
        while True:
            conn, addr = server_sock.accept()

            # TCP_NODELAY – disable Nagle's algorithm for lower latency
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            handle_client(conn, addr, model)
            print("[INFO] Waiting for next connection...\n")

    except KeyboardInterrupt:
        print("\n[INFO] Server shutting down.")
    finally:
        server_sock.close()
        cv2.destroyAllWindows()
        print("[INFO] Server stopped.")


if __name__ == "__main__":
    main()
