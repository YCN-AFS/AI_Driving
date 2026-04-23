"""
client_car.py
=============
Remote Driving Client – runs on the RoboGo car (Jetson Nano)

Captures camera frames, sends JPEG-compressed images to the remote
inference server (Pop!_OS), receives steering angles, and controls
the motors accordingly. No PyTorch required on the car.

Usage
-----
    python3 client_car.py                             # default server IP
    python3 client_car.py --server 192.168.100.148    # explicit IP
    python3 client_car.py --port 6000                 # custom port

Network Info
------------
    Server (Pop!_OS) : 192.168.100.148
    Client (Jetson)  : 192.168.100.172
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports  (NO PyTorch needed on the car!)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import socket
import struct
import sys
import time

import cv2
import numpy as np

from oneai.robogo.robogo_device_solver import RoboGoDeviceSolver

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
# Driving
MAX_ANGLE  = 20
SPEED      = 25        # base forward speed
MIN_SPEED  = 15        # minimum speed during sharp turns
DEADZONE   = 1.5       # ±degrees – within this band, drive straight

# Camera
CAMERA_ID  = 0
JPEG_QUALITY = 80      # JPEG compression (lower = smaller/faster, less detail)

# Network defaults
DEFAULT_SERVER = "192.168.100.149"
DEFAULT_PORT   = 5555

# Reconnection
MAX_RECONNECT_ATTEMPTS = 10
RECONNECT_DELAY        = 2.0    # seconds between reconnection attempts

# HUD display (local preview on car)
DISPLAY_W = 320
DISPLAY_H = 240


# ─────────────────────────────────────────────────────────────────────────────
# Motor control (same as autodrive.py)
# ─────────────────────────────────────────────────────────────────────────────
def compute_adaptive_speed(angle: float) -> int:
    ratio = min(abs(angle) / MAX_ANGLE, 1.0)
    speed = int(SPEED - ratio * (SPEED - MIN_SPEED))
    return max(MIN_SPEED, speed)


def execute_steering(robot: RoboGoDeviceSolver, angle: float):
    speed = compute_adaptive_speed(angle)

    if abs(angle) <= DEADZONE:
        robot.servo_comeback_center()
        robot.drive_forward(speed)
        return "STRAIGHT", speed
    elif angle < 0:
        robot.drive_left(abs(angle))
        robot.drive_forward(speed)
        return "LEFT", speed
    else:
        robot.drive_right(abs(angle))
        robot.drive_forward(speed)
        return "RIGHT", speed


# ─────────────────────────────────────────────────────────────────────────────
# HUD overlay (lightweight – no model info, just driving stats)
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(frame, angle, direction, speed, fps, rtt_ms):
    cv2.rectangle(frame, (0, 0), (DISPLAY_W, 70), (0, 0, 0), cv2.FILLED)

    cv2.putText(
        frame, "REMOTE DRIVE: ON",
        (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0, 255, 0), 2, cv2.LINE_AA,
    )

    angle_text = f"Angle: {angle:+6.2f} [{direction}] Spd:{speed}"
    cv2.putText(
        frame, angle_text,
        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
        (255, 255, 255), 1, cv2.LINE_AA,
    )

    stats_text = f"FPS: {fps:.1f}  RTT: {rtt_ms:.0f}ms"
    cv2.putText(
        frame, stats_text,
        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
        (0, 255, 255), 1, cv2.LINE_AA,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Network helpers
# ─────────────────────────────────────────────────────────────────────────────
def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket."""
    chunks = []
    received = 0
    while received < n:
        chunk = sock.recv(n - received)
        if not chunk:
            raise ConnectionError("Server disconnected")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def connect_to_server(host: str, port: int) -> socket.socket:
    """Connect to the inference server with retry logic."""
    for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
        try:
            print(f"[INFO] Connecting to {host}:{port} "
                  f"(attempt {attempt}/{MAX_RECONNECT_ATTEMPTS})...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(None)

            # TCP_NODELAY for low latency
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            print(f"[INFO] Connected to server {host}:{port}")
            return sock
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            print(f"[WARN] Connection failed: {e}")
            if attempt < MAX_RECONNECT_ATTEMPTS:
                print(f"[INFO] Retrying in {RECONNECT_DELAY}s...")
                time.sleep(RECONNECT_DELAY)
            else:
                raise ConnectionError(
                    f"Cannot connect to server after {MAX_RECONNECT_ATTEMPTS} attempts"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Protocol:
#   Client → Server:  [4 bytes: frame_size (uint32 BE)] [frame_size bytes: JPEG]
#   Server → Client:  [8 bytes: steering_angle (float64 BE)]
# ─────────────────────────────────────────────────────────────────────────────
def send_frame_recv_angle(sock: socket.socket, frame_bgr: np.ndarray) -> float:
    """Compress frame to JPEG, send to server, receive steering angle."""
    # ── Encode frame as JPEG ──────────────────────────────────────────────
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    ok, jpeg_buf = cv2.imencode(".jpg", frame_bgr, encode_params)
    if not ok:
        raise RuntimeError("JPEG encoding failed")

    jpeg_bytes = jpeg_buf.tobytes()
    frame_size = len(jpeg_bytes)

    # ── Send: [4 bytes size] + [JPEG data] ────────────────────────────────
    sock.sendall(struct.pack("!I", frame_size) + jpeg_bytes)

    # ── Receive: [8 bytes steering angle] ─────────────────────────────────
    angle_data = recv_exact(sock, 8)
    angle = struct.unpack("!d", angle_data)[0]

    return angle


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RoboGo Remote Driving Client")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"Server IP (default: {DEFAULT_SERVER})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Server port (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    print("=" * 60)
    print("  RoboGo Remote Driving Client")
    print("=" * 60)
    print(f"[INFO] Server    : {args.server}:{args.port}")
    print(f"[INFO] Speed     : {SPEED} (min: {MIN_SPEED})")
    print(f"[INFO] Deadzone  : ±{DEADZONE}°")
    print(f"[INFO] JPEG qual : {JPEG_QUALITY}")
    print()

    # ── Initialise robot ──────────────────────────────────────────────────
    print("[INFO] Initialising RoboGo hardware...")
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

    # ── Connect to server ─────────────────────────────────────────────────
    sock = None
    try:
        sock = connect_to_server(args.server, args.port)
    except ConnectionError as e:
        print(f"[ERROR] {e}")
        cap.release()
        robot.drive_stop()
        robot.unload()
        sys.exit(1)

    print("[INFO] Press 'Q' on the preview window to stop.\n")

    # ── Main driving loop ─────────────────────────────────────────────────
    try:
        frame_count = 0
        fps = 0.0
        t_fps = time.time()

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame – retrying...")
                continue

            # ── Send frame to server, get steering angle ──────────────────
            t_rtt = time.time()
            try:
                angle = send_frame_recv_angle(sock, frame_bgr)
            except (ConnectionError, BrokenPipeError, OSError) as e:
                print(f"[WARN] Server connection lost: {e}")
                print("[INFO] Stopping car, attempting reconnect...")
                robot.drive_stop()

                try:
                    sock.close()
                except Exception:
                    pass

                try:
                    sock = connect_to_server(args.server, args.port)
                    continue    # retry with next frame
                except ConnectionError:
                    print("[ERROR] Reconnection failed. Exiting.")
                    break

            rtt_ms = (time.time() - t_rtt) * 1000

            # Clamp to hardware limits
            angle = max(-MAX_ANGLE, min(MAX_ANGLE, angle))

            # ── Execute motor commands ────────────────────────────────────
            direction, speed = execute_steering(robot, angle)

            # ── FPS ───────────────────────────────────────────────────────
            frame_count += 1
            elapsed = time.time() - t_fps
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_fps = time.time()

            # ── Local HUD preview ─────────────────────────────────────────
            display = cv2.resize(frame_bgr, (DISPLAY_W, DISPLAY_H))
            draw_hud(display, angle, direction, speed, fps, rtt_ms)

            cv2.imshow("RoboGo Client", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("\n[INFO] 'Q' pressed – stopping.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt – stopping.")

    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}")

    finally:
        # ── CRITICAL CLEANUP ──────────────────────────────────────────────
        print("[INFO] Cleaning up...")

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
            if sock:
                sock.close()
            print("[INFO] Socket closed.")
        except Exception as e:
            print(f"[WARN] Failed to close socket: {e}")

        try:
            cap.release()
            print("[INFO] Camera released.")
        except Exception as e:
            print(f"[WARN] Failed to release camera: {e}")

        cv2.destroyAllWindows()
        print("[INFO] Client terminated safely. Goodbye!")


if __name__ == "__main__":
    main()
