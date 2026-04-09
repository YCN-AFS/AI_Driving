"""
route_recorder.py
=================
Route Recording & Playback – "Teach & Repeat" for RoboGo RC Car
Hardware: UBTECH RoboGo on Jetson Nano (JetPack 4.6)

Modes:
  1. RECORD  – User drives the car with W/A/D keys. Every motor command
               is timestamped and saved as a "route" JSON file.
  2. REPLAY  – The car replays a saved route autonomously, reproducing
               the exact steering/speed sequence with precise timing.
  3. MANAGE  – List, rename, delete saved routes from the CLI.

Controls (during RECORD):
  W  – Drive straight
  A  – Steer left  (incremental)
  D  – Steer right (incremental)
  S  – Stop / brake
  Q  – Quit and save the route

Usage
-----
    python3 route_recorder.py                # interactive CLI menu
    python3 route_recorder.py record         # start recording directly
    python3 route_recorder.py replay         # pick a route and replay
    python3 route_recorder.py list           # list saved routes
    python3 route_recorder.py rename         # rename a route
    python3 route_recorder.py delete         # delete a route
"""

import json
import os
import sys
import time
import datetime
import shutil

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional

from oneai.robogo.robogo_device_solver import RoboGoDeviceSolver

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_ANGLE   = 20        # degrees, max steering lock
SPEED       = 30        # motor speed unit
STEER_STEP  = 4         # degrees incremented per frame while key held
DEBOUNCE_S  = 0.20      # seconds: ignore key-release gaps shorter than this
FRAME_W     = 320
FRAME_H     = 240

ROUTES_DIR  = "saved_routes"    # directory to store route files

# ─────────────────────────────────────────────────────────────────────────────
# Route file structure
# ─────────────────────────────────────────────────────────────────────────────
# Each route is a JSON file:
# {
#     "name": "route_name",
#     "created": "2026-04-09T08:00:00",
#     "duration_s": 32.5,
#     "total_steps": 965,
#     "commands": [
#         {"t": 0.000, "action": "forward", "angle": 0.0, "speed": 30},
#         {"t": 0.033, "action": "left",    "angle": -4.0, "speed": 30},
#         ...
#         {"t": 32.5,  "action": "stop",    "angle": 0.0, "speed": 0},
#     ]
# }


def ensure_routes_dir():
    os.makedirs(ROUTES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HUD overlay (reused from collect_data.py style)
# ─────────────────────────────────────────────────────────────────────────────
def draw_record_hud(frame, angle_deg: float, is_moving: bool,
                    elapsed: float, step_count: int, active_key: str):
    """HUD for recording mode."""
    h, w = frame.shape[:2]
    norm_angle = np.clip(angle_deg / MAX_ANGLE, -1.0, 1.0)

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (w - 8, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Status
    if is_moving:
        cv2.circle(frame, (22, 24), 6, (0, 0, 220), -1)
        cv2.putText(frame, "RECORDING", (34, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (22, 24), 6, (80, 80, 80), -1)
        cv2.putText(frame, "PAUSED", (34, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)

    # Key
    key_label = active_key if active_key else "—"
    cv2.putText(frame, f"KEY    {key_label}", (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Steering bar
    bar_x, bar_y, bar_w, bar_h = 16, 60, w - 24, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    mid_x = bar_x + bar_w // 2
    fill_w = int(abs(norm_angle) * (bar_w // 2))
    if norm_angle < 0:
        cv2.rectangle(frame, (mid_x - fill_w, bar_y),
                      (mid_x, bar_y + bar_h), (220, 140, 0), -1)
    else:
        cv2.rectangle(frame, (mid_x, bar_y),
                      (mid_x + fill_w, bar_y + bar_h), (0, 160, 220), -1)
    cv2.line(frame, (mid_x, bar_y), (mid_x, bar_y + bar_h), (255, 255, 255), 1)

    # Info
    angle_sign = "+" if angle_deg >= 0 else ""
    cv2.putText(frame, f"ANGLE  {angle_sign}{angle_deg:.1f} deg",
                (16, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"TIME   {elapsed:.1f}s   STEPS {step_count}",
                (16, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)

    # Controls
    cv2.rectangle(frame, (8, h - 22), (w - 8, h - 4), (20, 20, 20), -1)
    cv2.putText(frame, "W:fwd  A:left  D:right  S:stop  Q:save&quit",
                (14, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                (140, 140, 140), 1, cv2.LINE_AA)

    return frame


def draw_replay_hud(frame, angle_deg: float, progress: float,
                    elapsed: float, total_duration: float):
    """HUD for replay mode."""
    h, w = frame.shape[:2]
    norm_angle = np.clip(angle_deg / MAX_ANGLE, -1.0, 1.0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (w - 8, 100), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Status
    cv2.circle(frame, (22, 24), 6, (0, 220, 80), -1)
    cv2.putText(frame, "REPLAYING", (34, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 1, cv2.LINE_AA)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 16, 42, w - 24, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    fill = int(progress * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                  (0, 220, 80), -1)

    # Info
    angle_sign = "+" if angle_deg >= 0 else ""
    cv2.putText(frame, f"ANGLE  {angle_sign}{angle_deg:.1f} deg",
                (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"TIME   {elapsed:.1f}s / {total_duration:.1f}s",
                (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)

    # Controls
    cv2.rectangle(frame, (8, h - 22), (w - 8, h - 4), (20, 20, 20), -1)
    cv2.putText(frame, "Q: abort replay",
                (14, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (140, 140, 140), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Motor helpers (same logic as collect_data.py)
# ─────────────────────────────────────────────────────────────────────────────
def apply_drive(robot: RoboGoDeviceSolver, key_char: Optional[str],
                prev_angle: float) -> Tuple[float, bool, str]:
    """
    Apply drive command. Returns (new_angle, is_moving, action_name).
    """
    if key_char == 'W':
        robot.servo_comeback_center()
        robot.drive_forward(SPEED)
        return 0.0, True, "forward"

    elif key_char == 'A':
        new_angle = max(prev_angle - STEER_STEP, -MAX_ANGLE)
        robot.drive_left(int(abs(new_angle)))
        robot.drive_forward(SPEED)
        return new_angle, True, "left"

    elif key_char == 'D':
        new_angle = min(prev_angle + STEER_STEP, MAX_ANGLE)
        robot.drive_right(int(abs(new_angle)))
        robot.drive_forward(SPEED)
        return new_angle, True, "right"

    elif key_char == 'S':
        robot.drive_stop()
        return prev_angle, False, "stop"

    else:
        robot.drive_stop()
        return prev_angle, False, "stop"


def execute_command(robot: RoboGoDeviceSolver, cmd: Dict):
    """Execute a single recorded command on the robot."""
    action = cmd["action"]
    angle  = cmd.get("angle", 0.0)
    speed  = cmd.get("speed", SPEED)

    if action == "forward":
        robot.servo_comeback_center()
        robot.drive_forward(speed)
    elif action == "left":
        robot.drive_left(int(abs(angle)))
        robot.drive_forward(speed)
    elif action == "right":
        robot.drive_right(int(abs(angle)))
        robot.drive_forward(speed)
    elif action == "stop":
        robot.drive_stop()


# ─────────────────────────────────────────────────────────────────────────────
# Route file I/O
# ─────────────────────────────────────────────────────────────────────────────
def save_route(name: str, commands: List[Dict]):
    """Save a route to a JSON file."""
    ensure_routes_dir()

    duration = commands[-1]["t"] if commands else 0.0
    route_data = {
        "name": name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(duration, 2),
        "total_steps": len(commands),
        "commands": commands,
    }

    filepath = os.path.join(ROUTES_DIR, f"{name}.json")
    with open(filepath, "w") as f:
        json.dump(route_data, f, indent=2)

    print(f"\n[SAVED] Route '{name}' → {filepath}")
    print(f"        Duration: {duration:.1f}s  |  Steps: {len(commands)}")
    return filepath


def load_route(filepath: str) -> Dict:
    """Load a route from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def list_routes() -> List[Dict]:
    """List all saved routes with metadata."""
    ensure_routes_dir()
    routes = []

    for filename in sorted(os.listdir(ROUTES_DIR)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(ROUTES_DIR, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            routes.append({
                "filename": filename,
                "filepath": filepath,
                "name": data.get("name", filename.replace(".json", "")),
                "created": data.get("created", "?"),
                "duration_s": data.get("duration_s", 0),
                "total_steps": data.get("total_steps", 0),
            })
        except (json.JSONDecodeError, IOError):
            print(f"[WARN] Could not read: {filename}")

    return routes


def print_routes(routes: List[Dict]):
    """Pretty-print route list."""
    if not routes:
        print("\n  (No saved routes found)")
        return

    print(f"\n{'#':<4} {'Name':<25} {'Duration':<12} {'Steps':<8} {'Created'}")
    print("─" * 75)
    for i, r in enumerate(routes, 1):
        dur = f"{r['duration_s']:.1f}s"
        print(f"{i:<4} {r['name']:<25} {dur:<12} {r['total_steps']:<8} {r['created']}")


def pick_route(prompt: str = "Select route #") -> Optional[Dict]:
    """Show routes and let user pick one. Returns route metadata or None."""
    routes = list_routes()
    print_routes(routes)

    if not routes:
        return None

    print()
    try:
        choice = input(f"  {prompt} (1-{len(routes)}, or 0 to cancel): ").strip()
        idx = int(choice) - 1
        if idx < 0 or idx >= len(routes):
            print("  Cancelled.")
            return None
        return routes[idx]
    except (ValueError, EOFError):
        print("  Cancelled.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RECORD mode
# ─────────────────────────────────────────────────────────────────────────────
def record_route():
    """Record a new route by driving the car with W/A/D keys."""
    ensure_routes_dir()

    # Get route name
    default_name = datetime.datetime.now().strftime("route_%Y%m%d_%H%M%S")
    print(f"\n  Enter route name (or press Enter for '{default_name}'): ", end="")
    try:
        name = input().strip()
    except EOFError:
        name = ""
    if not name:
        name = default_name

    # Sanitise name for filename
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)

    # Check if exists
    filepath = os.path.join(ROUTES_DIR, f"{safe_name}.json")
    if os.path.exists(filepath):
        print(f"  [WARN] Route '{safe_name}' already exists.")
        confirm = input("  Overwrite? (y/N): ").strip().lower()
        if confirm != 'y':
            print("  Cancelled.")
            return

    # ── Init robot ────────────────────────────────────────────────────────
    print("\n[INFO] Initializing RoboGo...")
    robot = RoboGoDeviceSolver()
    robot.load()
    print("[INFO] Robot loaded.")

    # ── Init camera ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera. Exiting.")
        robot.unload()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Route Recorder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Route Recorder", 640, 480)

    print(f"[INFO] Recording route: '{safe_name}'")
    print("[INFO] Use W/A/D to drive, S to stop, Q to save & quit.\n")

    commands = []
    current_angle = 0.0
    last_active_key = None
    last_key_time = 0.0
    is_moving = False
    step_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            raw_key = cv2.waitKey(30) & 0xFF
            now = time.time()
            elapsed = now - t_start

            # Quit
            if raw_key == ord('q') or raw_key == ord('Q') or raw_key == 27:
                print("\n[INFO] Saving route...")
                break

            # Map key
            pressed_now = None
            if raw_key in (ord('w'), ord('W')):
                pressed_now = 'W'
            elif raw_key in (ord('a'), ord('A')):
                pressed_now = 'A'
            elif raw_key in (ord('d'), ord('D')):
                pressed_now = 'D'
            elif raw_key in (ord('s'), ord('S')):
                pressed_now = 'S'

            # Debounce
            if pressed_now is not None:
                last_active_key = pressed_now
                last_key_time = now
                effective_key = pressed_now
            else:
                if last_active_key is not None:
                    gap = now - last_key_time
                    if gap < DEBOUNCE_S:
                        effective_key = last_active_key
                    else:
                        effective_key = None
                        last_active_key = None
                else:
                    effective_key = None

            # Apply command
            current_angle, is_moving, action = apply_drive(
                robot, effective_key, current_angle
            )

            # Record command
            cmd = {
                "t": round(elapsed, 4),
                "action": action,
                "angle": round(current_angle, 1),
                "speed": SPEED if is_moving else 0,
            }
            commands.append(cmd)
            step_count += 1

            # HUD
            display = draw_record_hud(
                frame.copy(), current_angle, is_moving,
                elapsed, step_count, effective_key
            )
            cv2.imshow("Route Recorder", display)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")

    finally:
        robot.drive_stop()
        robot.unload()
        cap.release()
        cv2.destroyAllWindows()

    # Save
    if commands:
        save_route(safe_name, commands)
    else:
        print("[INFO] No commands recorded, nothing saved.")


# ─────────────────────────────────────────────────────────────────────────────
# REPLAY mode
# ─────────────────────────────────────────────────────────────────────────────
def replay_route():
    """Replay a saved route on the car."""
    print("\n  === Select a route to replay ===")
    route_meta = pick_route("Replay route #")
    if route_meta is None:
        return

    route_data = load_route(route_meta["filepath"])
    commands = route_data["commands"]
    total_duration = route_data.get("duration_s", 0)

    if not commands:
        print("[WARN] Route has no commands.")
        return

    print(f"\n[INFO] Replaying: '{route_data['name']}'")
    print(f"[INFO] Duration: {total_duration:.1f}s  |  Steps: {len(commands)}")

    # ── Init robot ────────────────────────────────────────────────────────
    print("[INFO] Initializing RoboGo...")
    robot = RoboGoDeviceSolver()
    robot.load()
    print("[INFO] Robot loaded.")

    # ── Init camera (for HUD preview) ─────────────────────────────────────
    cap = cv2.VideoCapture(0)
    has_camera = cap.isOpened()
    if has_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cv2.namedWindow("Route Replay", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Route Replay", 640, 480)
    else:
        print("[WARN] Camera not available, no preview.")

    # ── Countdown ─────────────────────────────────────────────────────────
    print("[INFO] Starting in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)
    print("  GO!\n")

    t_start = time.time()
    cmd_idx = 0

    try:
        while cmd_idx < len(commands):
            elapsed = time.time() - t_start

            # Execute commands whose timestamp has been reached
            while cmd_idx < len(commands) and commands[cmd_idx]["t"] <= elapsed:
                cmd = commands[cmd_idx]
                execute_command(robot, cmd)
                cmd_idx += 1

            # Progress
            progress = min(elapsed / total_duration, 1.0) if total_duration > 0 else 1.0
            current_angle = commands[min(cmd_idx, len(commands) - 1)].get("angle", 0.0)

            # HUD
            if has_camera:
                ret, frame = cap.read()
                if ret:
                    display = draw_replay_hud(
                        frame.copy(), current_angle, progress,
                        elapsed, total_duration
                    )
                    cv2.imshow("Route Replay", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    print("\n[INFO] Replay aborted by user.")
                    break
            else:
                # No camera, just rudimentary console progress
                pct = int(progress * 100)
                print(f"\r  Replaying... {pct}%  ({cmd_idx}/{len(commands)} steps)",
                      end="", flush=True)
                time.sleep(0.01)

        else:
            print("\n[INFO] Replay complete!")

    except KeyboardInterrupt:
        print("\n[INFO] Replay interrupted.")

    finally:
        robot.drive_stop()
        robot.unload()
        if has_camera:
            cap.release()
            cv2.destroyAllWindows()
        print("[INFO] Robot stopped and unloaded.")


# ─────────────────────────────────────────────────────────────────────────────
# MANAGE routes (list / rename / delete)
# ─────────────────────────────────────────────────────────────────────────────
def manage_list():
    """List all saved routes."""
    print("\n  === Saved Routes ===")
    routes = list_routes()
    print_routes(routes)


def manage_rename():
    """Rename a saved route."""
    print("\n  === Rename Route ===")
    route_meta = pick_route("Rename route #")
    if route_meta is None:
        return

    new_name = input(f"  New name for '{route_meta['name']}': ").strip()
    if not new_name:
        print("  Cancelled.")
        return

    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in new_name)
    new_filepath = os.path.join(ROUTES_DIR, f"{safe_name}.json")

    if os.path.exists(new_filepath):
        print(f"  [WARN] Route '{safe_name}' already exists.")
        return

    # Update JSON content
    route_data = load_route(route_meta["filepath"])
    route_data["name"] = safe_name

    # Write new file
    with open(new_filepath, "w") as f:
        json.dump(route_data, f, indent=2)

    # Remove old file
    os.remove(route_meta["filepath"])

    print(f"  Renamed: '{route_meta['name']}' → '{safe_name}'")


def manage_delete():
    """Delete a saved route."""
    print("\n  === Delete Route ===")
    route_meta = pick_route("Delete route #")
    if route_meta is None:
        return

    confirm = input(f"  Delete '{route_meta['name']}'? (y/N): ").strip().lower()
    if confirm != 'y':
        print("  Cancelled.")
        return

    os.remove(route_meta["filepath"])
    print(f"  Deleted: '{route_meta['name']}'")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI Menu
# ─────────────────────────────────────────────────────────────────────────────
def interactive_menu():
    """Main interactive menu loop."""
    while True:
        print("\n" + "=" * 50)
        print("  RoboGo Route Manager")
        print("=" * 50)
        print("  1. Record a new route")
        print("  2. Replay a saved route")
        print("  3. List saved routes")
        print("  4. Rename a route")
        print("  5. Delete a route")
        print("  0. Exit")
        print()

        try:
            choice = input("  Select (0-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if choice == '1':
            record_route()
        elif choice == '2':
            replay_route()
        elif choice == '3':
            manage_list()
        elif choice == '4':
            manage_rename()
        elif choice == '5':
            manage_delete()
        elif choice == '0':
            print("Goodbye!")
            break
        else:
            print("  Invalid choice.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Direct subcommand support
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "record":
            record_route()
        elif cmd == "replay":
            replay_route()
        elif cmd == "list":
            manage_list()
        elif cmd == "rename":
            manage_rename()
        elif cmd == "delete":
            manage_delete()
        else:
            print(f"Unknown command: '{cmd}'")
            print("Usage: python3 route_recorder.py [record|replay|list|rename|delete]")
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
