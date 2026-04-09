"""
tools/balance_dataset.py
========================
Balance a driving dataset by downsampling over-represented steering angles.

Reads the source driving_log.csv, keeps ALL turning frames, and randomly
samples straight frames (angle ≈ 0) to a target ratio. Outputs a new
balanced CSV (and optionally copies images) ready for training.

Usage
-----
    python3 tools/balance_dataset.py                     # default my_dataset
    python3 tools/balance_dataset.py saved_routes/route1 # specific session
    python3 tools/balance_dataset.py --all               # merge all sessions

Options:
    --target-straight 0.35   Target ratio of straight frames (default: 35%)
    --output balanced_data   Output directory name (default: balanced_dataset)
"""

import argparse
import csv
import os
import random
import shutil
import sys


def load_csv(csv_path: str) -> list:
    """Load driving_log.csv and return list of rows."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                rows.append(row)
    return rows


def balance_rows(rows: list, target_straight_ratio: float = 0.35,
                 straight_threshold: float = 0.05) -> list:
    """
    Balance rows by downsampling straight frames.
    
    Args:
        rows: list of CSV rows [filename, angle, speed]
        target_straight_ratio: desired fraction of straight frames
        straight_threshold: |angle| below this = "straight"
    
    Returns:
        Balanced list of rows
    """
    straight = []
    turning  = []

    for row in rows:
        angle = abs(float(row[1]))
        if angle < straight_threshold:
            straight.append(row)
        else:
            turning.append(row)

    n_turning = len(turning)

    if n_turning == 0:
        print("[WARN] No turning data found!")
        return rows

    # Calculate how many straight frames to keep
    # target = straight / (straight + turning)
    # straight = target * turning / (1 - target)
    n_straight_target = int(target_straight_ratio * n_turning / (1 - target_straight_ratio))
    n_straight_target = min(n_straight_target, len(straight))

    # Random sample
    random.seed(42)
    straight_sampled = random.sample(straight, n_straight_target)

    balanced = straight_sampled + turning
    random.shuffle(balanced)

    total = len(balanced)
    print(f"  Original:  {len(rows):>6} ({len(straight)} straight + {n_turning} turning)")
    print(f"  Balanced:  {total:>6} ({n_straight_target} straight + {n_turning} turning)")
    print(f"  Straight%: {n_straight_target/total*100:.1f}% (target: {target_straight_ratio*100:.0f}%)")
    print(f"  Removed:   {len(straight) - n_straight_target} excess straight frames")

    return balanced


def merge_and_balance(sources: list, output_dir: str, target_ratio: float):
    """
    Merge multiple session CSVs, balance, and output to a new directory.
    
    Args:
        sources: list of (csv_path, img_dir) tuples
        output_dir: output directory path
        target_ratio: target straight frame ratio
    """
    # Collect all rows with source info
    all_rows = []
    for csv_path, img_dir in sources:
        rows = load_csv(csv_path)
        for row in rows:
            all_rows.append((row, img_dir))

    print(f"\n[INFO] Merged {len(all_rows)} frames from {len(sources)} sources")

    # Balance
    rows_only = [r for r, _ in all_rows]
    balanced_rows = balance_rows(rows_only, target_ratio)

    # Build lookup for image sources
    row_to_src = {}
    for row, img_dir in all_rows:
        key = (row[0], row[1])  # (filename, angle)
        row_to_src[key] = img_dir

    # Output
    out_img_dir = os.path.join(output_dir, "images")
    os.makedirs(out_img_dir, exist_ok=True)

    out_csv = os.path.join(output_dir, "driving_log.csv")
    written = 0
    skipped = 0

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "steering_angle", "speed"])

        for row in balanced_rows:
            key = (row[0], row[1])
            src_dir = row_to_src.get(key)
            if src_dir is None:
                skipped += 1
                continue

            src_img = os.path.join(src_dir, row[0])
            dst_img = os.path.join(out_img_dir, row[0])

            if os.path.exists(src_img):
                if not os.path.exists(dst_img):
                    shutil.copy2(src_img, dst_img)
                writer.writerow(row)
                written += 1
            else:
                skipped += 1

    print(f"\n[SAVED] {output_dir}/")
    print(f"        CSV:    {out_csv} ({written} rows)")
    print(f"        Images: {out_img_dir}/")
    if skipped:
        print(f"        Skipped: {skipped} (missing image files)")


def find_sources(base_dir: str = ".") -> list:
    """Find all (csv_path, img_dir) pairs."""
    sources = []

    # my_dataset
    csv_path = os.path.join(base_dir, "my_dataset", "driving_log.csv")
    img_dir  = os.path.join(base_dir, "my_dataset", "images")
    if os.path.exists(csv_path) and os.path.isdir(img_dir):
        sources.append((csv_path, img_dir))

    # saved_routes/*
    routes_dir = os.path.join(base_dir, "saved_routes")
    if os.path.isdir(routes_dir):
        for entry in sorted(os.listdir(routes_dir)):
            entry_path = os.path.join(routes_dir, entry)
            if os.path.isdir(entry_path):
                csv_p = os.path.join(entry_path, "driving_log.csv")
                img_d = os.path.join(entry_path, "images")
                if os.path.exists(csv_p) and os.path.isdir(img_d):
                    sources.append((csv_p, img_d))

    return sources


def main():
    parser = argparse.ArgumentParser(description="Balance driving dataset")
    parser.add_argument("source", nargs="?", default=None,
                        help="Source directory (default: my_dataset)")
    parser.add_argument("--all", action="store_true",
                        help="Merge all sessions (my_dataset + saved_routes)")
    parser.add_argument("--target-straight", type=float, default=0.35,
                        help="Target ratio of straight frames (default: 0.35)")
    parser.add_argument("--output", default="balanced_dataset",
                        help="Output directory name (default: balanced_dataset)")
    args = parser.parse_args()

    print("=" * 50)
    print("  Dataset Balancer")
    print("=" * 50)

    if args.all:
        sources = find_sources()
        if not sources:
            print("[ERROR] No datasets found")
            return
        print(f"[INFO] Found {len(sources)} data sources:")
        for csv_p, _ in sources:
            print(f"       - {csv_p}")
        merge_and_balance(sources, args.output, args.target_straight)

    elif args.source:
        src_dir = args.source
        csv_path = os.path.join(src_dir, "driving_log.csv")
        img_dir  = os.path.join(src_dir, "images")

        if not os.path.exists(csv_path):
            print(f"[ERROR] Not found: {csv_path}")
            return

        sources = [(csv_path, img_dir)]
        merge_and_balance(sources, args.output, args.target_straight)

    else:
        # Default: my_dataset
        csv_path = "my_dataset/driving_log.csv"
        img_dir  = "my_dataset/images"

        if not os.path.exists(csv_path):
            print(f"[ERROR] Not found: {csv_path}")
            return

        sources = [(csv_path, img_dir)]
        merge_and_balance(sources, args.output, args.target_straight)


if __name__ == "__main__":
    main()
