"""
tools/analyze_data.py
=====================
Analyze steering angle distribution of a driving dataset.
Shows per-bin counts and visual histogram to identify data imbalance.

Usage
-----
    python3 tools/analyze_data.py                          # default: my_dataset
    python3 tools/analyze_data.py saved_routes/my_route    # specific session
    python3 tools/analyze_data.py --all                    # all sessions combined
"""

import csv
import os
import sys
import glob


def analyze_csv(csv_path: str) -> dict:
    """Read a driving_log.csv and return angle distribution."""
    counts = {}
    total = 0

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                angle = float(row[1])
            except ValueError:
                continue
            # Round to 1 decimal for grouping
            angle = round(angle, 1)
            counts[angle] = counts.get(angle, 0) + 1
            total += 1

    return {"counts": counts, "total": total, "path": csv_path}


def print_distribution(result: dict, label: str = ""):
    """Pretty-print a steering angle distribution."""
    counts = result["counts"]
    total  = result["total"]

    if total == 0:
        print(f"  {label}: No data found")
        return

    print(f"\n{'=' * 60}")
    if label:
        print(f"  {label}")
    print(f"  Source: {result['path']}")
    print(f"  Total frames: {total:,}")
    print(f"{'=' * 60}")

    # Categorize
    straight = sum(v for k, v in counts.items() if -0.05 < k < 0.05)
    left     = sum(v for k, v in counts.items() if k <= -0.05)
    right    = sum(v for k, v in counts.items() if k >= 0.05)

    print(f"\n  Category breakdown:")
    print(f"    Straight (|a|<0.05):  {straight:>6}  ({straight/total*100:5.1f}%)")
    print(f"    Left     (a<-0.05):   {left:>6}  ({left/total*100:5.1f}%)")
    print(f"    Right    (a>+0.05):   {right:>6}  ({right/total*100:5.1f}%)")

    # Detailed histogram
    print(f"\n  Detailed distribution:")
    max_count = max(counts.values()) if counts else 1
    bar_width = 30

    for angle in sorted(counts.keys()):
        count = counts[angle]
        pct   = count / total * 100
        bar   = "█" * int(count / max_count * bar_width)
        print(f"    {angle:+5.1f}: {bar:<{bar_width}} {count:>6} ({pct:5.1f}%)")

    # Health check
    print(f"\n  Health check:")
    if straight / total > 0.6:
        print(f"    ⚠️  IMBALANCED: {straight/total*100:.0f}% straight frames!")
        print(f"        Recommend: use WeightedRandomSampler or collect more turn data")
    elif straight / total > 0.4:
        print(f"    ⚡ Slightly heavy on straight ({straight/total*100:.0f}%)")
    else:
        print(f"    ✅ Good balance ({straight/total*100:.0f}% straight)")

    unique = len(counts)
    if unique <= 11:
        print(f"    ⚠️  Only {unique} unique angle values (discrete)")
        print(f"        Recommend: use STEER_STEP=2 for smoother data")
    else:
        print(f"    ✅ {unique} unique angle values (smooth)")


def find_all_csvs(base_dir: str) -> list:
    """Find all driving_log.csv files recursively."""
    csvs = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == "driving_log.csv":
                csvs.append(os.path.join(root, f))
    return sorted(csvs)


def main():
    default_csv = "my_dataset/driving_log.csv"
    routes_dir  = "saved_routes"

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "--all":
            # Analyze all datasets
            all_csvs = []
            if os.path.exists(default_csv):
                all_csvs.append(default_csv)
            all_csvs.extend(find_all_csvs(routes_dir))

            if not all_csvs:
                print("[ERROR] No driving_log.csv files found")
                return

            # Individual analysis
            combined_counts = {}
            combined_total = 0
            for csv_path in all_csvs:
                result = analyze_csv(csv_path)
                label = os.path.dirname(csv_path)
                print_distribution(result, label)

                for k, v in result["counts"].items():
                    combined_counts[k] = combined_counts.get(k, 0) + v
                combined_total += result["total"]

            # Combined summary
            if len(all_csvs) > 1:
                combined = {"counts": combined_counts, "total": combined_total,
                            "path": f"{len(all_csvs)} files combined"}
                print_distribution(combined, "COMBINED (all sessions)")

        elif arg == "--help" or arg == "-h":
            print(__doc__)
        else:
            # Analyze specific path
            if os.path.isdir(arg):
                csv_path = os.path.join(arg, "driving_log.csv")
            else:
                csv_path = arg

            if not os.path.exists(csv_path):
                print(f"[ERROR] Not found: {csv_path}")
                return

            result = analyze_csv(csv_path)
            print_distribution(result, arg)
    else:
        # Default: analyze my_dataset
        if not os.path.exists(default_csv):
            print(f"[ERROR] Not found: {default_csv}")
            print("Usage: python3 tools/analyze_data.py [path|--all]")
            return

        result = analyze_csv(default_csv)
        print_distribution(result, "my_dataset")


if __name__ == "__main__":
    main()
