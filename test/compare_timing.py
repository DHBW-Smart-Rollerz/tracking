#!/usr/bin/env python3
"""
Compare two tracking timing recordings side by side.

Generates one line chart per tracker (object, sign, crossing) with both
recordings overlaid using distinct line styles.

Usage:
    python3 compare_timing.py recording_a.csv recording_b.csv
    python3 compare_timing.py before.csv after.csv --time
    python3 compare_timing.py before.csv after.csv --labels "Before Optimization" "After Optimization"
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use(
        "seaborn-whitegrid" if "seaborn-whitegrid" in plt.style.available else "default"
    )


# Phase display names
PHASES = ["predict_us", "associate_us", "update_us", "create_delete_us"]
PHASE_LABELS = {
    "predict_us": "Predict",
    "associate_us": "Associate",
    "update_us": "Update",
    "create_delete_us": "Create/Delete",
}


def read_csv(filepath: str) -> dict:
    """Read timing CSV into {tracker_name: data_dict}."""
    import csv

    trackers = {}
    global_t0 = None

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["tracker"]
            if name not in trackers:
                trackers[name] = {
                    key: []
                    for key in [
                        "frame",
                        "timestamp_raw",
                        "predict_us",
                        "associate_us",
                        "update_us",
                        "create_delete_us",
                        "total_us",
                        "num_tracks",
                        "num_detections",
                    ]
                }

            data = trackers[name]
            data["frame"].append(int(row["frame"]))
            ts = float(row["timestamp"])
            data["timestamp_raw"].append(ts)
            if global_t0 is None or ts < global_t0:
                global_t0 = ts
            for key in [
                "predict_us",
                "associate_us",
                "update_us",
                "create_delete_us",
                "total_us",
            ]:
                data[key].append(float(row[key]))
            data["num_tracks"].append(int(row["num_tracks"]))
            data["num_detections"].append(int(row["num_detections"]))

    for name in trackers:
        trackers[name]["timestamp_raw"] = np.array(trackers[name]["timestamp_raw"])
        trackers[name]["time_s"] = trackers[name]["timestamp_raw"] - global_t0
        del trackers[name]["timestamp_raw"]
        for key in [k for k in trackers[name] if k not in ("time_s",)]:
            trackers[name][key] = np.array(trackers[name][key])

    return trackers


def plot_comparison(
    data_a: dict,
    data_b: dict,
    tracker_name: str,
    output_dir: str,
    label_a: str,
    label_b: str,
    x_key: str = "frame",
) -> str:
    """
    Create a comparison line chart for one tracker with two recordings overlaid.
    Shows only the total duration per recording for clarity.
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    x_label = "Time (s)" if x_key == "time_s" else "Frame"

    x_a = data_a[x_key]
    x_b = data_b[x_key]

    # One total line per recording
    ax1.plot(
        x_a,
        data_a["total_us"],
        color="#1565C0",
        linewidth=1.2,
        alpha=0.85,
        label=label_a,
    )
    ax1.plot(
        x_b,
        data_b["total_us"],
        color="#E65100",
        linewidth=1.2,
        alpha=0.85,
        linestyle="--",
        label=label_b,
    )

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Duration (µs)")
    ax1.set_title(f"Comparison — {tracker_name.capitalize()} Tracker")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left")

    # Secondary axis for track counts
    ax2 = ax1.twinx()
    ax2.plot(
        x_a,
        data_a["num_tracks"],
        color="#607D8B",
        linewidth=0.8,
        linestyle=":",
        alpha=0.5,
        label=f"Tracks ({label_a})",
    )
    ax2.plot(
        x_b,
        data_b["num_tracks"],
        color="#90A4AE",
        linewidth=0.8,
        linestyle=":",
        alpha=0.5,
        label=f"Tracks ({label_b})",
    )
    ax2.set_ylabel("Active Tracks")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, f"compare_{tracker_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_comparison_stats(
    trackers_a: dict, trackers_b: dict, label_a: str, label_b: str
):
    """Print side-by-side statistics for both recordings."""
    all_names = sorted(set(trackers_a.keys()) | set(trackers_b.keys()))

    print("\n" + "=" * 80)
    print("COMPARISON STATISTICS")
    print("=" * 80)

    for name in all_names:
        has_a = name in trackers_a
        has_b = name in trackers_b

        print(f"\n{'─' * 80}")
        print(f"  {name.upper()} TRACKER")
        if has_a:
            print(f"    {label_a}: {len(trackers_a[name]['frame'])} frames")
        if has_b:
            print(f"    {label_b}: {len(trackers_b[name]['frame'])} frames")
        print(f"{'─' * 80}")

        header = f"  {'Phase':<16}"
        if has_a:
            header += f" {'Mean A (µs)':>12} {'Max A':>10}"
        if has_b:
            header += f" {'Mean B (µs)':>12} {'Max B':>10}"
        if has_a and has_b:
            header += f" {'Δ Mean':>10} {'Δ %':>8}"
        print(header)
        print(f"  {'─' * (len(header) - 2)}")

        for phase in PHASES + ["total_us"]:
            label = PHASE_LABELS.get(phase, "Total")
            line = f"  {label:<16}"

            mean_a = np.mean(trackers_a[name][phase]) if has_a else None
            max_a = np.max(trackers_a[name][phase]) if has_a else None
            mean_b = np.mean(trackers_b[name][phase]) if has_b else None
            max_b = np.max(trackers_b[name][phase]) if has_b else None

            if has_a:
                line += f" {mean_a:>12.1f} {max_a:>10.1f}"
            if has_b:
                line += f" {mean_b:>12.1f} {max_b:>10.1f}"
            if has_a and has_b:
                delta = mean_b - mean_a
                pct = (delta / mean_a * 100) if mean_a > 0 else 0
                sign = "+" if delta > 0 else ""
                line += f" {sign}{delta:>9.1f} {sign}{pct:>6.1f}%"

            print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two tracking timing recordings."
    )
    parser.add_argument(
        "csv_a",
        help="First CSV file (e.g. before optimization)",
    )
    parser.add_argument(
        "csv_b",
        help="Second CSV file (e.g. after optimization)",
    )
    parser.add_argument(
        "--labels",
        "-l",
        nargs=2,
        default=None,
        help="Labels for the two recordings (default: filenames). "
        'Example: --labels "Before" "After"',
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory (default: folder named compare_<a>_vs_<b>)",
    )
    parser.add_argument(
        "--time",
        "-t",
        action="store_true",
        help="Use wall-clock time (seconds) on x-axis instead of frame number",
    )
    args = parser.parse_args()

    for f in [args.csv_a, args.csv_b]:
        if not os.path.isfile(f):
            print(f"Error: File not found: {f}", file=sys.stderr)
            sys.exit(1)

    # Default labels: filename without extension
    if args.labels:
        label_a, label_b = args.labels
    else:
        label_a = os.path.splitext(os.path.basename(args.csv_a))[0]
        label_b = os.path.splitext(os.path.basename(args.csv_b))[0]

    # Default output dir
    if args.output:
        output_dir = args.output
    else:
        name_a = os.path.splitext(os.path.basename(args.csv_a))[0]
        name_b = os.path.splitext(os.path.basename(args.csv_b))[0]
        parent = os.path.dirname(os.path.abspath(args.csv_a))
        output_dir = os.path.join(parent, f"compare_{name_a}_vs_{name_b}")

    os.makedirs(output_dir, exist_ok=True)
    x_key = "time_s" if args.time else "frame"

    print(f"Reading A: {args.csv_a}  ({label_a})")
    trackers_a = read_csv(args.csv_a)
    print(f"Reading B: {args.csv_b}  ({label_b})")
    trackers_b = read_csv(args.csv_b)

    print(f"X-axis: {'Wall-clock time (s)' if args.time else 'Frame number'}")

    # Statistics
    print_comparison_stats(trackers_a, trackers_b, label_a, label_b)

    # Generate plots for trackers present in both
    common = sorted(set(trackers_a.keys()) & set(trackers_b.keys()))
    if not common:
        print("\nWarning: No common trackers found between the two files.")
        sys.exit(1)

    generated = []
    for name in common:
        if len(trackers_a[name]["frame"]) < 2 or len(trackers_b[name]["frame"]) < 2:
            print(f"Skipping {name}: not enough data")
            continue

        path = plot_comparison(
            trackers_a[name],
            trackers_b[name],
            name,
            output_dir,
            label_a,
            label_b,
            x_key,
        )
        generated.append(path)

    print(f"\n{'=' * 80}")
    print(f"Generated {len(generated)} plots in: {output_dir}")
    for path in generated:
        print(f"  → {path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
