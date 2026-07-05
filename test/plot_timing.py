#!/usr/bin/env python3
"""
Timing Visualization for Object Tracking Performance Data.

Reads a CSV file exported by ObjectTrackingNode and generates
line plots and stacked area charts for each tracker.

Usage:
    python3 plot_timing.py <path_to_csv>
    python3 plot_timing.py tracking_timing_20250308_143022.csv
    python3 plot_timing.py tracking_timing_20250308_143022.csv --output plots/
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Optional: Use a cleaner style if available
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use(
        "seaborn-whitegrid" if "seaborn-whitegrid" in plt.style.available else "default"
    )

# Font sizes tuned for embedding in A4 documents alongside body text.
# At ~14 cm figure width the labels remain crisp and legible.
plt.rcParams.update(
    {
        "font.size": 16,  # base size (legend, annotations)
        "axes.titlesize": 16,  # chart title
        "axes.labelsize": 16,  # x/y axis labels
        "xtick.labelsize": 16,  # tick numbers
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 16,
    }
)


# Phase display names and colors (consistent across all plots)
PHASES = ["predict_us", "associate_us", "update_us", "create_delete_us"]
PHASE_LABELS = {
    "predict_us": "Vorhersage",
    "associate_us": "Assoziation",
    "update_us": "Korrektur",
    "create_delete_us": "Erstellen/Löschen",
}
PHASE_COLORS = {
    "predict_us": "#2196F3",  # Blue
    "associate_us": "#FF9800",  # Orange
    "update_us": "#4CAF50",  # Green
    "create_delete_us": "#9C27B0",  # Purple
}


def slice_frames(data: dict, max_frames: int) -> dict:
    """Return a copy of data limited to the first max_frames entries."""
    if max_frames <= 0:
        return data
    return {key: arr[:max_frames] for key, arr in data.items()}


def read_csv(filepath: str) -> dict:
    """
    Read the timing CSV into a dict of {tracker_name: data_dict}.

    Returns:
        {
            "object": {
                "frame": [1, 2, 3, ...],
                "time_s": [0.0, 0.033, 0.066, ...],  # seconds since first entry
                "predict_us": [12.3, 15.7, ...],
                ...
            },
            "sign": { ... },
            "crossing": { ... },
        }
    """
    import csv

    trackers = {}
    global_t0 = None  # Earliest timestamp across all trackers

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

    # Convert lists to numpy arrays + compute relative time
    for name in trackers:
        trackers[name]["timestamp_raw"] = np.array(trackers[name]["timestamp_raw"])
        trackers[name]["time_s"] = trackers[name]["timestamp_raw"] - global_t0
        del trackers[name]["timestamp_raw"]
        for key in [k for k in trackers[name] if k not in ("time_s",)]:
            trackers[name][key] = np.array(trackers[name][key])

    return trackers


def plot_line_chart(
    data: dict, tracker_name: str, output_dir: str, x_key: str = "frame"
) -> str:
    """
    Create a line chart showing timing of each phase over frames or time.
    Includes total time and a secondary axis for track/detection counts.
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))

    x = data[x_key]
    x_label = "Time (s)" if x_key == "time_s" else "Frame"

    # Plot individual phases
    for phase in PHASES:
        ax1.plot(
            x,
            data[phase],
            label=PHASE_LABELS[phase],
            color=PHASE_COLORS[phase],
            linewidth=0.8,
            alpha=0.85,
        )

    # Plot total as dashed line
    ax1.plot(
        x,
        data["total_us"],
        label="Total",
        color="#E53935",
        linewidth=1.5,
        linestyle="--",
        alpha=0.9,
    )

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Dauer (µs)")
    # ax1.set_title(f"Tracking Performance — {tracker_name.capitalize()} Tracker")
    ax1.legend(loc="upper left")
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(output_dir, f"timing_lines_{tracker_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_stacked_area(
    data: dict,
    tracker_name: str,
    output_dir: str,
    x_key: str = "frame",
    clip_percentile: float = 99.0,  # <-- neu
) -> str:
    fig, ax = plt.subplots(figsize=(12, 5))

    x = data[x_key]
    x_label = "Time (s)" if x_key == "time_s" else "Frame"
    phase_data = [data[phase] for phase in PHASES]
    colors = [PHASE_COLORS[phase] for phase in PHASES]
    labels = [PHASE_LABELS[phase] for phase in PHASES]

    ax.stackplot(x, *phase_data, labels=labels, colors=colors, alpha=0.8)

    # Y-Achse auf Perzentil kappen
    total = data["total_us"]
    y_max = np.percentile(total, clip_percentile)
    ax.set_ylim(bottom=0, top=y_max * 1.05)

    # Anzahl geclippter Frames annotieren
    n_clipped = int(np.sum(total > y_max))
    # if n_clipped > 0:
    #     ax.annotate(
    #         f"{n_clipped} frame(s) exceed axis limit (max {total.max():.0f} µs)",
    #         xy=(0.99, 0.97),
    #         xycoords="axes fraction",
    #         ha="right",
    #         va="top",
    #         fontsize=8,
    #         color="#777777",
    #         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8),
    #     )
    print(
        f"  {tracker_name}: Clipped {n_clipped} frames above {y_max:.1f} µs (max {total.max():.1f} µs)"
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Dauer (µs)")
    # ax.set_title(f"Time Composition — {tracker_name.capitalize()} Tracker")
    ax.legend(loc="upper left")
    ax.set_xlim(x[0], x[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, f"timing_stacked_{tracker_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_summary_stats(trackers: dict, output_dir: str) -> str:
    """
    Create a bar chart comparing average timings across all trackers.
    """
    tracker_names = list(trackers.keys())
    n_trackers = len(tracker_names)

    if n_trackers == 0:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(n_trackers)
    bar_width = 0.18

    for i, phase in enumerate(PHASES):
        means = [np.mean(trackers[name][phase]) for name in tracker_names]
        ax.bar(
            x + i * bar_width,
            means,
            width=bar_width,
            label=PHASE_LABELS[phase],
            color=PHASE_COLORS[phase],
            alpha=0.85,
        )

    ax.set_xlabel("Tracker")
    ax.set_ylabel("Durchschnittliche Dauer (µs)")
    # ax.set_title("Durchschnittliche Phasendauer pro Tracker")
    ax.set_xticks(x + bar_width * (len(PHASES) - 1) / 2)
    ax.set_xticklabels([n.capitalize() for n in tracker_names])
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(output_dir, "timing_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_statistics(trackers: dict):
    """Print a text summary of timing statistics to stdout."""
    print("\n" + "=" * 72)
    print("TIMING STATISTICS SUMMARY")
    print("=" * 72)

    for name, data in trackers.items():
        n_frames = len(data["frame"])
        print(f"\n{'─' * 72}")
        print(f"  {name.upper()} TRACKER  ({n_frames} frames)")
        print(f"{'─' * 72}")
        print(
            f"  {'Phase':<16} {'Mean (µs)':>10} {'Median':>10} {'Max':>10} {'Std':>10}"
        )
        print(f"  {'─'*56}")

        for phase in PHASES:
            vals = data[phase]
            print(
                f"  {PHASE_LABELS[phase]:<16} "
                f"{np.mean(vals):>10.1f} "
                f"{np.median(vals):>10.1f} "
                f"{np.max(vals):>10.1f} "
                f"{np.std(vals):>10.1f}"
            )

        total = data["total_us"]
        print(f"  {'─'*56}")
        print(
            f"  {'Total':<16} "
            f"{np.mean(total):>10.1f} "
            f"{np.median(total):>10.1f} "
            f"{np.max(total):>10.1f} "
            f"{np.std(total):>10.1f}"
        )
        print(
            f"\n  Avg tracks: {np.mean(data['num_tracks']):.1f}  |  "
            f"Avg detections: {np.mean(data['num_detections']):.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot tracking performance timing data."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the timing CSV file exported by ObjectTrackingNode",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for plots (default: folder named after CSV file)",
    )
    parser.add_argument(
        "--time",
        "-t",
        action="store_true",
        help="Use wall-clock time (seconds) on x-axis instead of frame number",
    )
    parser.add_argument(
        "--clip-percentile",
        "-p",
        type=float,
        default=99.0,
        help="Cap y-axis at this percentile of total_us (default: 99)",
    )
    parser.add_argument(
        "--max-frames",
        "-n",
        type=int,
        default=500,
        help="Only plot the first N frames (default: 500, 0 = all)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv_file):
        print(f"Error: File not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    x_key = "time_s" if args.time else "frame"

    # Default: create folder next to CSV with same name (minus extension)
    if args.output:
        output_dir = args.output
    else:
        csv_abs = os.path.abspath(args.csv_file)
        output_dir = os.path.splitext(csv_abs)[0]

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading: {args.csv_file}")
    trackers = read_csv(args.csv_file)

    if not trackers:
        print("Error: No data found in CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"Found trackers: {', '.join(trackers.keys())}")
    print(f"X-axis: {'Wall-clock time (s)' if args.time else 'Frame number'}")

    # Print text statistics
    print_statistics(trackers)

    # Generate plots
    generated = []
    for name, data in trackers.items():
        data = slice_frames(data, args.max_frames)
        if len(data["frame"]) < 2:
            print(f"Skipping {name}: not enough data points")
            continue
        generated.append(plot_line_chart(data, name, output_dir, x_key))
        generated.append(
            plot_stacked_area(data, name, output_dir, x_key, args.clip_percentile)
        )

    if len(trackers) > 1:
        summary = plot_summary_stats(trackers, output_dir)
        if summary:
            generated.append(summary)

    print(f"\n{'=' * 72}")
    print(f"Generated {len(generated)} plots in: {output_dir}")
    for path in generated:
        print(f"  → {path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
