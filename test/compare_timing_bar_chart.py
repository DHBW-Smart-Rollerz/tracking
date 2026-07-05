#!/usr/bin/env python3
"""
Compare timing data from two tracking CSV files.

Generates one combined A4-landscape chart with object and sign trackers side by side.

Usage:
    python3 compare_timing.py <csv_a> <csv_b> [--labels A B] [--output dir/]
"""

import argparse
import csv
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

PHASES = ["predict_us", "associate_us", "update_us", "create_delete_us"]
PHASE_LABELS = {
    "predict_us": "Vorhersage",
    "associate_us": "Assoziation",
    "update_us": "Korrektur",
    "create_delete_us": "Erstellen/\nLöschen",
}

# Two shades per phase: darker = run A, lighter = run B
PHASE_COLORS = {
    "predict_us": ("#1565C0", "#90CAF9"),
    "associate_us": ("#E65100", "#FFCC80"),
    "update_us": ("#2E7D32", "#A5D6A7"),
    "create_delete_us": ("#6A1B9A", "#CE93D8"),
}


def read_csv(filepath: str) -> dict:
    """Return {tracker_name: {phase: np.array, ...}}."""
    trackers: dict = {}
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["tracker"]
            if name not in trackers:
                trackers[name] = {p: [] for p in PHASES + ["total_us"]}
            for p in PHASES + ["total_us"]:
                trackers[name][p].append(float(row[p]))
    for name in trackers:
        for p in PHASES + ["total_us"]:
            trackers[name][p] = np.array(trackers[name][p])
    return trackers


def plot_comparison(
    trackers_a: dict,
    trackers_b: dict,
    label_a: str,
    label_b: str,
    output_dir: str,
) -> list:
    """
    Combined A4-landscape figure with two subplots side by side:
    left = object tracker (klassenuebergreifend), right = sign tracker (klassenspezifisch).
    """
    from matplotlib.patches import Patch

    TRACKER_TITLES = {
        "object": "Dynamische Objekte",
        "sign": "Statische Objekte",
    }

    wanted = ["object", "sign"]
    trackers_to_plot = [t for t in wanted if t in trackers_a and t in trackers_b]

    if not trackers_to_plot:
        print("Warning: neither 'object' nor 'sign' tracker found in both CSVs.")
        return []

    # Font sizes tuned for A4 landscape at 150 dpi
    TITLE_FS = 15
    SUBTITLE_FS = 13
    LABEL_FS = 14
    TICK_FS = 14
    LEGEND_FS = 14

    # A4 landscape in inches
    fig, axes = plt.subplots(
        1,
        len(trackers_to_plot),
        figsize=(11.69, 8.27),
        sharey=False,
    )
    if len(trackers_to_plot) == 1:
        axes = [axes]

    bar_width = 0.35
    x = np.arange(len(PHASES))

    for ax, tracker in zip(axes, trackers_to_plot):
        means_a = [np.mean(trackers_a[tracker][p]) for p in PHASES]
        means_b = [np.mean(trackers_b[tracker][p]) for p in PHASES]

        for i, phase in enumerate(PHASES):
            color_a, color_b = PHASE_COLORS[phase]
            ax.bar(x[i] - bar_width / 2, means_a[i], width=bar_width, color=color_a)
            ax.bar(x[i] + bar_width / 2, means_b[i], width=bar_width, color=color_b)

        ax.set_xticks(x)
        ax.set_xticklabels([PHASE_LABELS[p] for p in PHASES], fontsize=TICK_FS)
        for tick, phase in zip(ax.get_xticklabels(), PHASES):
            tick.set_color(PHASE_COLORS[phase][0])

        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.set_ylabel("Durchschnittliche Dauer (µs)", fontsize=LABEL_FS)
        ax.set_title(
            TRACKER_TITLES.get(tracker, tracker.capitalize()), fontsize=SUBTITLE_FS
        )
        ax.set_ylim(bottom=0)

    # Shared legend at bottom center
    legend_handles = [
        Patch(facecolor="#455A64", label=label_a),
        Patch(facecolor="#B0BEC5", label=label_b),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=LEGEND_FS,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        "Vergleich der Phasendauer für klassenübergreifende\nund klassenspezifische Assoziation",
        fontsize=TITLE_FS,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    out_path = os.path.join(output_dir, "compare_object_sign.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return [out_path]


def print_statistics(trackers_a, trackers_b, label_a, label_b):
    common = sorted(set(trackers_a) & set(trackers_b))
    print("\n" + "=" * 80)
    print("COMPARISON STATISTICS")
    print("=" * 80)
    for tracker in common:
        print(f"\n{'─' * 80}")
        print(f"  {tracker.upper()} TRACKER")
        print(f"{'─' * 80}")
        print(
            f"  {'Phase':<16} {'Mean A':>10} {'Mean B':>10} {'Δ (µs)':>10} {'Δ (%)':>8}"
        )
        print(f"  {'─' * 58}")
        for phase in PHASES:
            ma = np.mean(trackers_a[tracker][phase])
            mb = np.mean(trackers_b[tracker][phase])
            delta = mb - ma
            pct = (delta / ma * 100) if ma > 0 else float("nan")
            sign = "+" if delta >= 0 else ""
            print(
                f"  {PHASE_LABELS[phase]:<16} "
                f"{ma:>10.1f} "
                f"{mb:>10.1f} "
                f"{sign}{delta:>9.1f} "
                f"{sign}{pct:>7.1f}%"
            )
        ta = np.mean(trackers_a[tracker]["total_us"])
        tb = np.mean(trackers_b[tracker]["total_us"])
        dt = tb - ta
        dp = (dt / ta * 100) if ta > 0 else float("nan")
        sign = "+" if dt >= 0 else ""
        print(f"  {'─' * 58}")
        print(
            f"  {'Total':<16} "
            f"{ta:>10.1f} "
            f"{tb:>10.1f} "
            f"{sign}{dt:>9.1f} "
            f"{sign}{dp:>7.1f}%"
        )
    print(f"\n  A = {label_a}")
    print(f"  B = {label_b}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare tracking timing from two CSV files."
    )
    parser.add_argument("csv_a", help="First CSV file (baseline)")
    parser.add_argument("csv_b", help="Second CSV file (comparison)")
    parser.add_argument(
        "--labels",
        "-l",
        nargs=2,
        default=["Run A", "Run B"],
        metavar=("LABEL_A", "LABEL_B"),
        help="Display labels for the two runs (default: 'Run A' 'Run B')",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory (default: 'compare_<stem_a>_vs_<stem_b>')",
    )
    args = parser.parse_args()

    for path in (args.csv_a, args.csv_b):
        if not os.path.isfile(path):
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)

    label_a, label_b = args.labels

    if args.output:
        output_dir = args.output
    else:
        stem_a = os.path.splitext(os.path.basename(args.csv_a))[0]
        stem_b = os.path.splitext(os.path.basename(args.csv_b))[0]
        output_dir = f"compare_{stem_a}_vs_{stem_b}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading A: {args.csv_a}")
    trackers_a = read_csv(args.csv_a)
    print(f"Reading B: {args.csv_b}")
    trackers_b = read_csv(args.csv_b)

    print(f"Trackers in A: {', '.join(sorted(trackers_a))}")
    print(f"Trackers in B: {', '.join(sorted(trackers_b))}")

    print_statistics(trackers_a, trackers_b, label_a, label_b)

    generated = plot_comparison(trackers_a, trackers_b, label_a, label_b, output_dir)

    print(f"\n{'=' * 80}")
    print(f"Generated {len(generated)} plot(s) in: {output_dir}")
    for p in generated:
        print(f"  → {p}")
    print("=" * 80)


if __name__ == "__main__":
    main()
