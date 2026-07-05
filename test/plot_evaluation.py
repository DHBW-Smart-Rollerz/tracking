#!/usr/bin/env python3
"""
Vergleich Detection vs. Tracking: F1-Score und mittlerer Pixelfehler.

Verwendung:
  python3 plot_evaluation.py eval_detection.json eval_tracking.json
  python3 plot_evaluation.py eval_detection.json eval_tracking.json --out plots/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ── Daten laden ──────────────────────────────────────────────────────────────


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_metric(data, metric):
    """Gibt {class_name: wert} zurück. None-Werte → 0."""
    return {
        r["class_name"]: (r.get(metric) or 0.0)
        for r in data["per_class"]
        if str(r.get("class_id")) != "TOTAL"
    }


def get_metric_nullable(data, metric):
    """Gibt {class_name: wert | None} zurück. None bleibt erhalten (= keine Matches)."""
    return {
        r["class_name"]: r.get(metric)
        for r in data["per_class"]
        if str(r.get("class_id")) != "TOTAL"
    }


def all_classes(runs_data):
    names = set()
    for data in runs_data:
        for r in data["per_class"]:
            if str(r.get("class_id")) != "TOTAL":
                names.add(r["class_name"])
    return sorted(names)


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot_bars(ax, classes, runs_data, labels, metric, ylabel, title):
    """
    Für jede Klasse stehen zwei Balken nebeneinander (Detection / Tracking).
    Jede Klasse hat ihre eigene Farbe; der zweite Lauf wird durch Schraffur unterschieden.
    """
    # Nur Klassen anzeigen, bei denen mindestens ein Run einen Wert > 0 hat
    classes = [
        cls
        for cls in classes
        if any(get_metric(data, metric).get(cls, 0.0) > 0 for data in runs_data)
    ]

    n_classes = len(classes)
    n_runs = len(runs_data)
    bar_w = 0.35
    gap = 0.20

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_classes)]

    group_w = n_runs * bar_w
    group_pos = np.arange(n_classes) * (group_w + gap)

    for run_i, (data, label) in enumerate(zip(runs_data, labels)):
        vals = get_metric(data, metric)
        for cls_i, cls in enumerate(classes):
            x = group_pos[cls_i] + run_i * bar_w
            v = vals.get(cls, 0.0)

            hatch = None if run_i == 0 else "////"
            ax.bar(
                x,
                v,
                width=bar_w,
                color=colors[cls_i],
                alpha=0.85 if run_i == 0 else 0.50,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

    tick_pos = group_pos + (n_runs - 1) * bar_w / 2
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=24)

    ax.set_ylabel(ylabel, fontsize=24)
    ax.tick_params(axis="y", labelsize=24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    if metric in ("f1", "precision", "recall"):
        ax.set_ylim(0, 1.12)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

    legend_handles = [
        Patch(facecolor="gray", alpha=0.85, edgecolor="white", label=labels[0]),
        Patch(
            facecolor="gray",
            alpha=0.50,
            edgecolor="white",
            hatch="////",
            label=labels[1],
        ),
    ]
    ax.legend(handles=legend_handles, fontsize=24, loc="upper right")


# ── TP / FP / FN – gestapeltes Balkendiagramm ────────────────────────────────


def plot_counts(ax, classes, runs_data, labels):
    """
    Für jede Klasse stehen zwei gestapelte Balken nebeneinander (Detection / Tracking).
    Jeder Balken zeigt TP (unten), FN (mitte), FP (oben) in festen Farben.
    Klassen ohne einzigen TP/FP/FN-Eintrag > 0 werden übersprungen.
    """
    COUNT_KEYS = ["tp", "fn", "fp"]
    COUNT_COLORS = {"tp": "#4CAF50", "fn": "#FF9800", "fp": "#F44336"}
    COUNT_LABELS = {"tp": "TP", "fn": "FN", "fp": "FP"}

    # Nur Klassen mit mindestens einem TP in einem der Runs
    classes = [
        cls
        for cls in classes
        if any(get_metric(data, "tp").get(cls, 0.0) > 0 for data in runs_data)
    ]

    n_classes = len(classes)
    n_runs = len(runs_data)
    bar_w = 0.35
    gap = 0.30

    group_w = n_runs * bar_w
    group_pos = np.arange(n_classes) * (group_w + gap)

    for run_i, (data, label) in enumerate(zip(runs_data, labels)):
        hatch = None if run_i == 0 else "////"
        alpha = 0.90 if run_i == 0 else 0.55
        bottoms = np.zeros(n_classes)

        for key in COUNT_KEYS:
            vals = get_metric(data, key)
            heights = np.array([vals.get(cls, 0.0) for cls in classes])
            xs = group_pos + run_i * bar_w

            ax.bar(
                xs,
                heights,
                width=bar_w,
                bottom=bottoms,
                color=COUNT_COLORS[key],
                alpha=alpha,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                label=COUNT_LABELS[key] if run_i == 0 else None,
            )
            bottoms += heights

    tick_pos = group_pos + (n_runs - 1) * bar_w / 2
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=24)

    ax.set_ylabel("Anzahl", fontsize=24)
    ax.tick_params(axis="y", labelsize=24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Legende: TP/FN/FP-Farben + Detection/Tracking-Muster
    color_handles = [
        Patch(facecolor=COUNT_COLORS[k], label=COUNT_LABELS[k]) for k in COUNT_KEYS
    ]
    run_handles = [
        Patch(facecolor="gray", alpha=0.90, edgecolor="white", label=labels[0]),
        Patch(
            facecolor="gray",
            alpha=0.55,
            edgecolor="white",
            hatch="////",
            label=labels[1],
        ),
    ]
    ax.legend(
        handles=color_handles + run_handles,
        fontsize=24,
        loc="upper right",
        ncol=2,
    )


# ── Mittlerer Pixelfehler ─────────────────────────────────────────────────────


def plot_pixel_error(ax, classes, runs_data, labels):
    """
    Balkendiagramm des mittleren Pixelfehlers (mean_err_px) pro Klasse.

    Nur Klassen, bei denen mindestens ein Run einen echten Messwert (≠ None)
    hat, werden angezeigt.  Balken ohne Messwert (None = keine gematchten
    Paare) werden als leerer, gestrichelter Rahmen mit dem Hinweis „keine
    Matches" dargestellt, damit die Information nicht schweigend fehlt.
    """
    # Nur Klassen mit mindestens einem echten Wert zeigen
    classes = [
        cls
        for cls in classes
        if any(
            get_metric_nullable(data, "mean_err_px").get(cls) is not None
            for data in runs_data
        )
    ]

    if not classes:
        ax.text(
            0.5,
            0.5,
            "Keine Pixelfehler-Daten vorhanden",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=18,
            color="gray",
        )
        return

    n_classes = len(classes)
    n_runs = len(runs_data)
    bar_w = 0.35
    gap = 0.20

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_classes)]

    group_w = n_runs * bar_w
    group_pos = np.arange(n_classes) * (group_w + gap)

    for run_i, (data, label) in enumerate(zip(runs_data, labels)):
        vals = get_metric(data, "mean_err_px")
        hatch = None if run_i == 0 else "////"
        alpha_fill = 0.85 if run_i == 0 else 0.50

        for cls_i, cls in enumerate(classes):
            x = group_pos[cls_i] + run_i * bar_w
            v = vals.get(cls, 0.0)

            ax.bar(
                x,
                v,
                width=bar_w,
                color=colors[cls_i],
                alpha=alpha_fill,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

    tick_pos = group_pos + (n_runs - 1) * bar_w / 2
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=24)

    ax.set_ylabel("Mittlerer Pixelfehler [px]", fontsize=24)
    ax.tick_params(axis="y", labelsize=24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # y-Achse etwas über das Maximum ausdehnen
    all_vals = [
        get_metric(data, "mean_err_px").get(cls, 0.0)
        for data in runs_data
        for cls in classes
    ]
    if any(v > 0 for v in all_vals):
        ax.set_ylim(0, max(all_vals) * 1.15)

    legend_handles = [
        Patch(facecolor="gray", alpha=0.85, edgecolor="white", label=labels[0]),
        Patch(
            facecolor="gray",
            alpha=0.50,
            edgecolor="white",
            hatch="////",
            label=labels[1],
        ),
    ]
    ax.legend(handles=legend_handles, fontsize=20, loc="upper right")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files", nargs=2, metavar="JSON", help="Detection-JSON und Tracking-JSON"
    )
    parser.add_argument(
        "--out", default=None, help="Ausgabe-Ordner (Default: Ordner der ersten Datei)"
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        default=None,
        metavar=("LABEL_A", "LABEL_B"),
        help="Legenden-Labels (Default: Modus aus JSON-Meta)",
    )
    args = parser.parse_args()

    runs_data = [load(p) for p in args.files]
    classes = all_classes(runs_data)

    if not classes:
        print("Keine Klassen in den JSON-Dateien gefunden.", file=sys.stderr)
        sys.exit(1)

    labels = args.labels or [
        d.get("meta", {}).get("mode", Path(p).stem).upper()
        for d, p in zip(runs_data, args.files)
    ]

    out_dir = args.out or os.path.dirname(os.path.abspath(args.files[0]))
    os.makedirs(out_dir, exist_ok=True)
    stems = [Path(p).stem for p in args.files]

    metrics = [
        ("f1", "F1-Score", "F1-Score pro Klasse"),
        ("precision", "Precision", "Precision pro Klasse"),
        ("recall", "Recall", "Recall pro Klasse"),
    ]

    for metric, ylabel, title in metrics:
        fig, ax = plt.subplots(
            figsize=(max(10, len(classes) * 1.8), 6),
            constrained_layout=True,
        )
        plot_bars(
            ax, classes, runs_data, labels, metric=metric, ylabel=ylabel, title=title
        )

        out_path = os.path.join(out_dir, f"{'__vs__'.join(stems)}__{metric}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Gespeichert: {out_path}")
        plt.close(fig)

    # ── TP / FP / FN ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(max(10, len(classes) * 1.8), 6),
        constrained_layout=True,
    )
    plot_counts(ax, classes, runs_data, labels)

    out_path = os.path.join(out_dir, f"{'__vs__'.join(stems)}__counts.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Gespeichert: {out_path}")
    plt.close(fig)

    # ── Mittlerer Pixelfehler ──────────────────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(max(10, len(classes) * 1.8), 6),
        constrained_layout=True,
    )
    plot_pixel_error(ax, classes, runs_data, labels)

    out_path = os.path.join(out_dir, f"{'__vs__'.join(stems)}__pixel_error.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Gespeichert: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
