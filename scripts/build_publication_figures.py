#!/usr/bin/env python3
"""Build the evidence figures used by the public reproduction report."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "reports" / "rynnbrain-reproduction" / "images"
RYNN = "#2463a6"
QWEN = "#e76f51"
ACCENT = "#2a9d8f"
INK = "#22313f"
GRID = "#d9e2ec"


def finish(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def protocol_adherence() -> None:
    tasks = ["Contact pose\n(native prompt)", "Official\nlocalization", "Native 3D\n(zero-shot)"]
    rynn = np.array([8, 7, 7])
    qwen = np.array([0, 3, 2])
    x = np.arange(len(tasks))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars_r = ax.bar(x - width / 2, 100 * rynn / 8, width, label="RynnBrain 1.1 122B", color=RYNN)
    bars_q = ax.bar(x + width / 2, 100 * qwen / 8, width, label="Matched Qwen 122B", color=QWEN)
    for bars, vals in ((bars_r, rynn), (bars_q, qwen)):
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{value}/8", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Strict protocol-valid cases (%)")
    ax.set_xticks(x, tasks)
    ax.set_title("RynnBrain more reliably follows released embodied interfaces")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "Eight released cases per task; terminal Kubernetes summaries only.", color="#52606d", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    finish(fig, "01_protocol_adherence.png")


def contact_robustness() -> None:
    labels = ["Baseline", "0.8× inset", "1.25× crop", "+10% x", "90°", "180°", "Gray", "Invert", "Blur", "White", "Black"]
    r_pos = [56.52, 36.28, 73.75, 70.74, 70.70, 56.21, 61.29, 58.40, 47.38, 119.53, 110.77]
    q_pos = [64.68, 53.78, 77.21, 64.06, 46.31, 60.52, 64.11, 57.48, 66.39, 153.39, 163.52]
    r_ang = [7.74, 14.68, 11.38, 8.80, 28.74, 21.33, 17.25, 15.33, 24.81, 39.96, 49.24]
    q_ang = [24.04, 45.01, 20.56, 34.84, 36.19, 47.16, 27.54, 39.81, 47.81, 44.16, 49.30]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 7.4), sharex=True)
    for ax, rv, qv, ylabel in (
        (axes[0], r_pos, q_pos, "Position error (pixels) ↓"),
        (axes[1], r_ang, q_ang, "Symmetry-aware angle error (°) ↓"),
    ):
        ax.plot(x, rv, "o-", color=RYNN, linewidth=2.2, label="RynnBrain 122B")
        ax.plot(x, qv, "s--", color=QWEN, linewidth=2.0, label="Qwen 122B")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title("Contact-pose robustness: geometry and orientation respond differently")
    axes[1].set_xticks(x, labels, rotation=28, ha="right")
    fig.text(0.01, 0.01, "Errors are measured against analytically transformed recorded generations; lower is better.", color="#52606d", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    finish(fig, "02_contact_robustness.png")


def checkpoint_scale() -> None:
    sizes = ["2B", "9B", "122B"]
    r_valid = [8, 7, 7]
    q_valid = [1, 2, 3]
    r_iou = [0.961, 0.723, 0.712]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    width = 0.34
    axes[0].bar(x - width / 2, r_valid, width, color=RYNN, label="RynnBrain")
    axes[0].bar(x + width / 2, q_valid, width, color=QWEN, label="Qwen")
    axes[0].set_xticks(x, sizes)
    axes[0].set_ylim(0, 8.8)
    axes[0].set_ylabel("Strict-valid cases (of 8)")
    axes[0].set_title("90° localization validity")
    axes[0].legend(frameon=False)
    for xpos, value in zip(x - width / 2, r_valid):
        axes[0].text(xpos, value + 0.15, str(value), ha="center", fontsize=10)
    for xpos, value in zip(x + width / 2, q_valid):
        axes[0].text(xpos, value + 0.15, str(value), ha="center", fontsize=10)
    axes[1].plot(x, r_iou, "o-", color=ACCENT, linewidth=2.4, markersize=8)
    axes[1].set_xticks(x, sizes)
    axes[1].set_ylim(0.65, 1.0)
    axes[1].set_ylabel("Rynn mean box IoU ↑")
    axes[1].set_title("Larger checkpoints are not monotonic")
    for xpos, value in zip(x, r_iou):
        axes[1].text(xpos, value + 0.012, f"{value:.3f}", ha="center", fontsize=10)
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Checkpoint-scale robustness on the exact 90° transform", y=1.03, fontsize=14, weight="bold")
    fig.tight_layout()
    finish(fig, "03_checkpoint_scale.png")


def campaign_runtime() -> None:
    counts = np.array([316, 15, 6])
    labels = ["Successful", "Cancelled", "Failed"]
    colors = [ACCENT, "#f4a261", "#c44536"]
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(10.4, 4.7))
    left = 0
    for count, label, color in zip(counts, labels, colors):
        ax.barh([0], [count], left=left, color=color, height=0.48, label=f"{label}: {count}")
        if count >= 10:
            ax.text(left + count / 2, 0, str(count), ha="center", va="center", color="white", fontsize=12, weight="bold")
        left += count
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel("Terminal Kubernetes runs")
    ax.set_title("Campaign closure: 316 measurement-bearing successes in 11.8785 hours")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.19))
    ax.text(0.01, 0.72, "26.60 successful runs / observed campaign wall-hour", transform=ax.transAxes, color=INK, fontsize=12, weight="bold")
    ax.text(0.01, 0.61, "Maximum concurrent allocation: 16 GPUs", transform=ax.transAxes, color="#52606d", fontsize=10)
    ax.text(0.01, 0.50, "Only successful terminal summaries enter scientific claims", transform=ax.transAxes, color="#52606d", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    finish(fig, "04_campaign_runtime.png")


def gauge_diagnostic() -> None:
    gauge = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    data = {
        "Chair": {
            "RynnBrain": [1.14, 1.24, 1.41, 1.51, 1.69, 1.93, 2.21, 2.44],
            "Qwen": [1.17, 1.16, 1.16, 1.16, 1.18, 1.34, 1.31, 1.16],
        },
        "Sofa": {
            "RynnBrain": [3.10, 3.32, 3.64, 4.01, 4.41, 5.31, 6.04, 6.94],
            "Qwen": [3.36, 3.36, 3.36, 3.36, 3.36, 3.36, 3.36, 3.43],
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharex=True)
    for ax, (scene, curves) in zip(axes, data.items()):
        ax.plot(gauge, curves["RynnBrain"], "o-", color=RYNN, linewidth=2.3, label="RynnBrain 122B")
        ax.plot(gauge, curves["Qwen"], "s--", color=QWEN, linewidth=2.0, label="Qwen 122B")
        ax.set_title(f"Fixed {scene.lower()} image")
        ax.set_xlabel("Raw homogeneous scale λ in λK")
        ax.set_ylabel("Predicted first-box depth (m)")
        ax.grid(color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Algebraically identical cameras change predicted geometry", y=1.03, fontsize=14, weight="bold")
    fig.text(0.01, 0.01, "A projectively coherent model would be invariant to λ because λK and K encode the same camera.", color="#52606d", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    finish(fig, "05_gauge_diagnostic.png")


def main() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    })
    protocol_adherence()
    contact_robustness()
    checkpoint_scale()
    campaign_runtime()
    gauge_diagnostic()
    print(f"Wrote five figures to {OUT}")


if __name__ == "__main__":
    main()
