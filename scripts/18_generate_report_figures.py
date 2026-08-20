"""Generate reproducible figures for the GeoVLM experiment report.

Usage:
    python scripts/18_generate_report_figures.py

The script reads the current comparison and geometry JSON files, then writes
publication-style PNG figures under report/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = REPO_ROOT / "outputs" / "evaluations" / "track_comparison.json"
FIGURE_DIR = REPO_ROOT / "report" / "figures"

TRACK_LABELS = {
    "pure_vlm": "Pure VLM",
    "geometry_only": "Geometry-only",
    "geovlm": "GeoVLM",
}
TYPE_LABELS = {
    "closer_farther": "Closer/farther",
    "support_relation": "Support relation",
    "physical_size": "Physical size",
}
COLORS = {
    "pure_vlm": "#4C78A8",
    "geometry_only": "#F58518",
    "geovlm": "#54A24B",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / name,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def generate_track_accuracy(comparison: dict[str, Any]) -> None:
    tracks = list(TRACK_LABELS)
    metrics = comparison["tracks"]
    values = [metrics[track]["accuracy"] * 100.0 for track in tracks]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bars = ax.bar(
        [TRACK_LABELS[track] for track in tracks],
        values,
        color=[COLORS[track] for track in tracks],
        width=0.58,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall accuracy on the 81-question clean subset", fontsize=13)
    style_axes(ax)
    save_figure(fig, "track_accuracy.png")


def generate_question_type_accuracy(comparison: dict[str, Any]) -> None:
    tracks = list(TRACK_LABELS)
    types = ["closer_farther", "support_relation", "physical_size"]
    x_positions = list(range(len(types)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for index, track in enumerate(tracks):
        values = [
            comparison["tracks"][track]["by_type"][question_type]["accuracy"] * 100.0
            for question_type in types
        ]
        offsets = [position + (index - 1) * width for position in x_positions]
        bars = ax.bar(
            offsets,
            values,
            width=width,
            color=COLORS[track],
            label=TRACK_LABELS[track],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([TYPE_LABELS[question_type] for question_type in types])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.03))
    style_axes(ax)
    save_figure(fig, "question_type_accuracy.png")


def generate_pairwise_outcomes(comparison: dict[str, Any]) -> None:
    pair_key = "pure_vlm_vs_geovlm"
    pair = comparison["pairwise"][pair_key]
    labels = ["Both correct", "GeoVLM only\ncorrect", "Both wrong"]
    values = [pair["both_correct"], pair["right_only_correct"], pair["both_wrong"]]
    colors = ["#72B7B2", "#54A24B", "#E45756"]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            str(value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylim(0, 84)
    ax.set_ylabel("Number of questions")
    ax.set_title("Paired outcome: Pure VLM versus GeoVLM", fontsize=13)
    style_axes(ax)
    save_figure(fig, "pairwise_outcomes.png")


def generate_closeness_ambiguity() -> None:
    samples = {
        "View 18": (5.540544, 5.740040),
        "View 19": (5.163826, 5.189509),
        "View 21": (5.789515, 5.798141),
    }
    labels = list(samples)
    laptop_scores = [samples[label][0] for label in labels]
    mouse_scores = [samples[label][1] for label in labels]
    x_positions = list(range(len(labels)))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    laptop_bars = ax.bar(
        [x - width / 2 for x in x_positions],
        laptop_scores,
        width=width,
        color="#4C78A8",
        label="Laptop",
    )
    mouse_bars = ax.bar(
        [x + width / 2 for x in x_positions],
        mouse_scores,
        width=width,
        color="#F58518",
        label="Mouse",
    )
    for bars in (laptop_bars, mouse_bars):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Near-surface closeness score")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.03))
    style_axes(ax)
    save_figure(fig, "closeness_ambiguity.png")


def add_flow_box(
    ax: plt.Axes,
    center: tuple[float, float],
    text: str,
    width: float = 2.2,
    height: float = 0.72,
    color: str = "#EAF2F8",
    fontsize: float = 9,
) -> None:
    left = center[0] - width / 2
    bottom = center[1] - height / 2
    patch = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=1.2,
        edgecolor="#376A8A",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        center[0],
        center[1],
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
    )


def add_flow_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "mutation_scale": 15,
            "linewidth": 1.2,
            "color": "#5B6770",
        },
    )


def generate_pipeline_flow() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.1))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.35)
    ax.axis("off")

    centers = [
        (0.95, 2.8),
        (3.05, 2.8),
        (5.15, 2.8),
        (7.25, 2.8),
        (9.35, 2.8),
        (11.45, 2.8),
    ]
    texts = [
        "Input image",
        "YOLO\nDetection",
        "SAM2\nSegmentation",
        "Depth Anything V2\nRelative depth",
        "Object-level\ngeometry",
        "Reasoning\ntracks",
    ]
    colors = ["#F7F7F7", "#E8F1FB", "#EAF7EE", "#FFF4DD", "#F3EAF7", "#FDECEC"]
    for center, text, color in zip(centers, texts, colors):
        add_flow_box(ax, center, text, width=1.72, height=0.62, color=color, fontsize=8.3)
    for start, end in zip(centers[:-1], centers[1:]):
        add_flow_arrow(ax, (start[0] + 0.88, start[1]), (end[0] - 0.88, end[1]))

    branch_y = 1.0
    branch_centers = [(6.75, branch_y), (9.05, branch_y), (11.35, branch_y)]
    branch_texts = [
        "Pure VLM\nimage +\nquestion",
        "Geometry-only\ngeometry +\nquestion",
        "GeoVLM\nimage + geometry\n+ question",
    ]
    branch_colors = ["#DCEAF7", "#FCE5C5", "#DDF1D9"]
    for center, text, color in zip(branch_centers, branch_texts, branch_colors):
        add_flow_box(
            ax,
            center,
            text,
            width=1.65,
            height=0.75,
            color=color,
            fontsize=8.2,
        )

    add_flow_arrow(ax, (9.35, 2.43), (6.75, 1.45))
    add_flow_arrow(ax, (9.35, 2.43), (9.05, 1.45))
    add_flow_arrow(ax, (9.35, 2.43), (11.35, 1.45))
    add_flow_box(
        ax,
        (9.0, 0.43),
        "Exact-match evaluation\nand paired comparison",
        width=2.65,
        height=0.48,
        color="#F1F1F1",
        fontsize=8.5,
    )
    add_flow_arrow(ax, (6.75, 0.55), (8.0, 0.71))
    add_flow_arrow(ax, (9.05, 0.55), (9.0, 0.71))
    add_flow_arrow(ax, (11.35, 0.55), (10.0, 0.71))

    ax.text(
        6.0,
        3.78,
        "GeoVLM-SceneReasoner implementation flow",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, "pipeline_flow.png")


def main() -> int:
    comparison = load_json(COMPARISON_PATH)
    generate_track_accuracy(comparison)
    generate_question_type_accuracy(comparison)
    generate_pairwise_outcomes(comparison)
    generate_closeness_ambiguity()
    generate_pipeline_flow()
    print(f"Generated figures under {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
