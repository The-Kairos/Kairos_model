from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[1]
COUNTS_CSV = REPO_ROOT / "_processed_ablations" / "Titanic.1997.mkv" / "graph_part_counts_perc.csv"
MEASURES_CSV = REPO_ROOT / "_processed_ablations" / "Titanic.1997.mkv" / "graph_measures_perc.csv"
OUTPUT_PNG = REPO_ROOT / "_processed_ablations" / "Titanic.1997.mkv" / "titanic_ablation_heatmap.png"

VARIANTS = ["no_blip", "no_yolo", "no_asr", "no_ast"]
VARIANT_LABELS = {
    "no_blip": "BLIP removed",
    "no_yolo": "YOLO removed",
    "no_asr": "ASR removed",
    "no_ast": "AST removed",
}
METRICS = [
    "[nodes]",
    "[relationships:narrative]",
    "[relationships:spatial]",
    "[relationships:temporal]",
    "avg degree",
    "efficiency",
]

METRIC_LABELS = {
    "[nodes]": "Entity nodes",
    "[relationships:narrative]": "Narrative edges",
    "[relationships:spatial]": "Spatial edges",
    "[relationships:temporal]": "Temporal edges",
    "avg degree": "Average degree",
    "efficiency": "Global efficiency",
}


def extract_percent(cell: str) -> float:
    match = re.search(r"\(([+-]?\d+(?:\.\d+)?)%\)", cell)
    if not match:
        raise ValueError(f"Could not parse percentage delta from cell: {cell!r}")
    return float(match.group(1))


def read_counts_percentages(csv_path: Path) -> dict[str, dict[str, float]]:
    targets = {
        "[nodes]",
        "[relationships:narrative]",
        "[relationships:spatial]",
        "[relationships:temporal]",
    }
    percentages: dict[str, dict[str, float]] = {}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            part = (row.get("part") or "").strip()
            if part not in targets:
                continue
            percentages[part] = {variant: extract_percent(row[variant]) for variant in VARIANTS}

    missing = targets.difference(percentages)
    if missing:
        raise ValueError(f"Missing expected rows in {csv_path}: {sorted(missing)}")

    return percentages


def read_measure_percentages(csv_path: Path) -> dict[str, dict[str, float]]:
    row_map: dict[str, str] = {
        "avg degree": "average_degree",
        "efficiency": "global_efficiency",
    }
    percentages: dict[str, dict[str, float]] = {metric: {} for metric in row_map}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            variant = row.get("variant", "").strip()
            if variant == "full" or variant not in VARIANTS:
                continue
            for metric, column in row_map.items():
                percentages[metric][variant] = extract_percent(row[column])

    for metric, values in percentages.items():
        missing = [variant for variant in VARIANTS if variant not in values]
        if missing:
            raise ValueError(f"Missing expected values for {metric} in {csv_path}: {missing}")

    return percentages


def build_matrix() -> np.ndarray:
    counts = read_counts_percentages(COUNTS_CSV)
    measures = read_measure_percentages(MEASURES_CSV)

    metric_values: list[list[float]] = []
    for metric in METRICS:
        source = counts if metric in counts else measures
        metric_values.append([source[metric][variant] for variant in VARIANTS])

    return np.array(metric_values, dtype=float)


def render_heatmap(data: np.ndarray, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    vmax = float(np.max(np.abs(data)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list(
        "ablation_delta",
        ["#C50000", "#FFFFFF", "#1D9C11"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(8.6, 4), dpi=450)
    image = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(VARIANTS)), labels=[VARIANT_LABELS[variant] for variant in VARIANTS])
    ax.set_yticks(np.arange(len(METRICS)), labels=[METRIC_LABELS[metric] for metric in METRICS])
    ax.set_xlabel("Ablation Condition", fontweight="bold")
    ax.set_ylabel("Knowledge Graph Metrics", fontweight="bold")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            text_color = "white" if abs(value) > vmax * 0.45 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:+.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative change from full pipeline (%)", rotation=90, labelpad=10, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, len(VARIANTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    data = build_matrix()
    render_heatmap(data, OUTPUT_PNG)
    print(f"Saved heatmap to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
