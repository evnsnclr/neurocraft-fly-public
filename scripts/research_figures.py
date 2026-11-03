"""Deterministic figures for the small, explicitly modeled research example.

These helpers visualize supplied CSV rows without smoothing or normalization.
They require the optional ``research`` dependency group, not the Minecraft mod.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

Scalar = str | int | float
Rows = Sequence[Mapping[str, Scalar]]

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "neurocraft-research-example-v1",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

CONDITIONS = (
    ("original", "Original fixture", "#0072B2", "-"),
    ("shuffled_weights", "Shuffled weights", "#D55E00", "--"),
    ("no_input", "No input", "#4B5563", ":"),
)
SIDES = (("L", "Left", "#0072B2", "-"), ("R", "Right", "#D55E00", "--"))


def _series(rows: Rows, *, condition: str, field: str, value: str) -> list[Mapping[str, Scalar]]:
    selected = sorted(
        (row for row in rows if row["condition"] == condition and str(row[field]) == value),
        key=lambda row: float(row["time_ms"]),
    )
    if not selected:
        raise ValueError(f"No rows for condition={condition}, {field}={value}")
    return selected


def plot_network(nodes: Rows, edges: Rows) -> Figure:
    """Plot raw directed source weights with body IDs on both axes."""
    body_ids = [str(row["body_id"]) for row in nodes]
    index = {body_id: position for position, body_id in enumerate(body_ids)}
    weights = [[0.0 for _ in body_ids] for _ in body_ids]
    for edge in edges:
        if edge["condition"] == "original":
            weights[index[str(edge["body_post"])]][index[str(edge["body_pre"])]] = float(
                edge["raw_count"]
            )
    with mpl.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(8.6, 7.4), layout="constrained")
        image = axis.imshow(weights, cmap="cividis", interpolation="nearest", vmin=0)
        axis.set_xticks(range(len(body_ids)), body_ids, rotation=90, fontsize=9)
        axis.set_yticks(range(len(body_ids)), body_ids, fontsize=9)
        axis.set_xlabel("Presynaptic source body ID →")
        axis.set_ylabel("Postsynaptic source body ID →")
        axis.set_title("14 retained neurons · 22 directed source edges", pad=14)
        figure.colorbar(image, ax=axis, shrink=0.78, label="Raw source weight (contact count)")
    return figure


def plot_stimulation(inputs: Rows, readouts: Rows, *, condition: str = "original") -> Figure:
    """Separate the prescribed direct input from downstream modeled activity."""
    with mpl.rc_context(STYLE):
        figure, axes = plt.subplots(4, 1, figsize=(10.5, 9.0), sharex=True, layout="constrained")
        for body_id, label, color, linestyle in (
            ("805722", "Left entry · 805722", "#0072B2", "-"),
            ("77185", "Right entry · 77185", "#D55E00", "--"),
        ):
            selected = _series(inputs, condition=condition, field="body_id", value=body_id)
            axes[0].step(
                [float(row["time_ms"]) for row in selected],
                [float(row["applied_drive"]) for row in selected],
                where="post",
                label=label,
                color=color,
                linestyle=linestyle,
            )
        axes[0].set_title("A prescribed pulse enters the fixture", loc="left")
        axes[0].set_ylabel("Applied input\n(model units)")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)
        for axis, population, title in (
            (axes[1], "sensory", "Entry-neuron responses"),
            (axes[2], "DNg07", "Descending readouts"),
            (axes[3], "hg1_motor", "Motor readouts"),
        ):
            for side, label, color, linestyle in SIDES:
                selected = _series(
                    readouts, condition=condition, field="pool", value=f"{population}_{side}"
                )
                axis.plot(
                    [float(row["time_ms"]) for row in selected],
                    [float(row["mean_activity"]) for row in selected],
                    label=f"{label} · {population}_{side}",
                    color=color,
                    linestyle=linestyle,
                )
            axis.set_title(title, loc="left")
            axis.set_ylabel("Modeled activity\n(dimensionless)")
            axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)
        for axis in axes:
            axis.grid(axis="y")
            axis.set_xlim(0, 1000)
        axes[-1].set_xlabel("Modeled time (ms)")
    return figure


def plot_comparison(readouts: Rows, *, shuffle_seed: int = 7) -> Figure:
    """Compare matched conditions on four declared singleton populations."""
    with mpl.rc_context(STYLE):
        figure, axes = plt.subplots(
            2, 2, figsize=(10.0, 6.5), sharex=True, sharey="row", layout="constrained"
        )
        for axis, population, title in zip(
            axes.flat,
            ("DNg07_L", "DNg07_R", "hg1_motor_L", "hg1_motor_R"),
            (
                "Left descending · DNg07",
                "Right descending · DNg07",
                "Left motor · hg1",
                "Right motor · hg1",
            ),
            strict=True,
        ):
            for condition, label, color, linestyle in CONDITIONS:
                if condition == "shuffled_weights":
                    label = f"{label} · seed {shuffle_seed}"
                selected = _series(readouts, condition=condition, field="pool", value=population)
                axis.plot(
                    [float(row["time_ms"]) for row in selected],
                    [float(row["mean_activity"]) for row in selected],
                    label=label,
                    color=color,
                    linestyle=linestyle,
                )
            axis.set_title(title, loc="left")
            axis.set_xlim(0, 1000)
            axis.grid(axis="y")
        axes[0, 0].set_ylabel("Modeled activity\n(dimensionless)")
        axes[1, 0].set_ylabel("Modeled activity\n(dimensionless)")
        for axis in axes[-1]:
            axis.set_xlabel("Modeled time (ms)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=10)
    return figure


def export_figure(figure: Figure, stem: Path) -> tuple[Path, Path]:
    """Save editable SVG and 240-DPI PNG with stable metadata and no timestamps."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path, png_path = stem.with_suffix(".svg"), stem.with_suffix(".png")
    with mpl.rc_context(STYLE):
        figure.savefig(svg_path, metadata={"Date": None, "Creator": "NeuroCraft Fly"})
        figure.savefig(png_path, dpi=240, metadata={"Software": "NeuroCraft Fly"})
    return svg_path, png_path
