
#!/usr/bin/env python3
"""
visualize_midi_relation_graph.py

Separate visualization utility for graph files produced by
midi_relation_graph_edges.py.

This script DOES NOT modify or depend on the internals of the graph builder.
It only reads the saved .npz file and renders graph figures.

Outputs
-------
For input: song_name.npz

<out-dir>/
    song_name_rhythm_graph.png
    song_name_melody_graph.png
    song_name_harmony_graph.png
    song_name_structure_graph.png
    song_name_combined_graph.png
    song_name_edge_summary.csv

Visualization
-------------
- Nodes are placed left-to-right in musical order.
- One node = one non-overlapping 4-bar segment.
- Curved arcs show graph edges.
- Thicker / more opaque arcs indicate larger edge weights.
- Node labels show 4-bar segment numbers.
- The combined graph overlays all four relation types using different
  line styles rather than relying only on color.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd


RELATIONS = ("rhythm", "melody", "harmony", "structure")


def load_graph(path: Path):
    g = np.load(path, allow_pickle=False)

    missing = [name for name in RELATIONS if name not in g.files]
    if missing:
        raise KeyError(
            f"Missing relation matrices in {path}: {missing}"
        )

    matrices = {
        name: np.asarray(g[name], dtype=float)
        for name in RELATIONS
    }

    shapes = {m.shape for m in matrices.values()}
    if len(shapes) != 1:
        raise ValueError(
            f"Relation matrices have inconsistent shapes: {shapes}"
        )

    shape = next(iter(shapes))
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            f"Expected square adjacency matrices, got {shape}"
        )

    n = shape[0]

    start_bar = (
        np.asarray(g["start_bar"], dtype=int)
        if "start_bar" in g.files
        else np.arange(n, dtype=int) * 4
    )

    start_time = (
        np.asarray(g["start_time"], dtype=float)
        if "start_time" in g.files
        else None
    )

    end_time = (
        np.asarray(g["end_time"], dtype=float)
        if "end_time" in g.files
        else None
    )

    return g, matrices, start_bar, start_time, end_time


def iter_edges(matrix: np.ndarray):
    n = matrix.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            w = float(matrix[i, j])
            if np.isfinite(w) and w > 0:
                yield i, j, w


def edge_count(matrix: np.ndarray) -> int:
    return sum(1 for _ in iter_edges(matrix))


def node_degree(matrix: np.ndarray) -> np.ndarray:
    positive = (matrix > 0).astype(float)
    np.fill_diagonal(positive, 0.0)
    return positive.sum(axis=1)


def curved_edge_patch(
    x1: float,
    x2: float,
    y: float,
    height: float,
    linewidth: float,
    alpha: float,
    linestyle: str = "-",
):
    """
    Quadratic Bezier arc from (x1,y) to (x2,y).
    """
    mid = (x1 + x2) / 2.0

    verts = [
        (x1, y),
        (mid, y + height),
        (x2, y),
    ]

    codes = [
        MplPath.MOVETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
    ]

    path = MplPath(verts, codes)

    return PathPatch(
        path,
        fill=False,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
    )


def plot_relation(
    relation: str,
    matrix: np.ndarray,
    start_bar: np.ndarray,
    out_path: Path,
):
    n = matrix.shape[0]

    x = np.arange(n, dtype=float)
    degree = node_degree(matrix)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.38), 5.5))

    edges = list(iter_edges(matrix))

    if edges:
        max_w = max(w for _, _, w in edges)
    else:
        max_w = 1.0

    for i, j, w in edges:
        distance = j - i

        # Larger temporal gaps get taller arcs.
        height = 0.25 + 0.11 * np.sqrt(distance)

        scaled = w / max(max_w, 1e-12)

        patch = curved_edge_patch(
            x[i],
            x[j],
            0.0,
            height=height,
            linewidth=0.7 + 2.5 * scaled,
            alpha=0.18 + 0.72 * scaled,
        )

        ax.add_patch(patch)

    # Node size reflects relation degree.
    sizes = 45 + 18 * degree

    ax.scatter(
        x,
        np.zeros(n),
        s=sizes,
        zorder=3,
    )

    labels = [
        f"{int(b)+1}-{int(b)+4}"
        for b in start_bar
    ]

    for idx, label in enumerate(labels):
        ax.text(
            x[idx],
            -0.11,
            label,
            ha="center",
            va="top",
            rotation=90 if n > 18 else 0,
            fontsize=8,
        )

    ax.axhline(
        0,
        linewidth=0.8,
        alpha=0.35,
    )

    ax.set_title(
        f"{relation.capitalize()} relation graph "
        f"({n} nodes, {len(edges)} edges)"
    )

    ax.set_xlabel(
        "4-bar segment in musical order"
    )

    ax.set_yticks([])

    if n:
        ax.set_xlim(-0.8, n - 0.2)

    max_distance = max(
        [j - i for i, j, _ in edges],
        default=1,
    )

    ax.set_ylim(
        -0.4,
        0.55 + 0.14 * np.sqrt(max_distance),
    )

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_combined(
    matrices: dict[str, np.ndarray],
    start_bar: np.ndarray,
    out_path: Path,
):
    """
    Overlay all relation types in a single temporal arc graph.

    Relations are distinguished by line style:
      rhythm    solid
      melody    dashed
      harmony   dash-dot
      structure dotted
    """
    n = next(iter(matrices.values())).shape[0]

    x = np.arange(n, dtype=float)

    combined_degree = np.zeros(n, dtype=float)
    for matrix in matrices.values():
        combined_degree += node_degree(matrix)

    line_styles = {
        "rhythm": "-",
        "melody": "--",
        "harmony": "-.",
        "structure": ":",
    }

    vertical_offsets = {
        "rhythm": 0.00,
        "melody": 0.12,
        "harmony": 0.24,
        "structure": 0.36,
    }

    fig, ax = plt.subplots(
        figsize=(max(11, n * 0.4), 6.4)
    )

    total_edges = 0

    for relation in RELATIONS:
        matrix = matrices[relation]
        edges = list(iter_edges(matrix))
        total_edges += len(edges)

        max_w = max(
            [w for _, _, w in edges],
            default=1.0,
        )

        for i, j, w in edges:
            distance = j - i
            scaled = w / max(max_w, 1e-12)

            height = (
                0.25
                + vertical_offsets[relation]
                + 0.10 * np.sqrt(distance)
            )

            patch = curved_edge_patch(
                x[i],
                x[j],
                0.0,
                height=height,
                linewidth=0.6 + 2.0 * scaled,
                alpha=0.12 + 0.48 * scaled,
                linestyle=line_styles[relation],
            )

            ax.add_patch(patch)

    ax.scatter(
        x,
        np.zeros(n),
        s=50 + 12 * combined_degree,
        zorder=3,
    )

    labels = [
        f"{int(b)+1}-{int(b)+4}"
        for b in start_bar
    ]

    for idx, label in enumerate(labels):
        ax.text(
            x[idx],
            -0.12,
            label,
            ha="center",
            va="top",
            rotation=90 if n > 18 else 0,
            fontsize=8,
        )

    # Legend handles use the same line styles.
    handles = []
    for relation in RELATIONS:
        handle, = ax.plot(
            [],
            [],
            linestyle=line_styles[relation],
            linewidth=2.0,
            label=relation,
        )
        handles.append(handle)

    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
    )

    ax.axhline(
        0,
        linewidth=0.8,
        alpha=0.35,
    )

    ax.set_title(
        f"Combined MIDI relation graph "
        f"({n} nodes, {total_edges} typed edges)"
    )

    ax.set_xlabel(
        "4-bar segment in musical order"
    )

    ax.set_yticks([])

    if n:
        ax.set_xlim(-0.8, n - 0.2)

    ax.set_ylim(
        -0.45,
        1.35,
    )

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_edge_summary(
    matrices: dict[str, np.ndarray],
    out_path: Path,
):
    rows = []

    for relation, matrix in matrices.items():
        edges = list(iter_edges(matrix))

        weights = np.asarray(
            [w for _, _, w in edges],
            dtype=float,
        )

        distances = np.asarray(
            [j - i for i, j, _ in edges],
            dtype=float,
        )

        rows.append({
            "relation": relation,
            "num_nodes": matrix.shape[0],
            "num_edges": len(edges),
            "mean_edge_weight": (
                float(weights.mean())
                if len(weights)
                else float("nan")
            ),
            "max_edge_weight": (
                float(weights.max())
                if len(weights)
                else float("nan")
            ),
            "mean_edge_distance_4bar_blocks": (
                float(distances.mean())
                if len(distances)
                else float("nan")
            ),
            "max_edge_distance_4bar_blocks": (
                float(distances.max())
                if len(distances)
                else float("nan")
            ),
        })

    pd.DataFrame(rows).to_csv(
        out_path,
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph",
        type=Path,
        required=True,
        help="Graph .npz generated by midi_relation_graph_edges.py",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        _,
        matrices,
        start_bar,
        _,
        _,
    ) = load_graph(args.graph)

    # Use the input NPZ stem as the prefix for every generated file.
    # Example:
    #   my_song.npz
    #     -> my_song_rhythm_graph.png
    #     -> my_song_melody_graph.png
    #     -> my_song_harmony_graph.png
    #     -> my_song_structure_graph.png
    #     -> my_song_combined_graph.png
    #     -> my_song_edge_summary.csv
    graph_stem = args.graph.stem

    for relation in RELATIONS:
        out_path = (
            args.out_dir
            / f"{graph_stem}_{relation}_graph.png"
        )

        plot_relation(
            relation,
            matrices[relation],
            start_bar,
            out_path,
        )

        print(f"Saved: {out_path}")

    combined_path = (
        args.out_dir
        / f"{graph_stem}_combined_graph.png"
    )

    plot_combined(
        matrices,
        start_bar,
        combined_path,
    )

    print(f"Saved: {combined_path}")

    summary_path = (
        args.out_dir
        / f"{graph_stem}_edge_summary.csv"
    )

    save_edge_summary(
        matrices,
        summary_path,
    )

    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
