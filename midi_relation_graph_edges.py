
#!/usr/bin/env python3
"""
midi_relation_graph_edges.py

Build relation graphs for the current Real-vs-Fake MIDI analysis.

Graph node
----------
One node = one non-overlapping 4-bar MIDI segment.

Main relation edges
-------------------
rhythm:
    Recurrence of exact bar-level 16-bin onset patterns.

melody:
    Recurrence of transposition-invariant bar-level melodic interval patterns.

harmony:
    Recurrence of chord bigrams / trigrams / quadgrams.

structure:
    4-bar and 8-bar structural recurrence based on the same 28-D bar
    representation used by compare_real_fake_midi.py:
        12 pitch-class bins + 16 onset bins.
    Non-adjacent 4-bar recurrence uses:
        abs(i-j) >= 2
        cosine similarity >= 0.85

The graph code also saves the song-level analysis values used in
compare_real_fake_midi.py so graph construction and statistical analysis
remain aligned.

Saved NPZ keys
--------------
Main graph matrices [T,T]:
    rhythm
    melody
    harmony
    structure
    combined

Normalized graph matrices [T,T]:
    rhythm_norm
    melody_norm
    harmony_norm
    structure_norm
    combined_norm

Structure diagnostics:
    structure_4bar
    structure_8bar
    segment_distance_blocks

Per-node diagnostics [T]:
    novel_material_node
    notes_per_bar_node
    pitch_class_entropy_node
    chord_unique_ratio_node

Song-level scalar diagnostics:
    num_bars
    num_notes
    notes_per_bar

    recurrence_1bar
    recurrence_2bar
    recurrence_4bar
    recurrence_8bar
    nonadjacent_recurrence_ratio
    novel_material_ratio_4bar
    recurrence_distance_mean_blocks

    rhythm_pattern_recurrence

    melodic_pattern_recurrence
    pitch_class_entropy

    chord_bigram_recurrence
    chord_trigram_recurrence
    chord_quadgram_recurrence
    chord_unique_ratio
"""

from __future__ import annotations

import argparse
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pretty_midi


MIDI_EXTS = {".mid", ".midi"}
EPS = 1e-12

MAJOR_TEMPLATE = np.zeros(12, dtype=float)
MAJOR_TEMPLATE[[0, 4, 7]] = 1.0

MINOR_TEMPLATE = np.zeros(12, dtype=float)
MINOR_TEMPLATE[[0, 3, 7]] = 1.0


# ---------------------------------------------------------------------
# Basic MIDI utilities
# ---------------------------------------------------------------------

def collect_midis(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in MIDI_EXTS
    )


def shannon_entropy(prob: np.ndarray) -> float:
    prob = np.asarray(prob, dtype=float)
    prob = prob[prob > 0]
    if len(prob) == 0:
        return float("nan")
    return float(-(prob * np.log2(prob)).sum())


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norm, EPS)

    sim = x @ x.T
    sim = np.clip(sim, 0.0, 1.0)

    return sim.astype(np.float32)


def get_note_events(
    pm: pretty_midi.PrettyMIDI,
) -> list[tuple[float, float, int]]:
    notes = []

    for inst in pm.instruments:
        if inst.is_drum:
            continue

        for n in inst.notes:
            if n.end <= n.start:
                continue

            notes.append(
                (float(n.start), float(n.end), int(n.pitch))
            )

    notes.sort(key=lambda z: (z[0], z[2], z[1]))
    return notes


def get_bar_boundaries(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    """
    Same fallback policy as compare_real_fake_midi.py:
    1) explicit downbeats
    2) every 4 detected beats
    3) estimated tempo, assuming 4/4
    """
    try:
        downbeats = np.asarray(pm.get_downbeats(), dtype=float)
    except Exception:
        downbeats = np.array([], dtype=float)

    end_time = float(pm.get_end_time())

    if len(downbeats) >= 2:
        if downbeats[0] > 1e-6:
            downbeats = np.r_[0.0, downbeats]

        if downbeats[-1] < end_time:
            bar_dur = np.median(
                np.diff(downbeats[-min(8, len(downbeats)):])
            )

            if np.isfinite(bar_dur) and bar_dur > 0:
                extra = []
                t = downbeats[-1]

                while t < end_time:
                    t += bar_dur
                    extra.append(t)

                downbeats = np.r_[downbeats, extra]

        return downbeats

    try:
        beats = np.asarray(pm.get_beats(), dtype=float)
    except Exception:
        beats = np.array([], dtype=float)

    if len(beats) >= 5:
        bars = beats[::4]

        if bars[0] > 1e-6:
            bars = np.r_[0.0, bars]

        if bars[-1] < end_time:
            beat_dur = np.median(
                np.diff(beats[-min(16, len(beats)):])
            )
            bar_dur = 4.0 * beat_dur

            extra = []
            t = bars[-1]

            while t < end_time:
                t += bar_dur
                extra.append(t)

            bars = np.r_[bars, extra]

        return bars

    try:
        tempo = float(pm.estimate_tempo())
    except Exception:
        tempo = 120.0

    if not np.isfinite(tempo) or tempo <= 0:
        tempo = 120.0

    bar_dur = 4.0 * 60.0 / tempo

    if end_time <= 0:
        return np.array([0.0, bar_dur], dtype=float)

    n = max(1, int(math.ceil(end_time / bar_dur)))
    return np.arange(n + 1, dtype=float) * bar_dur


def notes_in_range(
    notes: list[tuple[float, float, int]],
    start: float,
    end: float,
) -> list[tuple[float, float, int]]:
    return [n for n in notes if start <= n[0] < end]


# ---------------------------------------------------------------------
# Bar-level features
# ---------------------------------------------------------------------

def bar_pitch_class_hist(
    bar_notes: list[tuple[float, float, int]]
) -> np.ndarray:
    hist = np.zeros(12, dtype=float)

    for _, _, pitch in bar_notes:
        hist[pitch % 12] += 1.0

    if hist.sum() > 0:
        hist /= hist.sum()

    return hist


def bar_onset_pattern(
    bar_notes: list[tuple[float, float, int]],
    start: float,
    end: float,
    bins: int = 16,
) -> np.ndarray:
    pattern = np.zeros(bins, dtype=float)
    duration = max(end - start, EPS)

    for onset, _, _ in bar_notes:
        pos = (onset - start) / duration
        idx = min(
            bins - 1,
            max(0, int(np.floor(pos * bins))),
        )
        pattern[idx] = 1.0

    return pattern


def bar_topnote_interval_signature(
    bar_notes: list[tuple[float, float, int]],
    onset_tol: float = 0.04,
    max_intervals: int = 12,
) -> tuple[int, ...] | None:
    """
    Lightweight pseudo-melody, identical in spirit to the analysis code:
    - group near-simultaneous onsets
    - keep highest pitch
    - convert to pitch intervals
    => transposition-invariant melodic signature
    """
    if len(bar_notes) < 2:
        return None

    pitches = []
    current_onset = None
    current_pitches = []

    for onset, _, pitch in sorted(
        bar_notes,
        key=lambda z: (z[0], z[2]),
    ):
        if (
            current_onset is None
            or abs(onset - current_onset) <= onset_tol
        ):
            if current_onset is None:
                current_onset = onset

            current_pitches.append(pitch)

        else:
            pitches.append(max(current_pitches))
            current_onset = onset
            current_pitches = [pitch]

    if current_pitches:
        pitches.append(max(current_pitches))

    if len(pitches) < 3:
        return None

    intervals = np.diff(pitches).astype(int)
    intervals = np.clip(intervals, -24, 24)

    if len(intervals) > max_intervals:
        intervals = intervals[:max_intervals]

    return tuple(int(x) for x in intervals)


def estimate_bar_chord(
    bar_notes: list[tuple[float, float, int]]
) -> str | None:
    """
    Lightweight 24-triad estimator:
    12 roots x {major, minor}.
    """
    if not bar_notes:
        return None

    pc = np.zeros(12, dtype=float)

    for onset, end, pitch in bar_notes:
        duration = max(end - onset, 0.02)
        pc[pitch % 12] += duration

    if pc.sum() <= EPS:
        return None

    pc /= pc.sum()

    names = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B",
    ]

    best_label = None
    best_score = -np.inf

    for root in range(12):
        for quality, template in (
            ("maj", MAJOR_TEMPLATE),
            ("min", MINOR_TEMPLATE),
        ):
            shifted = np.roll(template, root)

            score = float(
                np.dot(pc, shifted)
                / (
                    np.linalg.norm(pc)
                    * np.linalg.norm(shifted)
                    + EPS
                )
            )

            if score > best_score:
                best_score = score
                best_label = f"{names[root]}:{quality}"

    return best_label


def build_bars(
    notes: list[tuple[float, float, int]],
    boundaries: np.ndarray,
) -> list[dict]:
    bars = []

    for i in range(len(boundaries) - 1):
        start = float(boundaries[i])
        end = float(boundaries[i + 1])

        if end <= start:
            continue

        bar_notes = notes_in_range(notes, start, end)

        pc = bar_pitch_class_hist(bar_notes)
        rhythm = bar_onset_pattern(
            bar_notes,
            start,
            end,
            bins=16,
        )

        bars.append({
            "start": start,
            "end": end,
            "notes": bar_notes,
            "pc": pc,
            "rhythm": rhythm,
            "bar_vector": np.r_[pc, rhythm],
            "rhythm_signature": tuple(
                int(v) for v in rhythm
            ),
            "melody_signature":
                bar_topnote_interval_signature(bar_notes),
            "chord": estimate_bar_chord(bar_notes),
        })

    return bars


# ---------------------------------------------------------------------
# Song-level features: aligned with compare_real_fake_midi.py
# ---------------------------------------------------------------------

def block_vectors(
    bar_vectors: np.ndarray,
    scale_bars: int,
) -> np.ndarray:
    n = len(bar_vectors)

    if n < scale_bars:
        return np.zeros(
            (0, bar_vectors.shape[1]),
            dtype=float,
        )

    blocks = []

    for start in range(
        0,
        n - scale_bars + 1,
        scale_bars,
    ):
        blocks.append(
            bar_vectors[
                start:start + scale_bars
            ].mean(axis=0)
        )

    if not blocks:
        return np.zeros(
            (0, bar_vectors.shape[1]),
            dtype=float,
        )

    return np.vstack(blocks)


def recurrence_ratio(
    vectors: np.ndarray,
    threshold: float = 0.85,
    min_index_distance: int = 2,
) -> float:
    if len(vectors) < 2:
        return float("nan")

    ssm = cosine_matrix(vectors)

    recurrent = []
    n = len(vectors)

    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) < min_index_distance:
                continue

            recurrent.append(
                ssm[i, j] >= threshold
            )

    if not recurrent:
        return float("nan")

    return float(np.mean(recurrent))


def recurrence_distances(
    vectors: np.ndarray,
    threshold: float = 0.85,
    min_index_distance: int = 2,
) -> list[int]:
    if len(vectors) < 2:
        return []

    ssm = cosine_matrix(vectors)

    distances = []
    n = len(vectors)

    for i in range(n):
        for j in range(i + 1, n):
            distance = j - i

            if distance < min_index_distance:
                continue

            if ssm[i, j] >= threshold:
                distances.append(distance)

    return distances


def novel_material_flags(
    vectors: np.ndarray,
    threshold: float = 0.85,
) -> np.ndarray:
    """
    Per-4-bar-node novelty:
      1 = no previous 4-bar node has similarity >= threshold
      0 = previous similar material exists
    """
    n = len(vectors)

    if n == 0:
        return np.zeros(0, dtype=np.float32)

    flags = np.zeros(n, dtype=np.float32)
    flags[0] = 1.0

    ssm = cosine_matrix(vectors)

    for i in range(1, n):
        if np.max(ssm[i, :i]) < threshold:
            flags[i] = 1.0

    return flags


def recurrence_from_unique_ratio(
    signatures,
) -> float:
    valid = [
        x for x in signatures
        if x is not None
    ]

    if not valid:
        return float("nan")

    unique_ratio = len(set(valid)) / len(valid)
    return float(1.0 - unique_ratio)


def ngram_recurrence(
    sequence: list[str | None],
    n: int,
) -> float:
    clean = [
        x for x in sequence
        if x is not None
    ]

    if len(clean) < n:
        return float("nan")

    grams = [
        tuple(clean[i:i+n])
        for i in range(len(clean) - n + 1)
    ]

    return float(
        1.0 - len(set(grams)) / len(grams)
    )


def chord_unique_ratio(
    sequence: list[str | None],
) -> float:
    clean = [
        x for x in sequence
        if x is not None
    ]

    if not clean:
        return float("nan")

    return float(
        len(set(clean)) / len(clean)
    )


def global_pitch_class_entropy(
    notes: list[tuple[float, float, int]]
) -> float:
    if not notes:
        return float("nan")

    hist = np.zeros(12, dtype=float)

    for _, _, pitch in notes:
        hist[pitch % 12] += 1.0

    hist /= max(hist.sum(), EPS)
    return shannon_entropy(hist)


# ---------------------------------------------------------------------
# Pairwise relation scores for graph edges
# ---------------------------------------------------------------------

def multiset_overlap(
    a,
    b,
) -> float:
    """
    Multiset overlap in [0,1].
    Exact pattern recurrence gives positive similarity.
    """
    aa = [x for x in a if x is not None]
    bb = [x for x in b if x is not None]

    if not aa or not bb:
        return 0.0

    ca = Counter(aa)
    cb = Counter(bb)

    overlap = sum(
        min(ca[k], cb[k])
        for k in (ca.keys() | cb.keys())
    )

    denom = max(len(aa), len(bb))
    return float(overlap / max(denom, 1))


def ngrams(
    sequence: list[str | None],
    n: int,
) -> list[tuple[str, ...]]:
    clean = [
        x for x in sequence
        if x is not None
    ]

    if len(clean) < n:
        return []

    return [
        tuple(clean[i:i+n])
        for i in range(len(clean) - n + 1)
    ]


def build_four_bar_nodes(
    bars: list[dict],
) -> list[dict]:
    """
    Non-overlapping 4-bar nodes:
      node 0 = bars 0..3
      node 1 = bars 4..7
      ...
    """
    nodes = []

    for start in range(
        0,
        len(bars) - 3,
        4,
    ):
        chunk = bars[start:start + 4]

        bar_vectors = np.vstack([
            bar["bar_vector"]
            for bar in chunk
        ])

        notes = []
        for bar in chunk:
            notes.extend(bar["notes"])

        node = {
            "start_bar": start,
            "start_time": chunk[0]["start"],
            "end_time": chunk[-1]["end"],

            # Same representation used by recurrence_4bar
            "structure_4bar_vector":
                bar_vectors.mean(axis=0),

            # Exact recurrence signatures
            "rhythm_signatures": [
                bar["rhythm_signature"]
                for bar in chunk
            ],
            "melody_signatures": [
                bar["melody_signature"]
                for bar in chunk
            ],
            "chords": [
                bar["chord"]
                for bar in chunk
            ],

            # Auxiliary node-level information
            "notes_per_bar":
                len(notes) / 4.0,
            "pitch_class_entropy":
                global_pitch_class_entropy(notes),
            "chord_unique_ratio":
                chord_unique_ratio(
                    [bar["chord"] for bar in chunk]
                ),
        }

        nodes.append(node)

    return nodes


def pairwise_rhythm_scores(
    nodes: list[dict],
) -> np.ndarray:
    n = len(nodes)
    score = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            s = multiset_overlap(
                nodes[i]["rhythm_signatures"],
                nodes[j]["rhythm_signatures"],
            )
            score[i, j] = s
            score[j, i] = s

    return score


def pairwise_melody_scores(
    nodes: list[dict],
) -> np.ndarray:
    n = len(nodes)
    score = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            s = multiset_overlap(
                nodes[i]["melody_signatures"],
                nodes[j]["melody_signatures"],
            )
            score[i, j] = s
            score[j, i] = s

    return score


def pairwise_harmony_scores(
    nodes: list[dict],
) -> np.ndarray:
    """
    Each 4-bar node contains 4 estimated chords.

    Pairwise harmony score =
        mean(
            bigram recurrence overlap,
            trigram recurrence overlap,
            quadgram recurrence overlap
        )

    This directly reflects the current harmony feature family.
    """
    n_nodes = len(nodes)
    score = np.zeros(
        (n_nodes, n_nodes),
        dtype=np.float32,
    )

    for i in range(n_nodes):
        chords_i = nodes[i]["chords"]

        grams_i = {
            2: ngrams(chords_i, 2),
            3: ngrams(chords_i, 3),
            4: ngrams(chords_i, 4),
        }

        for j in range(i + 1, n_nodes):
            chords_j = nodes[j]["chords"]

            grams_j = {
                2: ngrams(chords_j, 2),
                3: ngrams(chords_j, 3),
                4: ngrams(chords_j, 4),
            }

            components = []

            for n in (2, 3, 4):
                a = grams_i[n]
                b = grams_j[n]

                if a and b:
                    components.append(
                        multiset_overlap(a, b)
                    )

            s = (
                float(np.mean(components))
                if components
                else 0.0
            )

            score[i, j] = s
            score[j, i] = s

    return score


def pairwise_structure_scores(
    bars: list[dict],
    nodes: list[dict],
    recurrence_threshold: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        structure_4bar edge matrix
        structure_8bar edge matrix
        combined structure edge matrix

    4-bar relation:
        node distance >= 2
        cosine >= 0.85

    8-bar relation:
        uses the exact non-overlapping 8-bar block construction from
        compare_real_fake_midi.py.
        8-bar block k maps to 4-bar graph node index 2*k.
    """
    n_nodes = len(nodes)

    # ----- 4-bar relation -----
    node_vectors = np.vstack([
        node["structure_4bar_vector"]
        for node in nodes
    ])

    sim4 = cosine_matrix(node_vectors)
    edge4 = np.zeros_like(
        sim4,
        dtype=np.float32,
    )

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if (j - i) < 2:
                continue

            if sim4[i, j] >= recurrence_threshold:
                edge4[i, j] = sim4[i, j]
                edge4[j, i] = sim4[i, j]

    # ----- 8-bar relation -----
    bar_vectors = np.vstack([
        bar["bar_vector"]
        for bar in bars
    ])

    block8 = block_vectors(
        bar_vectors,
        scale_bars=8,
    )

    sim8 = cosine_matrix(block8)
    edge8 = np.zeros(
        (n_nodes, n_nodes),
        dtype=np.float32,
    )

    # 8-bar block k starts at 4-bar node 2*k
    for i8 in range(len(block8)):
        for j8 in range(i8 + 1, len(block8)):
            # Same recurrence-analysis rule:
            # exclude immediately adjacent 8-bar blocks
            if (j8 - i8) < 2:
                continue

            if sim8[i8, j8] < recurrence_threshold:
                continue

            i_node = 2 * i8
            j_node = 2 * j8

            if (
                i_node < n_nodes
                and j_node < n_nodes
            ):
                edge8[i_node, j_node] = sim8[i8, j8]
                edge8[j_node, i_node] = sim8[i8, j8]

    structure = np.maximum(
        edge4,
        edge8,
    ).astype(np.float32)

    return (
        edge4.astype(np.float32),
        edge8.astype(np.float32),
        structure,
    )


# ---------------------------------------------------------------------
# Edge sparsification / normalization
# ---------------------------------------------------------------------

def topk_symmetric(
    score: np.ndarray,
    top_k: int,
    min_score: float = 0.0,
) -> np.ndarray:
    """
    Keep up to top-k positive relations per node.
    top_k <= 0 means keep every score > min_score.
    """
    n = score.shape[0]

    if top_k <= 0:
        out = np.where(
            score > min_score,
            score,
            0.0,
        ).astype(np.float32)

        np.fill_diagonal(out, 0.0)
        return np.maximum(
            out,
            out.T,
        ).astype(np.float32)

    out = np.zeros_like(
        score,
        dtype=np.float32,
    )

    for i in range(n):
        candidates = [
            (float(score[i, j]), j)
            for j in range(n)
            if (
                j != i
                and score[i, j] > min_score
            )
        ]

        candidates.sort(
            key=lambda z: z[0],
            reverse=True,
        )

        for s, j in candidates[:top_k]:
            out[i, j] = s

    out = np.maximum(
        out,
        out.T,
    )

    return out.astype(np.float32)


def normalize_rows(
    adjacency: np.ndarray,
    add_self_loop: bool = True,
) -> np.ndarray:
    x = adjacency.astype(
        np.float32,
    ).copy()

    if add_self_loop:
        np.fill_diagonal(x, 1.0)

    denom = x.sum(
        axis=1,
        keepdims=True,
    )

    denom = np.maximum(
        denom,
        EPS,
    )

    return (x / denom).astype(
        np.float32
    )


# ---------------------------------------------------------------------
# Complete graph construction
# ---------------------------------------------------------------------

def build_relation_graph(
    midi_path: Path,
    recurrence_threshold: float = 0.85,
    top_k: int = 4,
) -> dict[str, np.ndarray]:
    pm = pretty_midi.PrettyMIDI(
        str(midi_path)
    )

    notes = get_note_events(pm)

    if len(notes) < 5:
        raise ValueError(
            "too few non-drum notes"
        )

    boundaries = get_bar_boundaries(pm)

    if len(boundaries) < 3:
        raise ValueError(
            "too few bars"
        )

    bars = build_bars(
        notes,
        boundaries,
    )

    if len(bars) < 8:
        raise ValueError(
            "too few bars for graph construction"
        )

    nodes = build_four_bar_nodes(bars)

    if len(nodes) < 2:
        raise ValueError(
            "too few 4-bar graph nodes"
        )

    # -------------------------------------------------------------
    # Pairwise relation scores
    # -------------------------------------------------------------
    rhythm_score = pairwise_rhythm_scores(
        nodes
    )

    melody_score = pairwise_melody_scores(
        nodes
    )

    harmony_score = pairwise_harmony_scores(
        nodes
    )

    (
        structure_4bar,
        structure_8bar,
        structure_score,
    ) = pairwise_structure_scores(
        bars,
        nodes,
        recurrence_threshold=recurrence_threshold,
    )

    # Exact recurrence relations can be sparse.
    # Keep positive relations, then cap by top-k for a lightweight graph.
    rhythm = topk_symmetric(
        rhythm_score,
        top_k=top_k,
        min_score=0.0,
    )

    melody = topk_symmetric(
        melody_score,
        top_k=top_k,
        min_score=0.0,
    )

    harmony = topk_symmetric(
        harmony_score,
        top_k=top_k,
        min_score=0.0,
    )

    # Structure already has the 0.85 recurrence threshold built in.
    structure = topk_symmetric(
        structure_score,
        top_k=top_k,
        min_score=0.0,
    )

    # Equal initial relation weights.
    # The GNN can learn relation-specific transformations later.
    combined = (
        rhythm
        + melody
        + harmony
        + structure
    ).astype(np.float32)

    # -------------------------------------------------------------
    # Song-level analysis values
    # -------------------------------------------------------------
    bar_vectors = np.vstack([
        bar["bar_vector"]
        for bar in bars
    ])

    rhythm_signatures = [
        bar["rhythm_signature"]
        for bar in bars
    ]

    melody_signatures = [
        bar["melody_signature"]
        for bar in bars
    ]

    chord_sequence = [
        bar["chord"]
        for bar in bars
    ]

    song_features = {
        "num_bars":
            float(len(bars)),

        "num_notes":
            float(len(notes)),

        "notes_per_bar":
            float(len(notes) / len(bars)),

        "rhythm_pattern_recurrence":
            recurrence_from_unique_ratio(
                rhythm_signatures
            ),

        "melodic_pattern_recurrence":
            recurrence_from_unique_ratio(
                melody_signatures
            ),

        "pitch_class_entropy":
            global_pitch_class_entropy(
                notes
            ),

        "chord_bigram_recurrence":
            ngram_recurrence(
                chord_sequence,
                n=2,
            ),

        "chord_trigram_recurrence":
            ngram_recurrence(
                chord_sequence,
                n=3,
            ),

        "chord_quadgram_recurrence":
            ngram_recurrence(
                chord_sequence,
                n=4,
            ),

        "chord_unique_ratio":
            chord_unique_ratio(
                chord_sequence
            ),
    }

    for scale in (1, 2, 4, 8):
        block = block_vectors(
            bar_vectors,
            scale_bars=scale,
        )

        song_features[
            f"recurrence_{scale}bar"
        ] = recurrence_ratio(
            block,
            threshold=recurrence_threshold,
            min_index_distance=2,
        )

    block4 = block_vectors(
        bar_vectors,
        scale_bars=4,
    )

    song_features[
        "nonadjacent_recurrence_ratio"
    ] = recurrence_ratio(
        block4,
        threshold=recurrence_threshold,
        min_index_distance=2,
    )

    distances = recurrence_distances(
        block4,
        threshold=recurrence_threshold,
        min_index_distance=2,
    )

    song_features[
        "recurrence_distance_mean_blocks"
    ] = (
        float(np.mean(distances))
        if distances
        else float("nan")
    )

    novel_flags = novel_material_flags(
        block4,
        threshold=recurrence_threshold,
    )

    song_features[
        "novel_material_ratio_4bar"
    ] = (
        float(novel_flags.mean())
        if len(novel_flags)
        else float("nan")
    )

    # -------------------------------------------------------------
    # Per-node diagnostics
    # -------------------------------------------------------------
    node_count = len(nodes)

    distance_matrix = np.abs(
        np.arange(node_count)[:, None]
        - np.arange(node_count)[None, :]
    ).astype(np.int32)

    output = {
        # Main relations
        "rhythm":
            rhythm,

        "melody":
            melody,

        "harmony":
            harmony,

        "structure":
            structure,

        "combined":
            combined,

        # Row-normalized relations for message passing
        "rhythm_norm":
            normalize_rows(rhythm),

        "melody_norm":
            normalize_rows(melody),

        "harmony_norm":
            normalize_rows(harmony),

        "structure_norm":
            normalize_rows(structure),

        "combined_norm":
            normalize_rows(combined),

        # Structure decomposition
        "structure_4bar":
            structure_4bar,

        "structure_8bar":
            structure_8bar,

        "segment_distance_blocks":
            distance_matrix,

        # Node metadata
        "start_bar":
            np.asarray(
                [node["start_bar"] for node in nodes],
                dtype=np.int32,
            ),

        "start_time":
            np.asarray(
                [node["start_time"] for node in nodes],
                dtype=np.float32,
            ),

        "end_time":
            np.asarray(
                [node["end_time"] for node in nodes],
                dtype=np.float32,
            ),

        "novel_material_node":
            novel_flags[:node_count].astype(
                np.float32
            ),

        "notes_per_bar_node":
            np.asarray(
                [node["notes_per_bar"] for node in nodes],
                dtype=np.float32,
            ),

        "pitch_class_entropy_node":
            np.asarray(
                [node["pitch_class_entropy"] for node in nodes],
                dtype=np.float32,
            ),

        "chord_unique_ratio_node":
            np.asarray(
                [node["chord_unique_ratio"] for node in nodes],
                dtype=np.float32,
            ),
    }

    # Save every song-level scalar into the same NPZ.
    for key, value in song_features.items():
        output[key] = np.asarray(
            value,
            dtype=np.float32,
        )

    return output


def save_graph(
    midi_path: Path,
    out_path: Path,
    recurrence_threshold: float,
    top_k: int,
) -> None:
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph = build_relation_graph(
        midi_path,
        recurrence_threshold=recurrence_threshold,
        top_k=top_k,
    )

    np.savez_compressed(
        out_path,
        **graph,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    source = parser.add_mutually_exclusive_group(
        required=True
    )

    source.add_argument(
        "--midi",
        type=Path,
    )

    source.add_argument(
        "--midi-dir",
        type=Path,
    )

    parser.add_argument(
        "--out",
        type=Path,
        help="Output .npz for single-MIDI mode",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for folder mode",
    )

    parser.add_argument(
        "--recurrence-threshold",
        type=float,
        default=0.85,
        help="Cosine threshold for structural recurrence",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help=(
            "Maximum neighbors per relation and node. "
            "Use 0 to keep every positive relation."
        ),
    )

    args = parser.parse_args()

    if args.midi is not None:
        if args.out is None:
            output_path = args.midi.with_suffix(".npz")
        else:
            output_path = args.out

        save_graph(
            args.midi,
            output_path,
            recurrence_threshold=args.recurrence_threshold,
            top_k=args.top_k,
        )

        print(f"Saved: {output_path}")
        return

    if args.out_dir is None:
        raise SystemExit(
            "--out-dir is required with --midi-dir"
        )

    midis = collect_midis(
        args.midi_dir
    )

    print(
        f"MIDI files: {len(midis)}"
    )

    ok = 0
    skipped = 0

    for midi in midis:
        relative = midi.relative_to(
            args.midi_dir
        )

        out_path = (
            args.out_dir / relative
        ).with_suffix(".npz")

        try:
            save_graph(
                midi,
                out_path,
                recurrence_threshold=args.recurrence_threshold,
                top_k=args.top_k,
            )
            ok += 1

        except Exception as exc:
            warnings.warn(
                f"[SKIP] {midi}: "
                f"{type(exc).__name__}: {exc}"
            )
            skipped += 1

    print(f"Saved: {ok}")
    print(f"Skipped: {skipped}")
    print(
        f"Output directory: {args.out_dir}"
    )


if __name__ == "__main__":
    main()
