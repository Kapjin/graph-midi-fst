
#!/usr/bin/env python3
"""
compare_real_fake_midi.py

Human-composed vs AI-composed MIDI exploratory analysis.

Features
--------
Metadata / diagnostics
- num_bars
- num_notes
- notes_per_bar

Structure
- recurrence_1bar
- recurrence_2bar
- recurrence_4bar
- recurrence_8bar
- nonadjacent_recurrence_ratio
- novel_material_ratio_4bar
- recurrence_distance_mean_blocks

Rhythm
- rhythm_pattern_recurrence

Melody
- melodic_pattern_recurrence
- pitch_class_entropy

Harmony
- chord_bigram_recurrence
- chord_trigram_recurrence
- chord_quadgram_recurrence
- chord_unique_ratio
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
from scipy.stats import mannwhitneyu
from tqdm import tqdm

MIDI_EXTS = {".mid", ".midi"}
EPS = 1e-12


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
        return np.zeros((0, 0), dtype=float)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norm, EPS)
    return x @ x.T


def get_note_events(pm: pretty_midi.PrettyMIDI) -> list[tuple[float, float, int]]:
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.end <= n.start:
                continue
            notes.append((float(n.start), float(n.end), int(n.pitch)))
    notes.sort(key=lambda z: (z[0], z[2], z[1]))
    return notes


def get_bar_boundaries(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    try:
        downbeats = np.asarray(pm.get_downbeats(), dtype=float)
    except Exception:
        downbeats = np.array([], dtype=float)

    end_time = float(pm.get_end_time())

    if len(downbeats) >= 2:
        if downbeats[0] > 1e-6:
            downbeats = np.r_[0.0, downbeats]
        if downbeats[-1] < end_time:
            bar_dur = np.median(np.diff(downbeats[-min(8, len(downbeats)):]))
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
            beat_dur = np.median(np.diff(beats[-min(16, len(beats)):]))
            bar_dur = 4.0 * beat_dur
            t = bars[-1]
            extra = []
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


def bar_pitch_class_hist(
    bar_notes: list[tuple[float, float, int]]
) -> np.ndarray:
    hist = np.zeros(12, dtype=float)
    for _, _, p in bar_notes:
        hist[p % 12] += 1.0
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def bar_onset_pattern(
    bar_notes: list[tuple[float, float, int]],
    start: float,
    end: float,
    bins: int = 16,
) -> np.ndarray:
    pat = np.zeros(bins, dtype=float)
    dur = max(end - start, EPS)
    for onset, _, _ in bar_notes:
        pos = (onset - start) / dur
        idx = min(bins - 1, max(0, int(np.floor(pos * bins))))
        pat[idx] = 1.0
    return pat


def bar_topnote_interval_signature(
    bar_notes: list[tuple[float, float, int]],
    onset_tol: float = 0.04,
    max_intervals: int = 12,
) -> tuple[int, ...] | None:
    if len(bar_notes) < 2:
        return None

    groups = []
    cur_onset = None
    cur_pitches = []

    for onset, _, pitch in sorted(bar_notes, key=lambda z: (z[0], z[2])):
        if cur_onset is None or abs(onset - cur_onset) <= onset_tol:
            if cur_onset is None:
                cur_onset = onset
            cur_pitches.append(pitch)
        else:
            groups.append(max(cur_pitches))
            cur_onset = onset
            cur_pitches = [pitch]
    if cur_pitches:
        groups.append(max(cur_pitches))

    if len(groups) < 3:
        return None

    intervals = np.diff(groups).astype(int)
    intervals = np.clip(intervals, -24, 24)
    if len(intervals) > max_intervals:
        intervals = intervals[:max_intervals]
    return tuple(int(x) for x in intervals)


MAJOR_TEMPLATE = np.zeros(12, dtype=float)
MAJOR_TEMPLATE[[0, 4, 7]] = 1.0
MINOR_TEMPLATE = np.zeros(12, dtype=float)
MINOR_TEMPLATE[[0, 3, 7]] = 1.0


def estimate_bar_chord(
    bar_notes: list[tuple[float, float, int]]
) -> str | None:
    if not bar_notes:
        return None

    pc = np.zeros(12, dtype=float)
    for onset, end, pitch in bar_notes:
        dur = max(end - onset, 0.02)
        pc[pitch % 12] += dur

    if pc.sum() <= EPS:
        return None

    pc /= pc.sum()
    scores = []
    labels = []
    names = ["C", "C#", "D", "D#", "E", "F",
             "F#", "G", "G#", "A", "A#", "B"]

    for root in range(12):
        maj = np.roll(MAJOR_TEMPLATE, root)
        min_ = np.roll(MINOR_TEMPLATE, root)

        maj_score = float(
            np.dot(pc, maj) /
            (np.linalg.norm(pc) * np.linalg.norm(maj) + EPS)
        )
        min_score = float(
            np.dot(pc, min_) /
            (np.linalg.norm(pc) * np.linalg.norm(min_) + EPS)
        )

        scores.extend([maj_score, min_score])
        labels.extend([f"{names[root]}:maj", f"{names[root]}:min"])

    return labels[int(np.argmax(scores))]


def chord_sequence(
    notes: list[tuple[float, float, int]],
    boundaries: np.ndarray,
) -> list[str | None]:
    out = []
    for i in range(len(boundaries) - 1):
        a, b = float(boundaries[i]), float(boundaries[i + 1])
        if b <= a:
            continue
        bn = notes_in_range(notes, a, b)
        out.append(estimate_bar_chord(bn))
    return out


def chord_unique_ratio(seq: list[str | None]) -> float:
    clean = [x for x in seq if x is not None]
    if not clean:
        return float("nan")
    return float(len(set(clean)) / len(clean))


def ngram_recurrence(seq: list[str | None], n: int) -> float:
    clean = [x for x in seq if x is not None]
    if len(clean) < n:
        return float("nan")
    grams = [tuple(clean[i:i+n]) for i in range(len(clean) - n + 1)]
    return float(1.0 - len(set(grams)) / len(grams))


def build_bar_features(
    notes: list[tuple[float, float, int]],
    boundaries: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, ...]], list[tuple[int, ...] | None]]:
    vectors = []
    rhythm_sigs = []
    melodic_sigs = []

    for i in range(len(boundaries) - 1):
        a, b = float(boundaries[i]), float(boundaries[i + 1])
        if b <= a:
            continue
        bn = notes_in_range(notes, a, b)

        pc = bar_pitch_class_hist(bn)
        onset = bar_onset_pattern(bn, a, b, bins=16)
        vectors.append(np.r_[pc, onset])

        rhythm_sigs.append(tuple(int(v) for v in onset))
        melodic_sigs.append(bar_topnote_interval_signature(bn))

    if not vectors:
        return np.zeros((0, 28), dtype=float), [], []
    return np.vstack(vectors), rhythm_sigs, melodic_sigs


def block_vectors(bar_vectors: np.ndarray, scale_bars: int) -> np.ndarray:
    n = len(bar_vectors)
    if n < scale_bars:
        return np.zeros((0, bar_vectors.shape[1]), dtype=float)

    blocks = []
    for s in range(0, n - scale_bars + 1, scale_bars):
        blocks.append(bar_vectors[s:s + scale_bars].mean(axis=0))
    return np.vstack(blocks) if blocks else np.zeros((0, bar_vectors.shape[1]))


def recurrence_ratio(
    vectors: np.ndarray,
    threshold: float = 0.85,
    min_index_distance: int = 1,
) -> float:
    if len(vectors) < 2:
        return float("nan")

    ssm = cosine_matrix(vectors)
    vals = []
    n = len(vectors)

    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) < min_index_distance:
                continue
            vals.append(ssm[i, j] >= threshold)

    if not vals:
        return float("nan")
    return float(np.mean(vals))


def recurrence_distances(
    vectors: np.ndarray,
    threshold: float = 0.85,
    min_index_distance: int = 2,
) -> list[int]:
    if len(vectors) < 2:
        return []

    ssm = cosine_matrix(vectors)
    out = []
    n = len(vectors)

    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            if d < min_index_distance:
                continue
            if ssm[i, j] >= threshold:
                out.append(d)

    return out


def novel_material_ratio(
    vectors: np.ndarray,
    threshold: float = 0.85,
) -> float:
    if len(vectors) == 0:
        return float("nan")
    if len(vectors) == 1:
        return 1.0

    ssm = cosine_matrix(vectors)
    novel = 1

    for i in range(1, len(vectors)):
        if np.max(ssm[i, :i]) < threshold:
            novel += 1

    return float(novel / len(vectors))


def recurrence_from_unique_ratio(signatures) -> float:
    sigs = [s for s in signatures if s is not None]
    if not sigs:
        return float("nan")
    unique_ratio = len(set(sigs)) / len(sigs)
    return float(1.0 - unique_ratio)


def global_pitch_class_entropy(
    notes: list[tuple[float, float, int]]
) -> float:
    if not notes:
        return float("nan")

    hist = np.zeros(12, dtype=float)
    for _, _, p in notes:
        hist[p % 12] += 1

    hist /= max(hist.sum(), EPS)
    return shannon_entropy(hist)


def extract_features(
    midi_path: Path,
    recurrence_threshold: float = 0.85,
) -> tuple[dict, np.ndarray | None]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = get_note_events(pm)

    if len(notes) < 5:
        raise ValueError("too few non-drum notes")

    boundaries = get_bar_boundaries(pm)
    if len(boundaries) < 3:
        raise ValueError("too few bars")

    bar_vecs, rhythm_sigs, melodic_sigs = build_bar_features(notes, boundaries)
    if len(bar_vecs) < 2:
        raise ValueError("too few valid bars")

    chords = chord_sequence(notes, boundaries)

    out = {
        # Metadata / diagnostics
        "num_bars": len(bar_vecs),
        "num_notes": len(notes),
        "notes_per_bar": len(notes) / len(bar_vecs),

        # Rhythm
        "rhythm_pattern_recurrence": recurrence_from_unique_ratio(rhythm_sigs),

        # Melody
        "melodic_pattern_recurrence": recurrence_from_unique_ratio(melodic_sigs),
        "pitch_class_entropy": global_pitch_class_entropy(notes),

        # Harmony
        "chord_bigram_recurrence": ngram_recurrence(chords, n=2),
        "chord_trigram_recurrence": ngram_recurrence(chords, n=3),
        "chord_quadgram_recurrence": ngram_recurrence(chords, n=4),
        "chord_unique_ratio": chord_unique_ratio(chords),
    }

    # Multi-scale structure recurrence
    for scale in (1, 2, 4, 8):
        bv = block_vectors(bar_vecs, scale)
        out[f"recurrence_{scale}bar"] = recurrence_ratio(
            bv,
            threshold=recurrence_threshold,
            min_index_distance=2,  # exclude immediately adjacent blocks
        )

    # 4-bar blocks: non-adjacent recurrence
    bv4 = block_vectors(bar_vecs, 4)
    out["nonadjacent_recurrence_ratio"] = recurrence_ratio(
        bv4,
        threshold=recurrence_threshold,
        min_index_distance=2,  # exclude only adjacent 4-bar blocks
    )

    dists = recurrence_distances(
        bv4,
        threshold=recurrence_threshold,
        min_index_distance=2,
    )
    out["recurrence_distance_mean_blocks"] = (
        float(np.mean(dists)) if dists else float("nan")
    )

    out["novel_material_ratio_4bar"] = novel_material_ratio(
        bv4,
        threshold=recurrence_threshold,
    )

    ssm4 = cosine_matrix(bv4) if len(bv4) >= 2 else None
    return out, ssm4


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return float("nan")

    ys = np.sort(y)
    greater = 0
    less = 0

    for v in x:
        less += np.searchsorted(ys, v, side="left")
        greater += len(ys) - np.searchsorted(ys, v, side="right")

    return float((less - greater) / (len(x) * len(y)))


def bootstrap_median_diff_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = np.median(xb) - np.median(yb)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def bh_fdr(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]

    if len(pv) == 0:
        return q.tolist()

    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    tmp = np.empty_like(adj)
    tmp[order] = adj
    q[valid] = tmp

    return q.tolist()


def plot_boxplot(
    df: pd.DataFrame,
    feature: str,
    labels: list[str],
    out_path: Path,
) -> None:
    values = [df.loc[df["label"] == label, feature].dropna().values for label in labels]
    valid = [(label, value) for label, value in zip(labels, values) if len(value)]

    if len(valid) < 2:
        return

    plt.figure(figsize=(6, 4))
    plt.boxplot(
        [value for _, value in valid],
        tick_labels=[label for label, _ in valid],
        showfliers=False,
    )
    plt.ylabel(feature)
    plt.title(feature)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_ssm(ssm: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.imshow(ssm, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    plt.xlabel("4-bar block index")
    plt.ylabel("4-bar block index")
    plt.title(title)
    plt.colorbar(label="cosine similarity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def analyze_group(
    paths: list[Path],
    label: str,
    recurrence_threshold: float,
    ssm_dir: Path,
    save_ssm_n: int,
) -> list[dict]:
    rows = []
    saved_ssm = 0

    for path in tqdm(paths, desc=label):
        try:
            feat, ssm = extract_features(
                path,
                recurrence_threshold=recurrence_threshold,
            )

            row = {
                "label": label,
                "file": str(path),
                **feat,
            }
            rows.append(row)

            if (
                ssm is not None
                and saved_ssm < save_ssm_n
                and ssm.shape[0] >= 2
            ):
                class_dir = ssm_dir / safe_label(label)
                class_dir.mkdir(parents=True, exist_ok=True)
                safe_name = f"{saved_ssm:04d}_{path.stem}.png"
                save_ssm(ssm, path.name, class_dir / safe_name)
                saved_ssm += 1

        except Exception as e:
            warnings.warn(f"[SKIP] {path}: {type(e).__name__}: {e}")

    return rows


def compute_stats(df: pd.DataFrame, labels: list[str], n_boot: int) -> pd.DataFrame:
    ignore = {"label", "file"}
    features = [
        c for c in df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []
    pvals = []

    for f in features:
        for index, group_a in enumerate(labels):
            for group_b in labels[index + 1:]:
                real = df.loc[df["label"] == group_a, f].dropna().to_numpy(float)
                fake = df.loc[df["label"] == group_b, f].dropna().to_numpy(float)

                if len(real) == 0 or len(fake) == 0:
                    continue

                try:
                    u, p = mannwhitneyu(real, fake, alternative="two-sided")
                except Exception:
                    u, p = float("nan"), float("nan")

                delta = cliffs_delta(real, fake)
                ci_lo, ci_hi = bootstrap_median_diff_ci(
                    real, fake, n_boot=n_boot
                )

                rows.append({
                    "feature": f,
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_group_a": len(real),
                    "n_group_b": len(fake),
                    "group_a_median": float(np.median(real)),
                    "group_b_median": float(np.median(fake)),
                    "group_a_iqr": float(np.percentile(real, 75) - np.percentile(real, 25)),
                    "group_b_iqr": float(np.percentile(fake, 75) - np.percentile(fake, 25)),
                    "median_diff_group_a_minus_group_b": float(np.median(real) - np.median(fake)),
                    "median_diff_ci95_low": ci_lo,
                    "median_diff_ci95_high": ci_hi,
                    "mannwhitney_u": float(u),
                    "p_value": float(p),
                    "cliffs_delta_group_a_minus_group_b": delta,
                    "abs_cliffs_delta": abs(delta) if np.isfinite(delta) else float("nan"),
                })
                pvals.append(float(p))

    out = pd.DataFrame(rows)

    if len(out):
        out["q_value_bh"] = bh_fdr(pvals)
        out = out.sort_values(
            ["abs_cliffs_delta", "q_value_bh"],
            ascending=[False, True],
        ).reset_index(drop=True)

    return out


def safe_label(label: str) -> str:
    value = "".join(character if character.isalnum() or character in "-_" else "_" for character in label)
    return value.strip("_") or "group"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        nargs=2,
        action="append",
        metavar=("LABEL", "DIR"),
        required=True,
        help="Group label and MIDI directory; repeat for each group",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--per-group-count",
        type=int,
        help="Analyze this many MIDI files from every group; omit to use all files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --per-group-count is specified",
    )

    parser.add_argument(
        "--recurrence-threshold",
        type=float,
        default=0.85,
        help="Cosine-similarity threshold for recurrence (default: 0.85)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap repetitions for median-difference CI",
    )
    parser.add_argument(
        "--save-ssm",
        type=int,
        default=20,
        help="Number of SSM figures to save per class",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    box_dir = args.out_dir / "boxplots"
    ssm_dir = args.out_dir / "ssm"
    box_dir.mkdir(parents=True, exist_ok=True)
    ssm_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    rows = []
    rng = np.random.default_rng(args.seed)
    for label, directory in args.group:
        root = Path(directory)
        files = collect_midis(root)
        if args.per_group_count is not None:
            if args.per_group_count < 1:
                parser.error("--per-group-count must be at least 1")
            if args.per_group_count > len(files):
                parser.error(
                    f"Group '{label}' has only {len(files)} MIDI files; "
                    f"cannot select {args.per_group_count}"
                )
            selected_indices = rng.choice(
                len(files), size=args.per_group_count, replace=False
            )
            files = [files[int(index)] for index in selected_indices]
        print(f"{label} MIDI files: {len(files)}")
        labels.append(label)
        rows.extend(
            analyze_group(
                files,
                label,
                args.recurrence_threshold,
                ssm_dir,
                args.save_ssm,
            )
        )

    if len(labels) < 2:
        parser.error("at least two --group arguments are required")
    if len(labels) != len(set(labels)):
        parser.error("each --group label must be unique")

    df = pd.DataFrame(rows)

    feature_csv = args.out_dir / "feature_values.csv"
    df.to_csv(feature_csv, index=False)

    stats = compute_stats(df, labels, n_boot=args.bootstrap)

    stats_csv = args.out_dir / "feature_stats.csv"
    stats.to_csv(stats_csv, index=False)

    for feature in stats["feature"].drop_duplicates().tolist():
        plot_boxplot(df, feature, labels, box_dir / f"{feature}.png")

    print()
    print(f"Saved: {feature_csv}")
    print(f"Saved: {stats_csv}")
    print(f"Saved boxplots: {box_dir}")
    print(f"Saved SSMs: {ssm_dir}")
    print()

    if len(stats):
        cols = ["feature", "group_a", "group_b", "cliffs_delta_group_a_minus_group_b", "q_value_bh"]
        print("Features by |Cliff's delta|:")
        print(stats[cols].to_string(index=False))


if __name__ == "__main__":
    main()
