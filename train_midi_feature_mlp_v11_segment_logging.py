#!/usr/bin/env python3
"""Train and run an MLP on bar-segmented symbolic MIDI features.

This script reuses the feature definitions from compare_real_fake_midi.py.

Feature selection is controlled near the top of THIS FILE rather than being
forced from the command line.

Two modes are supported:

1) FEATURE_SELECTION_MODE = "manual"
   Use exactly the feature names listed in MANUAL_FEATURES.

2) FEATURE_SELECTION_MODE = "auto_group_topk"
   Rank features ONLY on the training split using absolute Cliff's delta, but
   select the top-k separately inside the four musical aspects:
   structural / rhythm / melody / harmony.

This keeps the four aspects represented independently instead of allowing one
aspect with many features to dominate a single global top-k ranking.

Commands
--------
make-split
    Create a Human/AI train/validation split manifest.

train
    Split each MIDI into bar segments, extract one feature vector per segment,
    fit train-only median imputation and standardization, and train MLP.

infer
    Classify one MIDI file or evaluate a labeled MIDI directory.

Expected companion file
-----------------------
Put compare_real_fake_midi.py in the same directory. The MLP imports
extract_features() and cliffs_delta() from that file so that classifier and
exploratory analysis use the same feature definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
    import pretty_midi
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Activate the MIDI project environment and install "
        "numpy, pretty_midi, and torch."
    ) from exc

try:
    import compare_real_fake_midi as midi_feature_impl
    from compare_real_fake_midi import cliffs_delta, extract_features
except ImportError as exc:
    raise SystemExit(
        "Could not import compare_real_fake_midi.py. Put this script in the same "
        "directory as compare_real_fake_midi.py."
    ) from exc


MIDI_FEATURE_MLP_BUILD = "2026-08-26-global-midi-feature-mlp-v11-segment-logging"
MIDI_EXTS = {".mid", ".midi"}

# =============================================================================
# SEGMENT CONFIG
# =============================================================================
# Number of bars per non-overlapping segment.
# Set to 0 to use the previous whole-song behavior.
SEGMENT_BARS = 32

# Final train/inference summaries are appended here automatically.
# The file is created in the SAME directory as this Python script.
RESULT_LOG_PATH = Path(__file__).resolve().parent / "midi_feature_mlp_results_2608261903.txt"

# =============================================================================
# FEATURE SELECTION CONFIG
# =============================================================================
# Choose ONE:
#   "manual"          -> use MANUAL_FEATURES exactly as written below.
#   "auto_group_topk" -> choose top-k separately within each of the four aspects,
#                        using |Cliff's delta| computed ONLY from the training set.
#
FEATURE_SELECTION_MODE = "manual"

# Used only when FEATURE_SELECTION_MODE == "manual".
# Add/remove/reorder feature names here directly.
MANUAL_FEATURES = [
    "recurrence_1bar",
    "recurrence_2bar",
    "recurrence_4bar",
    "recurrence_8bar",
    "novel_material_ratio_4bar",
    "rhythm_pattern_recurrence",
    "melodic_pattern_recurrence",
    "pitch_class_entropy",
    "chord_bigram_recurrence",
    "chord_trigram_recurrence",
    "chord_quadgram_recurrence",
    "chord_unique_ratio",
]

# Used only when FEATURE_SELECTION_MODE == "auto_group_topk".
# 0 means "use every available feature in this group".
AUTO_TOP_K_BY_GROUP = {
    "structural": 3,
    "rhythm": 1,
    "melody": 2,
    "harmony": 2,
}

# -----------------------------------------------------------------------------
# Feature groups
# -----------------------------------------------------------------------------

FEATURE_GROUPS = {
    "structural": (
        "recurrence_1bar",
        "recurrence_2bar",
        "recurrence_4bar",
        "recurrence_8bar",
        "nonadjacent_recurrence_ratio",
        "novel_material_ratio_4bar",
        "recurrence_distance_mean_blocks",
    ),
    "rhythm": (
        "rhythm_pattern_recurrence",
    ),
    "melody": (
        "melodic_pattern_recurrence",
        "pitch_class_entropy",
    ),
    "harmony": (
        "chord_bigram_recurrence",
        "chord_trigram_recurrence",
        "chord_quadgram_recurrence",
        "chord_unique_ratio",
    ),
}

# These remain available for reference, but are not part of the four-aspect
# selector because they are dataset/length diagnostics.
DIAGNOSTIC_FEATURES = (
    "num_bars",
    "num_notes",
    "notes_per_bar",
)

ALL_MUSICAL_FEATURES = tuple(
    feature
    for group_features in FEATURE_GROUPS.values()
    for feature in group_features
)


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


class TeeStream:
    """Write print output to both the terminal and a log file."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def make_tensorboard_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise SystemExit(
            "TensorBoard logging requires tensorboard. "
            "Install it with: python3 -m pip install tensorboard"
        ) from exc

    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU")
    return device


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def append_result_log(record: dict[str, Any]) -> None:
    """Save only the minimal final inference result next to this script."""
    event = record.get("event")

    # Do not write a training summary to the final result file.
    if event == "train":
        return

    if event == "infer_folder":
        lines = [
            f"Label: {record.get('label', '-')}",
            f"Evaluated: {record.get('evaluated', '-')}",
            f"Correct: {record.get('correct', '-')}",
            f"Accuracy: {float(record.get('accuracy', 0.0)):.6f}",
            f"Failed: {record.get('failed', 0)}",
        ]
    elif event == "infer_single":
        lines = [
            f"Prediction: {record.get('prediction', '-')}",
            f"AI probability: {float(record.get('ai_probability', 0.0)):.6f}",
            f"Human probability: {float(record.get('human_probability', 0.0)):.6f}",
        ]
    else:
        return

    RESULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n\n")

    print(f"Final result saved: {RESULT_LOG_PATH}")



def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def discover_midis(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"MIDI directory does not exist: {directory}")
    paths = sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in MIDI_EXTS
    )
    if not paths:
        raise ValueError(f"No .mid or .midi files found under: {directory}")
    return paths


def discover_many_midis(directories: Sequence[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for directory in directories:
        for path in discover_midis(directory):
            unique[str(path)] = path
    return sorted(unique.values())


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format") != "midi-fst-split-v2":
        raise ValueError(
            f"Unsupported split manifest: {path}. Expected format midi-fst-split-v2"
        )
    for split_name in ("train", "validation"):
        records = manifest.get(split_name)
        if not records:
            raise ValueError(f"Manifest split is empty: {split_name}")
        for record in records:
            midi_path = Path(record["path"])
            if not midi_path.is_file():
                raise FileNotFoundError(f"Manifest MIDI file is missing: {midi_path}")
            if int(record["label"]) not in (0, 1):
                raise ValueError(f"Invalid label in manifest: {record}")
    return manifest


# -----------------------------------------------------------------------------
# Split creation
# -----------------------------------------------------------------------------


def _parse_dir_specs(
    specs: Sequence[Sequence[str]],
    option_name: str,
) -> list[tuple[Path, int]]:
    parsed: list[tuple[Path, int]] = []

    for raw_dir, raw_count in specs:
        directory = Path(raw_dir)
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(
                f"{option_name} count must be an integer: {raw_count!r}"
            ) from exc

        if count < 2:
            raise ValueError(
                f"{option_name} count must be at least 2 for train/validation split: "
                f"{directory} -> {count}"
            )
        parsed.append((directory, count))

    return parsed


def command_make_split(args: argparse.Namespace) -> None:
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    real_specs = _parse_dir_specs(args.real_dir, "--real-dir")
    fake_specs = _parse_dir_specs(args.fake_dir, "--fake-dir")

    # Check Human/AI source overlap before sampling.
    real_all = discover_many_midis([directory for directory, _count in real_specs])
    fake_all = discover_many_midis([directory for directory, _count in fake_specs])
    overlap = set(real_all) & set(fake_all)
    if overlap:
        raise ValueError(
            "Real and fake directories contain the same resolved MIDI path: "
            f"{next(iter(overlap))}"
        )

    rng = random.Random(args.seed)

    def select_and_divide(
        specs: Sequence[tuple[Path, int]],
        label: int,
        class_name: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        all_train: list[dict[str, Any]] = []
        all_validation: list[dict[str, Any]] = []
        source_summary: list[dict[str, Any]] = []
        already_selected: set[str] = set()

        for directory, requested_count in specs:
            paths = discover_midis(directory)

            # Avoid selecting the same resolved MIDI twice if source directories overlap.
            available = [
                path for path in paths
                if str(path.resolve()) not in already_selected
            ]

            if len(available) < requested_count:
                raise ValueError(
                    f"Requested {requested_count} {class_name} MIDI files from "
                    f"{directory}, but only {len(available)} unique files are available"
                )

            selected = rng.sample(available, requested_count)
            for path in selected:
                already_selected.add(str(path.resolve()))

            rng.shuffle(selected)
            validation_count = max(
                1, int(round(len(selected) * args.val_ratio))
            )
            if validation_count >= len(selected):
                raise ValueError(
                    f"--val-ratio leaves no training songs for {directory}"
                )

            validation_paths = selected[:validation_count]
            train_paths = selected[validation_count:]
            source_name = str(directory.resolve())

            all_train.extend(
                {
                    "path": str(path),
                    "label": label,
                    "source": source_name,
                }
                for path in train_paths
            )
            all_validation.extend(
                {
                    "path": str(path),
                    "label": label,
                    "source": source_name,
                }
                for path in validation_paths
            )
            source_summary.append(
                {
                    "dir": source_name,
                    "requested": int(requested_count),
                    "train": int(len(train_paths)),
                    "validation": int(len(validation_paths)),
                }
            )

        return all_train, all_validation, source_summary

    real_train, real_validation, real_summary = select_and_divide(
        real_specs, 0, "Human"
    )
    fake_train, fake_validation, fake_summary = select_and_divide(
        fake_specs, 1, "AI"
    )

    train_records = real_train + fake_train
    validation_records = real_validation + fake_validation
    rng.shuffle(train_records)
    rng.shuffle(validation_records)

    manifest = {
        "format": "midi-fst-split-v2",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "real_dirs": real_summary,
        "fake_dirs": fake_summary,
        "train": train_records,
        "validation": validation_records,
    }
    save_json(args.output, manifest)

    print(f"Split saved: {args.output.resolve()}")
    print("Human sources:")
    for item in real_summary:
        print(
            f"  {item['dir']} | selected={item['requested']} "
            f"train={item['train']} validation={item['validation']}"
        )
    print("AI sources:")
    for item in fake_summary:
        print(
            f"  {item['dir']} | selected={item['requested']} "
            f"train={item['train']} validation={item['validation']}"
        )
    print(
        f"Train | Human={len(real_train)}, AI={len(fake_train)}, "
        f"total={len(train_records)}"
    )
    print(
        f"Validation | Human={len(real_validation)}, AI={len(fake_validation)}, "
        f"total={len(validation_records)}"
    )


# -----------------------------------------------------------------------------
# Feature extraction / selection
# -----------------------------------------------------------------------------


@dataclass
class RawFeatureRecord:
    path: str
    label: int
    features: dict[str, float]


@dataclass
class SegmentFeature:
    start_bar: int
    end_bar: int
    features: dict[str, float]


def feature_group_of(feature: str) -> str:
    for group_name, group_features in FEATURE_GROUPS.items():
        if feature in group_features:
            return group_name
    if feature in DIAGNOSTIC_FEATURES:
        return "diagnostic"
    return "unknown"


def validate_feature_config() -> None:
    valid_modes = {"manual", "auto_group_topk"}
    if FEATURE_SELECTION_MODE not in valid_modes:
        raise ValueError(
            f"FEATURE_SELECTION_MODE must be one of {sorted(valid_modes)}, "
            f"got {FEATURE_SELECTION_MODE!r}"
        )

    known = set(ALL_MUSICAL_FEATURES) | set(DIAGNOSTIC_FEATURES)

    if FEATURE_SELECTION_MODE == "manual":
        if not MANUAL_FEATURES:
            raise ValueError("MANUAL_FEATURES must contain at least one feature")
        unknown = [feature for feature in MANUAL_FEATURES if feature not in known]
        if unknown:
            raise ValueError(
                "Unknown feature name(s) in MANUAL_FEATURES: " + ", ".join(unknown)
            )
        duplicates = [
            feature for feature in MANUAL_FEATURES if MANUAL_FEATURES.count(feature) > 1
        ]
        if duplicates:
            raise ValueError(
                "Duplicate feature name(s) in MANUAL_FEATURES: "
                + ", ".join(sorted(set(duplicates)))
            )

    if FEATURE_SELECTION_MODE == "auto_group_topk":
        unknown_groups = [
            group for group in AUTO_TOP_K_BY_GROUP if group not in FEATURE_GROUPS
        ]
        if unknown_groups:
            raise ValueError(
                "Unknown group name(s) in AUTO_TOP_K_BY_GROUP: "
                + ", ".join(unknown_groups)
            )
        missing_groups = [
            group for group in FEATURE_GROUPS if group not in AUTO_TOP_K_BY_GROUP
        ]
        if missing_groups:
            raise ValueError(
                "AUTO_TOP_K_BY_GROUP is missing group(s): " + ", ".join(missing_groups)
            )
        for group, top_k in AUTO_TOP_K_BY_GROUP.items():
            if int(top_k) < 0:
                raise ValueError(
                    f"AUTO_TOP_K_BY_GROUP[{group!r}] must be 0 or greater"
                )


def validate_segment_config() -> None:
    if SEGMENT_BARS < 0:
        raise ValueError("SEGMENT_BARS must be 0 or greater")
    if SEGMENT_BARS == 1:
        raise ValueError("SEGMENT_BARS must be 0 or at least 2")

    if FEATURE_SELECTION_MODE != "manual" or SEGMENT_BARS == 0:
        return

    minimum_bars = {
        "recurrence_1bar": 3,
        "recurrence_2bar": 6,
        "recurrence_4bar": 12,
        "recurrence_8bar": 24,
        "nonadjacent_recurrence_ratio": 12,
        "novel_material_ratio_4bar": 4,
        "chord_bigram_recurrence": 2,
        "chord_trigram_recurrence": 3,
        "chord_quadgram_recurrence": 4,
    }
    required = max((minimum_bars.get(feature, 2) for feature in MANUAL_FEATURES), default=2)
    if SEGMENT_BARS < required:
        raise ValueError(
            f"SEGMENT_BARS={SEGMENT_BARS} is too short for the current manual features. "
            f"Use at least {required} bars, or remove the feature that requires longer context."
        )


def _extract_features_from_bar_segment(
    notes: list[tuple[float, float, int]],
    boundaries: np.ndarray,
    recurrence_threshold: float,
) -> dict[str, float]:
    """Apply the same feature formulas as compare_real_fake_midi.extract_features()."""
    if len(notes) < 5:
        raise ValueError("too few non-drum notes")
    if len(boundaries) < 3:
        raise ValueError("too few bars")

    bar_vecs, rhythm_sigs, melodic_sigs = midi_feature_impl.build_bar_features(
        notes, boundaries
    )
    if len(bar_vecs) < 2:
        raise ValueError("too few valid bars")

    chords = midi_feature_impl.chord_sequence(notes, boundaries)

    out: dict[str, float] = {
        "num_bars": float(len(bar_vecs)),
        "num_notes": float(len(notes)),
        "notes_per_bar": float(len(notes) / len(bar_vecs)),
        "rhythm_pattern_recurrence": float(
            midi_feature_impl.recurrence_from_unique_ratio(rhythm_sigs)
        ),
        "melodic_pattern_recurrence": float(
            midi_feature_impl.recurrence_from_unique_ratio(melodic_sigs)
        ),
        "pitch_class_entropy": float(
            midi_feature_impl.global_pitch_class_entropy(notes)
        ),
        "chord_bigram_recurrence": float(
            midi_feature_impl.ngram_recurrence(chords, n=2)
        ),
        "chord_trigram_recurrence": float(
            midi_feature_impl.ngram_recurrence(chords, n=3)
        ),
        "chord_quadgram_recurrence": float(
            midi_feature_impl.ngram_recurrence(chords, n=4)
        ),
        "chord_unique_ratio": float(midi_feature_impl.chord_unique_ratio(chords)),
    }

    for scale in (1, 2, 4, 8):
        block_vectors = midi_feature_impl.block_vectors(bar_vecs, scale)
        out[f"recurrence_{scale}bar"] = float(
            midi_feature_impl.recurrence_ratio(
                block_vectors,
                threshold=recurrence_threshold,
                min_index_distance=2,
            )
        )

    bv4 = midi_feature_impl.block_vectors(bar_vecs, 4)
    out["nonadjacent_recurrence_ratio"] = float(
        midi_feature_impl.recurrence_ratio(
            bv4,
            threshold=recurrence_threshold,
            min_index_distance=2,
        )
    )

    distances = midi_feature_impl.recurrence_distances(
        bv4,
        threshold=recurrence_threshold,
        min_index_distance=2,
    )
    out["recurrence_distance_mean_blocks"] = (
        float(np.mean(distances)) if distances else float("nan")
    )
    out["novel_material_ratio_4bar"] = float(
        midi_feature_impl.novel_material_ratio(
            bv4,
            threshold=recurrence_threshold,
        )
    )
    return out


def extract_segment_features(
    midi_path: Path,
    recurrence_threshold: float,
    segment_bars: int,
) -> tuple[list[SegmentFeature], list[dict[str, Any]]]:
    """Split one MIDI by bar boundaries and return one feature vector per segment."""
    if segment_bars == 0:
        feature_dict, _ssm = extract_features(
            midi_path,
            recurrence_threshold=recurrence_threshold,
        )
        converted = {key: float(value) for key, value in feature_dict.items()}
        return [SegmentFeature(0, int(converted.get("num_bars", 0)), converted)], []

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = midi_feature_impl.get_note_events(pm)
    boundaries = np.asarray(midi_feature_impl.get_bar_boundaries(pm), dtype=float)
    total_bars = len(boundaries) - 1
    if total_bars < 2:
        raise ValueError("too few bars")

    segments: list[SegmentFeature] = []
    failures: list[dict[str, Any]] = []

    for start_bar in range(0, total_bars, segment_bars):
        end_bar = min(start_bar + segment_bars, total_bars)
        bar_count = end_bar - start_bar

        # A one-bar remainder cannot support the existing extractor's minimum.
        if bar_count < 2:
            continue

        start_time = float(boundaries[start_bar])
        end_time = float(boundaries[end_bar])
        segment_notes = [note for note in notes if start_time <= note[0] < end_time]
        segment_boundaries = boundaries[start_bar : end_bar + 1]

        try:
            feature_dict = _extract_features_from_bar_segment(
                segment_notes,
                segment_boundaries,
                recurrence_threshold,
            )
            segments.append(
                SegmentFeature(
                    start_bar=start_bar,
                    end_bar=end_bar,
                    features=feature_dict,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "segment_start_bar": int(start_bar + 1),
                    "segment_end_bar": int(end_bar),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if not segments:
        raise RuntimeError(
            f"No usable {segment_bars}-bar segments remained for MIDI: {midi_path}"
        )
    return segments, failures


def build_raw_records(
    items: Sequence[dict[str, Any]],
    recurrence_threshold: float,
    progress_name: str,
) -> tuple[list[RawFeatureRecord], list[dict[str, Any]]]:
    records: list[RawFeatureRecord] = []
    failures: list[dict[str, Any]] = []
    total = len(items)

    for index, item in enumerate(items, start=1):
        path = Path(item["path"])
        label = int(item["label"])
        try:
            segment_features, segment_failures = extract_segment_features(
                path,
                recurrence_threshold,
                SEGMENT_BARS,
            )
            for segment in segment_features:
                converted: dict[str, float] = {}
                for key, value in segment.features.items():
                    try:
                        converted[key] = float(value)
                    except (TypeError, ValueError):
                        converted[key] = float("nan")

                records.append(
                    RawFeatureRecord(
                        path=(
                            f"{path}#bars_{segment.start_bar + 1:04d}-"
                            f"{segment.end_bar:04d}"
                        ),
                        label=label,
                        features=converted,
                    )
                )

            for failure in segment_failures:
                failures.append(
                    {
                        "path": str(path),
                        "label": label,
                        **failure,
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "path": str(path),
                    "label": label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        if index % 100 == 0 or index == total:
            print(
                f"{progress_name} features: {index}/{total} MIDI files | "
                f"segments={len(records)}",
                flush=True,
            )

    if not records:
        raise RuntimeError(f"No usable MIDI segments remained in {progress_name}")
    return records, failures

def finite_values(
    records: Sequence[RawFeatureRecord], feature: str, label: int
) -> np.ndarray:
    values = [
        record.features.get(feature, float("nan"))
        for record in records
        if record.label == label
    ]
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def rank_features(
    train_records: Sequence[RawFeatureRecord],
    candidates: Sequence[str],
) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []

    for feature in candidates:
        human = finite_values(train_records, feature, 0)
        ai = finite_values(train_records, feature, 1)
        if len(human) == 0 or len(ai) == 0:
            continue

        delta = float(cliffs_delta(human, ai))
        ranking.append(
            {
                "feature": feature,
                "group": feature_group_of(feature),
                "human_n": int(len(human)),
                "ai_n": int(len(ai)),
                "human_median": float(np.median(human)),
                "ai_median": float(np.median(ai)),
                "cliffs_delta_human_minus_ai": delta,
                "abs_cliffs_delta": abs(delta) if math.isfinite(delta) else float("nan"),
            }
        )

    ranking = [row for row in ranking if math.isfinite(row["abs_cliffs_delta"])]
    ranking.sort(key=lambda row: row["abs_cliffs_delta"], reverse=True)
    if not ranking:
        raise RuntimeError("No candidate feature could be ranked on the training split")
    return ranking


def save_ranking_csv(path: Path, ranking: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature",
        "group",
        "human_n",
        "ai_n",
        "human_median",
        "ai_median",
        "cliffs_delta_human_minus_ai",
        "abs_cliffs_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranking)


def choose_features_manual() -> list[str]:
    return list(MANUAL_FEATURES)


def choose_features_auto_group_topk(
    ranking: Sequence[dict[str, Any]],
) -> list[str]:
    selected: list[str] = []

    for group_name, group_features in FEATURE_GROUPS.items():
        top_k = int(AUTO_TOP_K_BY_GROUP[group_name])
        group_set = set(group_features)
        group_ranking = [row for row in ranking if row["feature"] in group_set]

        if not group_ranking:
            raise RuntimeError(
                f"No rankable training feature remained in group {group_name!r}"
            )

        chosen_rows = (
            group_ranking if top_k == 0 else group_ranking[: min(top_k, len(group_ranking))]
        )
        selected.extend(str(row["feature"]) for row in chosen_rows)

    if not selected:
        raise RuntimeError("Automatic group-wise selection produced no features")
    return selected


def select_features(ranking: Sequence[dict[str, Any]]) -> list[str]:
    validate_feature_config()
    if FEATURE_SELECTION_MODE == "manual":
        return choose_features_manual()
    if FEATURE_SELECTION_MODE == "auto_group_topk":
        return choose_features_auto_group_topk(ranking)
    raise AssertionError("Unreachable feature selection mode")


def print_selected_features(
    selected_features: Sequence[str],
    ranking: Sequence[dict[str, Any]],
) -> None:
    ranking_by_feature = {str(row["feature"]): row for row in ranking}
    print(f"Feature selection mode: {FEATURE_SELECTION_MODE}")
    print("Selected features:")

    for index, feature in enumerate(selected_features, start=1):
        group = feature_group_of(feature)
        row = ranking_by_feature.get(feature)
        if row is None:
            print(f"  {index:02d}. [{group:<10}] {feature:<34} delta=unavailable")
            continue
        print(
            f"  {index:02d}. [{group:<10}] {feature:<34} "
            f"delta={row['cliffs_delta_human_minus_ai']:+.4f} "
            f"|delta|={row['abs_cliffs_delta']:.4f}"
        )



def records_to_matrix(
    records: Sequence[RawFeatureRecord],
    selected_features: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.full((len(records), len(selected_features)), np.nan, dtype=np.float32)
    y = np.asarray([record.label for record in records], dtype=np.float32)

    for row, record in enumerate(records):
        for column, feature in enumerate(selected_features):
            value = record.features.get(feature, float("nan"))
            if math.isfinite(value):
                x[row, column] = float(value)
    return x, y


def fit_preprocessor(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Training-only median imputation.
    medians = np.nanmedian(x_train.astype(np.float64), axis=0).astype(np.float32)
    if not np.isfinite(medians).all():
        bad = np.where(~np.isfinite(medians))[0].tolist()
        raise ValueError(f"Selected features are entirely NaN in training columns: {bad}")

    imputed = np.where(np.isfinite(x_train), x_train, medians[None, :]).astype(np.float32)
    mean = imputed.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = imputed.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return medians, mean, std


def transform_matrix(
    x: np.ndarray,
    medians: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    imputed = np.where(np.isfinite(x), x, medians[None, :]).astype(np.float32)
    return ((imputed - mean[None, :]) / std[None, :]).astype(np.float32)


# -----------------------------------------------------------------------------
# Dataset / model / metrics
# -----------------------------------------------------------------------------


class FeatureDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class MidiFeatureMLP(nn.Module):
    """Two-hidden-layer MLP over whole-song symbolic MIDI features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        bottleneck_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be at least 1")
        if hidden_dim < 1 or bottleneck_dim < 1:
            raise ValueError("hidden dimensions must be at least 1")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class Metrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    human_recall: float
    ai_recall: float


def compute_metrics(total_loss: float, labels: list[int], predictions: list[int]) -> Metrics:
    if not labels:
        raise ValueError("Cannot calculate metrics for an empty dataset")

    recalls = []
    for label in (0, 1):
        indices = [i for i, value in enumerate(labels) if value == label]
        recall = (
            sum(predictions[i] == label for i in indices) / len(indices)
            if indices
            else float("nan")
        )
        recalls.append(recall)

    available = [value for value in recalls if math.isfinite(value)]
    return Metrics(
        loss=total_loss / len(labels),
        accuracy=sum(a == b for a, b in zip(labels, predictions)) / len(labels),
        balanced_accuracy=sum(available) / len(available),
        human_recall=recalls[0],
        ai_recall=recalls[1],
    )


def run_epoch(
    model: MidiFeatureMLP,
    loader: DataLoader,
    device: torch.device,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    amp: bool,
    gradient_clip: float,
) -> Metrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_out: list[int] = []
    predictions_out: list[int] = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(features)
                loss = loss_function(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()

        predictions = (torch.sigmoid(logits) >= 0.5).long()
        total_loss += float(loss.detach()) * labels.numel()
        labels_out.extend(labels.long().detach().cpu().tolist())
        predictions_out.extend(predictions.detach().cpu().tolist())

    return compute_metrics(total_loss, labels_out, predictions_out)


def format_metrics(metrics: Metrics) -> str:
    return (
        f"loss {metrics.loss:.4f}, acc {metrics.accuracy:.4f}, "
        f"bal-acc {metrics.balanced_accuracy:.4f}, "
        f"human-recall {metrics.human_recall:.4f}, ai-recall {metrics.ai_recall:.4f}"
    )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def command_train(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1")
    if args.gradient_clip <= 0:
        raise ValueError("--gradient-clip must be greater than 0")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be at least 0 and less than 1")

    validate_feature_config()
    validate_segment_config()
    set_seed(args.seed)
    device = resolve_device(args.device)
    manifest = load_manifest(args.split_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_records, train_failures = build_raw_records(
        manifest["train"], args.recurrence_threshold, "Train"
    )
    validation_records, validation_failures = build_raw_records(
        manifest["validation"], args.recurrence_threshold, "Validation"
    )

    failures = train_failures + validation_failures
    if failures:
        save_json(args.output_dir / "midi_feature_mlp_preprocess_failures.json", failures)
        print(f"Skipped malformed/unusable MIDI files: {len(failures)}")

    train_counts = Counter(record.label for record in train_records)
    validation_counts = Counter(record.label for record in validation_records)
    if not all(train_counts.get(label, 0) for label in (0, 1)):
        raise ValueError("Both Human(label 0) and AI(label 1) are required in training")
    if not all(validation_counts.get(label, 0) for label in (0, 1)):
        raise ValueError("Both classes are required in validation")

    sample_name = "songs" if SEGMENT_BARS == 0 else "segments"
    print(
        f"Usable {sample_name} | "
        f"train Human={train_counts[0]}, AI={train_counts[1]} | "
        f"validation Human={validation_counts[0]}, AI={validation_counts[1]}"
    )
    print(f"Segment bars: {SEGMENT_BARS}")

    # Always rank all four musical groups on the TRAIN split for inspection.
    # Manual mode does NOT use this ranking to choose features.
    ranking = rank_features(train_records, ALL_MUSICAL_FEATURES)
    save_json(args.output_dir / "feature_ranking_train_only.json", ranking)
    save_ranking_csv(args.output_dir / "feature_ranking_train_only.csv", ranking)

    selected_features = select_features(ranking)
    print_selected_features(selected_features, ranking)

    x_train_raw, y_train = records_to_matrix(train_records, selected_features)
    x_validation_raw, y_validation = records_to_matrix(
        validation_records, selected_features
    )

    medians, mean, std = fit_preprocessor(x_train_raw)
    x_train = transform_matrix(x_train_raw, medians, mean, std)
    x_validation = transform_matrix(x_validation_raw, medians, mean, std)

    train_dataset = FeatureDataset(x_train, y_train)
    validation_dataset = FeatureDataset(x_validation, y_validation)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = MidiFeatureMLP(
        input_dim=len(selected_features),
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable MLP parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    pos_weight = torch.tensor(
        [train_counts[0] / train_counts[1]], dtype=torch.float32, device=device
    )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    checkpoint_path = args.output_dir / "midi_feature_mlp_best.pt"
    best_loss = float("inf")
    best_epoch = 0
    best_validation_metrics: EpochMetrics | None = None
    stale_epochs = 0

    tensorboard_dir = args.output_dir / "tensorboard"
    writer = make_tensorboard_writer(tensorboard_dir)
    training_history: list[dict[str, float | int]] = []

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                loss_function,
                optimizer,
                scaler,
                args.amp,
                args.gradient_clip,
            )
            validation_metrics = run_epoch(
                model,
                validation_loader,
                device,
                loss_function,
                None,
                scaler,
                args.amp,
                args.gradient_clip,
            )

            print(
                f"MLP epoch {epoch:03d} | train {format_metrics(train_metrics)} | "
                f"val {format_metrics(validation_metrics)}",
                flush=True,
            )

            writer.add_scalar("Loss/train", train_metrics.loss, epoch)
            writer.add_scalar("Loss/validation", validation_metrics.loss, epoch)
            writer.add_scalar("Accuracy/train", train_metrics.accuracy, epoch)
            writer.add_scalar("Accuracy/validation", validation_metrics.accuracy, epoch)
            writer.add_scalar(
                "BalancedAccuracy/train", train_metrics.balanced_accuracy, epoch
            )
            writer.add_scalar(
                "BalancedAccuracy/validation",
                validation_metrics.balanced_accuracy,
                epoch,
            )

            training_history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_metrics.loss),
                    "validation_loss": float(validation_metrics.loss),
                    "train_accuracy": float(train_metrics.accuracy),
                    "validation_accuracy": float(validation_metrics.accuracy),
                    "train_balanced_accuracy": float(train_metrics.balanced_accuracy),
                    "validation_balanced_accuracy": float(
                        validation_metrics.balanced_accuracy
                    ),
                    "train_human_recall": float(train_metrics.human_recall),
                    "validation_human_recall": float(validation_metrics.human_recall),
                    "train_ai_recall": float(train_metrics.ai_recall),
                    "validation_ai_recall": float(validation_metrics.ai_recall),
                }
            )

            if validation_metrics.loss < best_loss:
                best_loss = validation_metrics.loss
                best_epoch = epoch
                best_validation_metrics = validation_metrics
                stale_epochs = 0
                torch.save(
                    {
                        "format": "midi-global-feature-mlp-v1",
                        "build": MIDI_FEATURE_MLP_BUILD,
                        "model_state": model.state_dict(),
                        "candidate_features": list(ALL_MUSICAL_FEATURES),
                        "selected_features": list(selected_features),
                        "feature_ranking": ranking,
                        "feature_selection_mode": FEATURE_SELECTION_MODE,
                        "manual_features": list(MANUAL_FEATURES),
                        "auto_top_k_by_group": dict(AUTO_TOP_K_BY_GROUP),
                        "feature_medians": medians.tolist(),
                        "feature_mean": mean.tolist(),
                        "feature_std": std.tolist(),
                        "recurrence_threshold": float(args.recurrence_threshold),
                        "segment_bars": int(SEGMENT_BARS),
                        "hidden_dim": int(args.hidden_dim),
                        "bottleneck_dim": int(args.bottleneck_dim),
                        "dropout": float(args.dropout),
                        "threshold": 0.5,
                        "best_validation_loss": best_loss,
                        "best_validation_metrics": validation_metrics.__dict__,
                        "split_manifest": str(args.split_manifest.resolve()),
                        "training_data": {
                            "real_dirs": manifest.get("real_dirs"),
                            "fake_dirs": manifest.get("fake_dirs"),
                            "train_human": int(train_counts[0]),
                            "train_ai": int(train_counts[1]),
                            "validation_human": int(validation_counts[0]),
                            "validation_ai": int(validation_counts[1]),
                        },
                        "seed": int(args.seed),
                        "learning_rate": float(args.learning_rate),
                        "weight_decay": float(args.weight_decay),
                        "batch_size": int(args.batch_size),
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"MLP early stopping after {epoch} epochs")
                    break

    finally:
        writer.flush()
        writer.close()

    save_json(
        args.output_dir / "midi_feature_mlp_training_history.json",
        training_history,
    )

    print(f"TensorBoard logs: {tensorboard_dir.resolve()}")
    print("MLP training finished.")
    print(f"Best checkpoint: {checkpoint_path.resolve()}")
    print(f"Feature ranking: {(args.output_dir / 'feature_ranking_train_only.csv').resolve()}")

    if best_validation_metrics is None:
        raise RuntimeError("Training finished without producing a best validation result")

    append_result_log(
        {
            "event": "train",
            "build": MIDI_FEATURE_MLP_BUILD,
            "split_manifest": str(args.split_manifest.resolve()),
            "training_data": {
                "real_dirs": manifest.get("real_dirs"),
                "fake_dirs": manifest.get("fake_dirs"),
                "train_human": int(train_counts[0]),
                "train_ai": int(train_counts[1]),
                "validation_human": int(validation_counts[0]),
                "validation_ai": int(validation_counts[1]),
            },
            "feature_selection_mode": FEATURE_SELECTION_MODE,
            "segment_bars": int(SEGMENT_BARS),
            "selected_features": list(selected_features),
            "best_epoch": int(best_epoch),
            "best_validation": {
                "loss": float(best_validation_metrics.loss),
                "accuracy": float(best_validation_metrics.accuracy),
                "balanced_accuracy": float(best_validation_metrics.balanced_accuracy),
                "human_recall": float(best_validation_metrics.human_recall),
                "ai_recall": float(best_validation_metrics.ai_recall),
            },
            "checkpoint": str(checkpoint_path.resolve()),
        }
    )


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


def load_model_for_inference(
    checkpoint: Path,
    device: torch.device,
) -> tuple[MidiFeatureMLP, dict[str, Any]]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MLP checkpoint not found: {checkpoint}")
    payload = torch_load(checkpoint)
    if payload.get("format") != "midi-global-feature-mlp-v1":
        raise ValueError(f"Unsupported MLP checkpoint: {checkpoint}")

    selected_features = list(payload["selected_features"])
    model = MidiFeatureMLP(
        input_dim=len(selected_features),
        hidden_dim=int(payload["hidden_dim"]),
        bottleneck_dim=int(payload["bottleneck_dim"]),
        dropout=float(payload["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def vectors_for_one_midi(
    midi_path: Path,
    payload: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float], int]:
    recurrence_threshold = float(payload["recurrence_threshold"])
    segment_bars = int(payload.get("segment_bars", 0))
    segment_features, _segment_failures = extract_segment_features(
        midi_path,
        recurrence_threshold,
        segment_bars,
    )

    selected_features = list(payload["selected_features"])
    raw = np.full(
        (len(segment_features), len(selected_features)),
        np.nan,
        dtype=np.float32,
    )

    for row, segment in enumerate(segment_features):
        for column, feature in enumerate(selected_features):
            if feature not in segment.features:
                raise KeyError(
                    f"Feature '{feature}' is missing from segmented feature extraction"
                )
            value = float(segment.features[feature])
            if math.isfinite(value):
                raw[row, column] = value

    medians = np.asarray(payload["feature_medians"], dtype=np.float32)
    mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    std = np.asarray(payload["feature_std"], dtype=np.float32)
    transformed = transform_matrix(raw, medians, mean, std)

    raw_feature_mean: dict[str, float] = {}
    for column, feature in enumerate(selected_features):
        values = raw[:, column]
        finite = values[np.isfinite(values)]
        raw_feature_mean[feature] = (
            float(np.mean(finite)) if len(finite) else float("nan")
        )

    return transformed, raw_feature_mean, len(segment_features)


def predict_one_midi(
    midi_path: Path,
    model: MidiFeatureMLP,
    payload: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    features, raw_feature_dict, segment_count = vectors_for_one_midi(
        midi_path, payload
    )
    tensor = torch.from_numpy(features).to(device)

    with torch.no_grad():
        logits = model(tensor)
        segment_ai_probabilities = (
            torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        )

    # Folder evaluation remains song-level: average segment probabilities first.
    ai_probability = float(np.mean(segment_ai_probabilities))
    threshold = float(payload.get("threshold", 0.5))
    prediction_id = 1 if ai_probability >= threshold else 0
    return {
        "path": midi_path,
        "prediction_id": prediction_id,
        "prediction": "AI" if prediction_id == 1 else "Human",
        "ai_probability": ai_probability,
        "human_probability": 1.0 - ai_probability,
        "segment_count": int(segment_count),
        "selected_features": list(payload["selected_features"]),
        "raw_features": raw_feature_dict,
    }

def _seen_midi_paths_from_checkpoint(payload: dict[str, Any]) -> set[str]:
    """Return train+validation MIDI paths recorded by the checkpoint split."""
    split_manifest = payload.get("split_manifest")
    if not split_manifest:
        return set()

    manifest_path = Path(str(split_manifest))
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "The checkpoint's split manifest is missing, so seen MIDI files "
            f"cannot be excluded safely: {manifest_path}"
        )

    manifest = load_manifest(manifest_path)
    seen: set[str] = set()
    for split_name in ("train", "validation"):
        for record in manifest[split_name]:
            seen.add(str(Path(record["path"]).resolve()))
    return seen


def _select_inference_paths(
    midi_dir: Path,
    payload: dict[str, Any],
    count: int,
    seed: int,
    include_seen: bool,
) -> tuple[list[Path], int, int]:
    """Discover, optionally de-overlap, then randomly select folder MIDI files."""
    if count < 0:
        raise ValueError("--count must be 0 or greater")

    all_paths = discover_midis(midi_dir)
    before_exclusion = len(all_paths)

    excluded_seen = 0
    if not include_seen:
        seen = _seen_midi_paths_from_checkpoint(payload)
        if seen:
            remaining = [
                path for path in all_paths
                if str(path.resolve()) not in seen
            ]
            excluded_seen = before_exclusion - len(remaining)
            all_paths = remaining

    if not all_paths:
        raise ValueError(
            "No MIDI files remain for inference after excluding the training/"
            "validation split."
        )

    if count == 0:
        selected = all_paths
    else:
        if count > len(all_paths):
            raise ValueError(
                f"--count requested {count} MIDI files, but only {len(all_paths)} "
                "are available after excluding seen files."
            )
        rng = random.Random(seed)
        selected = rng.sample(all_paths, count)
        selected.sort()

    return selected, before_exclusion, excluded_seen


def command_infer(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, payload = load_model_for_inference(args.checkpoint, device)

    if args.midi is not None:
        result = predict_one_midi(args.midi, model, payload, device)
        print(f"MIDI: {result['path'].resolve()}")
        print(f"Prediction: {result['prediction']}")
        print(f"AI probability: {result['ai_probability']:.6f}")
        print(f"Human probability: {result['human_probability']:.6f}")
        print(f"Segments: {result['segment_count']}")
        print("Selected features (segment mean):")
        for feature in result["selected_features"]:
            value = result["raw_features"].get(feature, float("nan"))
            print(f"  {feature}: {value:.6f}")

        append_result_log(
            {
                "event": "infer_single",
                "build": MIDI_FEATURE_MLP_BUILD,
                "training_data": {
                    "split_manifest": payload.get("split_manifest"),
                    **dict(payload.get("training_data") or {}),
                },
                "checkpoint": str(args.checkpoint.resolve()),
                "inference_midi": str(args.midi.resolve()),
                "prediction": result["prediction"],
                "ai_probability": float(result["ai_probability"]),
                "human_probability": float(result["human_probability"]),
            }
        )
        return

    if args.label is None:
        raise ValueError("--label is required when using --midi-dir")

    label_id = 1 if args.label == "ai" else 0
    label_name = "AI" if label_id == 1 else "Human"
    paths, discovered_count, excluded_seen = _select_inference_paths(
        args.midi_dir,
        payload,
        args.count,
        args.seed,
        args.include_seen,
    )

    print(f"Discovered in folder: {discovered_count}")
    print(f"Excluded train/validation MIDI files: {excluded_seen}")
    print(f"Selected for inference: {len(paths)}")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    correct = 0

    for index, path in enumerate(paths, start=1):
        try:
            result = predict_one_midi(path, model, payload, device)
            is_correct = int(result["prediction_id"] == label_id)
            correct += is_correct
            rows.append(
                {
                    "path": str(path),
                    "label": label_name,
                    "prediction": result["prediction"],
                    "ai_probability": result["ai_probability"],
                    "human_probability": result["human_probability"],
                    "correct": is_correct,
                }
            )

            status = "CORRECT" if is_correct else "WRONG"
            print(
                f"[{index}/{len(paths)}] {path.name} | "
                f"Prediction: {result['prediction']} | "
                f"AI: {result['ai_probability']:.6f} | "
                f"Human: {result['human_probability']:.6f} | "
                f"Segments: {result['segment_count']} | "
                f"{status}",
                flush=True,
            )

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            failures.append({"path": str(path), "error": error_text})
            print(
                f"[{index}/{len(paths)}] {path.name} | ERROR: {error_text}",
                flush=True,
            )

    evaluated = len(rows)
    if evaluated == 0:
        raise RuntimeError("No MIDI files could be evaluated")

    accuracy = correct / evaluated
    mean_ai_probability = float(
        np.mean([float(row["ai_probability"]) for row in rows])
    )
    mean_human_probability = float(
        np.mean([float(row["human_probability"]) for row in rows])
    )

    print(f"Label: {label_name}")
    print(f"Evaluated: {evaluated}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Mean AI probability: {mean_ai_probability:.6f}")
    print(f"Mean Human probability: {mean_human_probability:.6f}")
    print(f"Failed: {len(failures)}")

    append_result_log(
        {
            "event": "infer_folder",
            "build": MIDI_FEATURE_MLP_BUILD,
            "training_data": {
                "split_manifest": payload.get("split_manifest"),
                **dict(payload.get("training_data") or {}),
            },
            "checkpoint": str(args.checkpoint.resolve()),
            "inference_dir": str(args.midi_dir.resolve()),
            "label": label_name,
            "discovered_in_folder": int(discovered_count),
            "excluded_seen": int(excluded_seen),
            "requested_count": int(args.count),
            "selected_count": int(len(paths)),
            "include_seen": bool(args.include_seen),
            "inference_seed": int(args.seed),
            "evaluated": int(evaluated),
            "correct": int(correct),
            "accuracy": float(accuracy),
            "mean_ai_probability": mean_ai_probability,
            "mean_human_probability": mean_human_probability,
            "failed": int(len(failures)),
        }
    )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "path",
                    "label",
                    "prediction",
                    "ai_probability",
                    "human_probability",
                    "correct",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions: {args.output_csv.resolve()}")

    if failures and args.failure_json is not None:
        save_json(args.failure_json, failures)
        print(f"Saved failures: {args.failure_json.resolve()}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Whole-song MIDI feature MLP for Human-vs-AI classification"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("make-split")
    split.add_argument(
        "--real-dir",
        nargs=2,
        action="append",
        metavar=("DIR", "COUNT"),
        required=True,
        help="Repeat for each Human source: --real-dir <directory> <count>",
    )
    split.add_argument(
        "--fake-dir",
        nargs=2,
        action="append",
        metavar=("DIR", "COUNT"),
        required=True,
        help="Repeat for each AI source: --fake-dir <directory> <count>",
    )
    split.add_argument("--val-ratio", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--output", type=Path, required=True)
    split.set_defaults(func=command_make_split)

    train = subparsers.add_parser("train")
    train.add_argument("--split-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--recurrence-threshold", type=float, default=0.85)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--bottleneck-dim", type=int, default=32)
    train.add_argument("--dropout", type=float, default=0.2)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--patience", type=int, default=12)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")
    train.add_argument("--amp", action="store_true")
    train.set_defaults(func=command_train)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--checkpoint", type=Path, required=True)
    source = infer.add_mutually_exclusive_group(required=True)
    source.add_argument("--midi", type=Path)
    source.add_argument("--midi-dir", type=Path)
    infer.add_argument("--label", choices=("human", "ai"))
    infer.add_argument(
        "--count",
        type=int,
        default=0,
        help=(
            "Number of MIDI files to randomly evaluate from --midi-dir after "
            "excluding train/validation files; 0 = use all remaining files"
        ),
    )
    infer.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --count is greater than 0",
    )
    infer.add_argument(
        "--include-seen",
        action="store_true",
        help=(
            "Allow MIDI files already present in the checkpoint's train/validation "
            "split. By default they are excluded from folder inference."
        ),
    )
    infer.add_argument("--output-csv", type=Path)
    infer.add_argument("--failure-json", type=Path)
    infer.add_argument("--device", default="auto")
    infer.set_defaults(func=command_infer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.output_dir / "midi_feature_mlp_train.log"
        log_stream = log_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = TeeStream(sys.stdout, log_stream)
        sys.stderr = TeeStream(sys.stderr, log_stream)
        print(f"Training log: {log_path.resolve()}", flush=True)

    print(f"MIDI Feature MLP build: {MIDI_FEATURE_MLP_BUILD}", flush=True)
    args.func(args)


if __name__ == "__main__":
    main()
