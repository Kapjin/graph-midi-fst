#!/usr/bin/env python3
"""Train and run Extra Trees on whole-song symbolic MIDI features.

This script uses the same whole-song feature extraction interface as
train_midi_feature_mlp_v5(1).py:

    compare_real_fake_midi.extract_features(midi_path, recurrence_threshold=...)

Each MIDI file becomes one fixed-length feature vector, then ExtraTreesClassifier
classifies the whole song as Human (0) or AI (1).

The existing split manifest made for the MLP can be reused directly.
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
from typing import Any, Sequence

try:
    import joblib
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        log_loss,
        recall_score,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install numpy, scikit-learn, and joblib."
    ) from exc

try:
    from compare_real_fake_midi import cliffs_delta, extract_features
except ImportError as exc:
    raise SystemExit(
        "Could not import compare_real_fake_midi.py. Put this script in the same "
        "directory as compare_real_fake_midi.py."
    ) from exc


MIDI_FEATURE_EXTRATREES_BUILD = "2026-08-26-global-midi-feature-extratrees-v3-logging"
MIDI_EXTS = {".mid", ".midi"}

# Final inference summaries are appended here automatically.
# The file is created in the SAME directory as this Python script.
RESULT_LOG_PATH = Path(__file__).resolve().parent / "midi_feature_extratrees_results_26082618.txt"

# =============================================================================
# FEATURE SELECTION CONFIG
# =============================================================================
# Kept the same as the uploaded MLP script so the classifier comparison uses
# the same input features.
FEATURE_SELECTION_MODE = "manual"

MANUAL_FEATURES = [
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

AUTO_TOP_K_BY_GROUP = {
    "structural": 3,
    "rhythm": 1,
    "melody": 2,
    "harmony": 2,
}

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
# Utilities
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
            "TensorBoard logging requires torch and tensorboard. "
            "Install tensorboard with: python3 -m pip install tensorboard"
        ) from exc

    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def append_result_log(record: dict[str, Any]) -> None:
    """Save only the minimal final inference result next to this script."""
    event = record.get("event")

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
# Feature extraction / selection
# -----------------------------------------------------------------------------


@dataclass
class RawFeatureRecord:
    path: str
    label: int
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

    if FEATURE_SELECTION_MODE == "auto_group_topk":
        missing_groups = [
            group for group in FEATURE_GROUPS if group not in AUTO_TOP_K_BY_GROUP
        ]
        if missing_groups:
            raise ValueError(
                "AUTO_TOP_K_BY_GROUP is missing group(s): " + ", ".join(missing_groups)
            )


def extract_record(
    midi_path: Path,
    label: int,
    recurrence_threshold: float,
) -> RawFeatureRecord:
    feature_dict, _ssm = extract_features(
        midi_path,
        recurrence_threshold=recurrence_threshold,
    )

    converted: dict[str, float] = {}
    for key, value in feature_dict.items():
        try:
            converted[key] = float(value)
        except (TypeError, ValueError):
            converted[key] = float("nan")

    return RawFeatureRecord(str(midi_path), int(label), converted)


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
        try:
            records.append(
                extract_record(path, int(item["label"]), recurrence_threshold)
            )
        except Exception as exc:
            failures.append(
                {
                    "path": str(path),
                    "label": int(item["label"]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        if index % 100 == 0 or index == total:
            print(f"{progress_name} features: {index}/{total} MIDI files", flush=True)

    if not records:
        raise RuntimeError(f"No usable MIDI files remained in {progress_name}")

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
        if not math.isfinite(delta):
            continue

        ranking.append(
            {
                "feature": feature,
                "group": feature_group_of(feature),
                "human_n": int(len(human)),
                "ai_n": int(len(ai)),
                "human_median": float(np.median(human)),
                "ai_median": float(np.median(ai)),
                "cliffs_delta_human_minus_ai": delta,
                "abs_cliffs_delta": abs(delta),
            }
        )

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


def choose_features(ranking: Sequence[dict[str, Any]]) -> list[str]:
    validate_feature_config()

    if FEATURE_SELECTION_MODE == "manual":
        return list(MANUAL_FEATURES)

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
            group_ranking
            if top_k == 0
            else group_ranking[: min(top_k, len(group_ranking))]
        )
        selected.extend(str(row["feature"]) for row in chosen_rows)

    return selected


def print_selected_features(
    selected_features: Sequence[str],
    ranking: Sequence[dict[str, Any]],
) -> None:
    ranking_by_feature = {str(row["feature"]): row for row in ranking}
    print(f"Feature selection mode: {FEATURE_SELECTION_MODE}")
    print("Selected features:")
    for index, feature in enumerate(selected_features, start=1):
        row = ranking_by_feature.get(feature)
        group = feature_group_of(feature)
        if row is None:
            print(f"  {index:02d}. [{group:<10}] {feature:<34} delta=unavailable")
        else:
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
    y = np.asarray([record.label for record in records], dtype=np.int64)

    for row, record in enumerate(records):
        for column, feature in enumerate(selected_features):
            value = record.features.get(feature, float("nan"))
            if math.isfinite(value):
                x[row, column] = float(value)

    return x, y


def fit_medians(x_train: np.ndarray) -> np.ndarray:
    medians = np.nanmedian(x_train.astype(np.float64), axis=0).astype(np.float32)
    if not np.isfinite(medians).all():
        bad = np.where(~np.isfinite(medians))[0].tolist()
        raise ValueError(f"Selected features are entirely NaN in training columns: {bad}")
    return medians


def impute_matrix(x: np.ndarray, medians: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, medians[None, :]).astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "human_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "ai_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def command_train(args: argparse.Namespace) -> None:
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
        save_json(args.output_dir / "midi_feature_extratrees_preprocess_failures.json", failures)
        print(f"Skipped malformed/unusable MIDI files: {len(failures)}")

    train_counts = Counter(record.label for record in train_records)
    validation_counts = Counter(record.label for record in validation_records)

    if not all(train_counts.get(label, 0) for label in (0, 1)):
        raise ValueError("Both Human(label 0) and AI(label 1) are required in training")
    if not all(validation_counts.get(label, 0) for label in (0, 1)):
        raise ValueError("Both classes are required in validation")

    print(
        "Usable songs | "
        f"train Human={train_counts[0]}, AI={train_counts[1]} | "
        f"validation Human={validation_counts[0]}, AI={validation_counts[1]}"
    )

    ranking = rank_features(train_records, ALL_MUSICAL_FEATURES)
    save_json(args.output_dir / "feature_ranking_train_only.json", ranking)
    save_ranking_csv(args.output_dir / "feature_ranking_train_only.csv", ranking)

    selected_features = choose_features(ranking)
    print_selected_features(selected_features, ranking)

    x_train_raw, y_train = records_to_matrix(train_records, selected_features)
    x_validation_raw, y_validation = records_to_matrix(
        validation_records, selected_features
    )

    # Trees do not require feature standardization. We only fill missing values
    # using medians computed from the training split.
    medians = fit_medians(x_train_raw)
    x_train = impute_matrix(x_train_raw, medians)
    x_validation = impute_matrix(x_validation_raw, medians)

    if args.n_estimators < 1:
        raise ValueError("--n-estimators must be at least 1")
    if args.log_every_trees < 1:
        raise ValueError("--log-every-trees must be at least 1")

    tensorboard_dir = args.output_dir / "tensorboard"
    writer = make_tensorboard_writer(tensorboard_dir)

    # Extra Trees has no epoch-based training loop. To obtain a train/validation
    # curve, grow the ensemble incrementally and evaluate it after every
    # --log-every-trees trees. The final model still contains --n-estimators trees.
    model = ExtraTreesClassifier(
        n_estimators=1,
        criterion="gini",
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        bootstrap=False,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        warm_start=True,
    )

    checkpoints = list(range(args.log_every_trees, args.n_estimators + 1, args.log_every_trees))
    if not checkpoints or checkpoints[-1] != args.n_estimators:
        checkpoints.append(args.n_estimators)

    training_history: list[dict[str, float | int]] = []

    try:
        for tree_count in checkpoints:
            model.set_params(n_estimators=tree_count)
            model.fit(x_train, y_train)

            train_probability = model.predict_proba(x_train)
            validation_probability = model.predict_proba(x_validation)
            train_pred_step = model.predict(x_train)
            validation_pred_step = model.predict(x_validation)

            train_loss = float(log_loss(y_train, train_probability, labels=[0, 1]))
            validation_loss = float(
                log_loss(y_validation, validation_probability, labels=[0, 1])
            )
            train_acc = float(accuracy_score(y_train, train_pred_step))
            validation_acc = float(accuracy_score(y_validation, validation_pred_step))
            train_bal_acc = float(
                balanced_accuracy_score(y_train, train_pred_step)
            )
            validation_bal_acc = float(
                balanced_accuracy_score(y_validation, validation_pred_step)
            )

            writer.add_scalar("Loss/train", train_loss, tree_count)
            writer.add_scalar("Loss/validation", validation_loss, tree_count)
            writer.add_scalar("Accuracy/train", train_acc, tree_count)
            writer.add_scalar("Accuracy/validation", validation_acc, tree_count)
            writer.add_scalar("BalancedAccuracy/train", train_bal_acc, tree_count)
            writer.add_scalar(
                "BalancedAccuracy/validation", validation_bal_acc, tree_count
            )

            training_history.append(
                {
                    "trees": int(tree_count),
                    "train_log_loss": train_loss,
                    "validation_log_loss": validation_loss,
                    "train_accuracy": train_acc,
                    "validation_accuracy": validation_acc,
                    "train_balanced_accuracy": train_bal_acc,
                    "validation_balanced_accuracy": validation_bal_acc,
                }
            )

            print(
                f"Trees {tree_count:04d} | "
                f"train loss {train_loss:.6f}, acc {train_acc:.4f} | "
                f"val loss {validation_loss:.6f}, acc {validation_acc:.4f}",
                flush=True,
            )
    finally:
        writer.flush()
        writer.close()

    train_pred = model.predict(x_train)
    validation_pred = model.predict(x_validation)
    train_metrics = compute_metrics(y_train, train_pred)
    validation_metrics = compute_metrics(y_validation, validation_pred)

    print(
        "Extra Trees train | "
        f"acc {train_metrics['accuracy']:.4f}, "
        f"bal-acc {train_metrics['balanced_accuracy']:.4f}, "
        f"human-recall {train_metrics['human_recall']:.4f}, "
        f"ai-recall {train_metrics['ai_recall']:.4f}"
    )
    print(
        "Extra Trees validation | "
        f"acc {validation_metrics['accuracy']:.4f}, "
        f"bal-acc {validation_metrics['balanced_accuracy']:.4f}, "
        f"human-recall {validation_metrics['human_recall']:.4f}, "
        f"ai-recall {validation_metrics['ai_recall']:.4f}"
    )

    importances = sorted(
        zip(selected_features, model.feature_importances_.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    print("Feature importances:")
    for feature, importance in importances:
        print(f"  {feature}: {importance:.6f}")

    model_path = args.output_dir / "midi_feature_extratrees.joblib"
    payload = {
        "format": "midi-global-feature-extratrees-v1",
        "build": MIDI_FEATURE_EXTRATREES_BUILD,
        "model": model,
        "candidate_features": list(ALL_MUSICAL_FEATURES),
        "selected_features": list(selected_features),
        "feature_ranking": ranking,
        "feature_selection_mode": FEATURE_SELECTION_MODE,
        "manual_features": list(MANUAL_FEATURES),
        "auto_top_k_by_group": dict(AUTO_TOP_K_BY_GROUP),
        "feature_medians": medians.tolist(),
        "recurrence_threshold": float(args.recurrence_threshold),
        "split_manifest": str(args.split_manifest.resolve()),
        "training_data": {
            "real_dirs": manifest.get("real_dirs"),
            "fake_dirs": manifest.get("fake_dirs"),
            "train_human": int(train_counts[0]),
            "train_ai": int(train_counts[1]),
            "validation_human": int(validation_counts[0]),
            "validation_ai": int(validation_counts[1]),
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "feature_importances": [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in importances
        ],
        "parameters": {
            "n_estimators": int(args.n_estimators),
            "max_depth": args.max_depth,
            "min_samples_split": int(args.min_samples_split),
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_features": args.max_features,
            "seed": int(args.seed),
        },
    }
    joblib.dump(payload, model_path)

    save_json(
        args.output_dir / "midi_feature_extratrees_summary.json",
        {
            key: value
            for key, value in payload.items()
            if key not in {"model", "feature_ranking"}
        },
    )
    save_json(
        args.output_dir / "midi_feature_extratrees_training_history.json",
        training_history,
    )

    print(f"TensorBoard logs: {tensorboard_dir.resolve()}")
    print("Extra Trees training finished.")
    print(f"Model: {model_path.resolve()}")


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


def load_model_payload(model_path: Path) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Extra Trees model not found: {model_path}")
    payload = joblib.load(model_path)
    if payload.get("format") != "midi-global-feature-extratrees-v1":
        raise ValueError(f"Unsupported Extra Trees model: {model_path}")
    return payload


def vector_for_one_midi(
    midi_path: Path,
    payload: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    feature_dict, _ssm = extract_features(
        midi_path,
        recurrence_threshold=float(payload["recurrence_threshold"]),
    )

    selected_features = list(payload["selected_features"])
    raw = np.full((1, len(selected_features)), np.nan, dtype=np.float32)

    for column, feature in enumerate(selected_features):
        if feature not in feature_dict:
            raise KeyError(
                f"Feature '{feature}' is missing from compare_real_fake_midi.extract_features()"
            )
        value = float(feature_dict[feature])
        if math.isfinite(value):
            raw[0, column] = value

    medians = np.asarray(payload["feature_medians"], dtype=np.float32)
    x = impute_matrix(raw, medians)
    return x, {key: float(value) for key, value in feature_dict.items()}


def predict_one_midi(
    midi_path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    x, raw_feature_dict = vector_for_one_midi(midi_path, payload)
    model: ExtraTreesClassifier = payload["model"]

    probabilities = model.predict_proba(x)[0]
    class_to_probability = {
        int(class_id): float(probability)
        for class_id, probability in zip(model.classes_, probabilities)
    }
    human_probability = class_to_probability.get(0, 0.0)
    ai_probability = class_to_probability.get(1, 0.0)
    prediction_id = int(model.predict(x)[0])

    return {
        "path": midi_path,
        "prediction_id": prediction_id,
        "prediction": "AI" if prediction_id == 1 else "Human",
        "ai_probability": ai_probability,
        "human_probability": human_probability,
        "selected_features": list(payload["selected_features"]),
        "raw_features": raw_feature_dict,
    }


def seen_midi_paths_from_payload(payload: dict[str, Any]) -> set[str]:
    split_manifest = payload.get("split_manifest")
    if not split_manifest:
        return set()

    manifest_path = Path(str(split_manifest))
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "The model's split manifest is missing, so seen MIDI files cannot be "
            f"excluded safely: {manifest_path}"
        )

    manifest = load_manifest(manifest_path)
    seen: set[str] = set()
    for split_name in ("train", "validation"):
        for record in manifest[split_name]:
            seen.add(str(Path(record["path"]).resolve()))
    return seen


def select_inference_paths(
    midi_dir: Path,
    payload: dict[str, Any],
    count: int,
    seed: int,
    include_seen: bool,
) -> tuple[list[Path], int, int]:
    if count < 0:
        raise ValueError("--count must be 0 or greater")

    all_paths = discover_midis(midi_dir)
    before_exclusion = len(all_paths)
    excluded_seen = 0

    if not include_seen:
        seen = seen_midi_paths_from_payload(payload)
        remaining = [
            path for path in all_paths if str(path.resolve()) not in seen
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
    payload = load_model_payload(args.model)

    if args.midi is not None:
        result = predict_one_midi(args.midi, payload)
        print(f"MIDI: {result['path'].resolve()}")
        print(f"Prediction: {result['prediction']}")
        print(f"AI probability: {result['ai_probability']:.6f}")
        print(f"Human probability: {result['human_probability']:.6f}")
        print("Selected features:")
        for feature in result["selected_features"]:
            value = result["raw_features"].get(feature, float("nan"))
            print(f"  {feature}: {value:.6f}")

        append_result_log(
            {
                "event": "infer_single",
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

    paths, discovered_count, excluded_seen = select_inference_paths(
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
            result = predict_one_midi(path, payload)
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
    print(f"Label: {label_name}")
    print(f"Evaluated: {evaluated}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Failed: {len(failures)}")

    append_result_log(
        {
            "event": "infer_folder",
            "label": label_name,
            "evaluated": int(evaluated),
            "correct": int(correct),
            "accuracy": float(accuracy),
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
        description="Whole-song MIDI feature Extra Trees for Human-vs-AI classification"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--split-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--recurrence-threshold", type=float, default=0.85)
    train.add_argument("--n-estimators", type=int, default=500)
    train.add_argument("--log-every-trees", type=int, default=10)
    train.add_argument("--max-depth", type=int, default=None)
    train.add_argument("--min-samples-split", type=int, default=2)
    train.add_argument("--min-samples-leaf", type=int, default=1)
    train.add_argument("--max-features", default="sqrt")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--n-jobs", type=int, default=-1)
    train.set_defaults(func=command_train)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--model", type=Path, required=True)
    source = infer.add_mutually_exclusive_group(required=True)
    source.add_argument("--midi", type=Path)
    source.add_argument("--midi-dir", type=Path)
    infer.add_argument("--label", choices=("human", "ai"))
    infer.add_argument("--count", type=int, default=0)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--include-seen", action="store_true")
    infer.add_argument("--output-csv", type=Path)
    infer.add_argument("--failure-json", type=Path)
    infer.set_defaults(func=command_infer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.output_dir / "midi_feature_extratrees_train.log"
        log_stream = log_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = TeeStream(sys.stdout, log_stream)
        sys.stderr = TeeStream(sys.stderr, log_stream)
        print(f"Training log: {log_path.resolve()}", flush=True)

    print(f"MIDI Feature Extra Trees build: {MIDI_FEATURE_EXTRATREES_BUILD}", flush=True)
    args.func(args)


if __name__ == "__main__":
    main()
