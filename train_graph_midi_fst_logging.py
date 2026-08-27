"""Train and run MIDI-FST with a frozen/partially-frozen MusicBERT-base.

The OctupleMIDI conversion follows Microsoft Muzic's MusicBERT preprocessing
(MIT License). The FST itself remains the implementation shipped in this
repository.

This file deliberately keeps the MIDI experiment separate from the original
waveform inference path.  It provides five independent commands:

  make-dictionary  Create MusicBERT's deterministic OctupleMIDI dictionary.
  make-split       Balance real/fake songs, then split whole MIDI files into
                   train/validation sets.
  train-stage1     Fine-tune only the last six MusicBERT-base layers on
                   four-bar segment labels.
  train-stage2     Freeze MusicBERT, build relation graphs over four-bar segments,
                   and train relation-specific GAT + FusionSegmentTransformer.
  infer            Classify one MIDI file, or evaluate a labeled MIDI folder.

No training command automatically starts another stage or evaluates external
datasets.  The user explicitly invokes each command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import miditoolkit
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError as exc:  # Give a useful message before any project imports.
    raise SystemExit(
        "Missing MIDI-FST dependencies. Activate the project environment and "
        "run: pip install -r requirements-midi.txt"
    ) from exc

from MIDI_FST.models.model import FusionSegmentTransformer

try:
    from midi_relation_graph_edges import build_relation_graph
except ImportError as exc:
    raise SystemExit(
        "Could not import midi_relation_graph_edges.py. Put it in the same directory "
        "as this script."
    ) from exc


MIDI_FST_BUILD = "2026-08-27-relation-gat-fst-r1"
GRAPH_ARCHITECTURE = "relation-specific-gat"
GRAPH_DISPLAY_NAME = "relation-specific GAT"


# MusicBERT's official OctupleMIDI constants.
POS_RESOLUTION = 16
BAR_MAX = 256
VELOCITY_QUANT = 4
TEMPO_QUANT = 12
MIN_TEMPO = 16
MAX_TEMPO = 256
DURATION_MAX = 8
MAX_TS_DENOMINATOR = 6
MAX_NOTES_PER_BAR = 2
BEAT_NOTE_FACTOR = 4
MAX_INST = 127
MAX_PITCH = 127
TOKENS_PER_NOTE = 8

SEGMENT_BARS = 4
MAX_SEGMENTS = 48
MUSICBERT_BASE_LAYERS = 12
MUSICBERT_BASE_DIM = 768
STAGE1_TRAINABLE_LAYERS = 6
MAX_NOTES_PER_SEGMENT = 1000


def _build_time_signature_tables() -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    table: dict[tuple[int, int], int] = {}
    values: list[tuple[int, int]] = []
    for i in range(MAX_TS_DENOMINATOR + 1):
        for numerator in range(1, (2**i) * MAX_NOTES_PER_BAR + 1):
            table[(numerator, 2**i)] = len(table)
            values.append((numerator, 2**i))
    return table, values


TS_DICT, TS_LIST = _build_time_signature_tables()


def _build_duration_tables() -> tuple[list[int], list[int]]:
    encoded: list[int] = []
    decoded: list[int] = []
    for i in range(DURATION_MAX):
        for _ in range(POS_RESOLUTION):
            decoded.append(len(encoded))
            for _ in range(2**i):
                encoded.append(len(decoded) - 1)
    return encoded, decoded


DUR_ENC, _ = _build_duration_tables()


def velocity_to_id(value: int) -> int:
    return value // VELOCITY_QUANT


def tempo_to_id(value: float) -> int:
    value = max(value, MIN_TEMPO)
    value = min(value, MAX_TEMPO)
    return round(math.log2(value / MIN_TEMPO) * TEMPO_QUANT)


def duration_to_id(value: int) -> int:
    return DUR_ENC[value] if value < len(DUR_ENC) else DUR_ENC[-1]


def reduce_time_signature(numerator: int, denominator: int) -> tuple[int, int]:
    while (
        denominator > 2**MAX_TS_DENOMINATOR
        and denominator % 2 == 0
        and numerator % 2 == 0
    ):
        denominator //= 2
        numerator //= 2
    while numerator > MAX_NOTES_PER_BAR * denominator:
        for divisor in range(2, numerator + 1):
            if numerator % divisor == 0:
                numerator //= divisor
                break
    reduced = (numerator, denominator)
    if reduced not in TS_DICT:
        raise ValueError(f"Unsupported time signature after reduction: {reduced}")
    return reduced


def midi_to_octuple_encoding(midi_path: Path) -> list[tuple[int, ...]]:
    """Convert one MIDI file using MusicBERT's official OctupleMIDI rules."""
    midi = miditoolkit.MidiFile(str(midi_path))

    def time_to_pos(tick: int) -> int:
        return round(tick * POS_RESOLUTION / midi.ticks_per_beat)

    note_starts = [time_to_pos(note.start) for inst in midi.instruments for note in inst.notes]
    if not note_starts:
        raise ValueError("MIDI contains no notes")

    # Same approximate 30-minute guard used by MusicBERT.
    max_pos = min(max(note_starts) + 1, 2**16)
    pos_info: list[list[int | None]] = [[None, None, None, None] for _ in range(max_pos)]

    for index, change in enumerate(midi.time_signature_changes):
        start = time_to_pos(change.time)
        end = (
            time_to_pos(midi.time_signature_changes[index + 1].time)
            if index + 1 < len(midi.time_signature_changes)
            else max_pos
        )
        ts_id = TS_DICT[reduce_time_signature(change.numerator, change.denominator)]
        for position in range(start, min(end, max_pos)):
            pos_info[position][1] = ts_id

    for index, change in enumerate(midi.tempo_changes):
        start = time_to_pos(change.time)
        end = (
            time_to_pos(midi.tempo_changes[index + 1].time)
            if index + 1 < len(midi.tempo_changes)
            else max_pos
        )
        tempo_id = tempo_to_id(change.tempo)
        for position in range(start, min(end, max_pos)):
            pos_info[position][3] = tempo_id

    default_ts = TS_DICT[reduce_time_signature(4, 4)]
    default_tempo = tempo_to_id(120.0)
    for info in pos_info:
        if info[1] is None:
            info[1] = default_ts
        if info[3] is None:
            info[3] = default_tempo

    count = 0
    bar = 0
    measure_length = 0
    for info in pos_info:
        ts_num, ts_den = TS_LIST[int(info[1])]
        if count == 0:
            measure_length = ts_num * BEAT_NOTE_FACTOR * POS_RESOLUTION // ts_den
        info[0] = bar
        info[2] = count
        count += 1
        if count >= measure_length:
            if count != measure_length:
                raise ValueError("Time-signature change is not aligned to a measure boundary")
            count -= measure_length
            bar += 1

    encoding: list[tuple[int, ...]] = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            start_pos = time_to_pos(note.start)
            if start_pos >= max_pos:
                continue
            info = pos_info[start_pos]
            program = MAX_INST + 1 if instrument.is_drum else instrument.program
            pitch = note.pitch + MAX_PITCH + 1 if instrument.is_drum else note.pitch
            duration = duration_to_id(time_to_pos(note.end) - start_pos)
            encoding.append(
                (
                    int(info[0]),
                    int(info[2]),
                    int(program),
                    int(pitch),
                    int(duration),
                    int(velocity_to_id(note.velocity)),
                    int(info[1]),
                    int(info[3]),
                )
            )

    if not encoding:
        raise ValueError("MIDI contains no encodable notes")
    encoding.sort()
    return encoding


def split_into_four_bar_segments(encoding: Sequence[tuple[int, ...]]) -> list[list[tuple[int, ...]]]:
    """Create non-overlapping four-bar groups; retain a non-empty final partial group."""
    maximum_bar = max(note[0] for note in encoding)
    segments: list[list[tuple[int, ...]]] = []
    for start_bar in range(0, maximum_bar + 1, SEGMENT_BARS):
        segment = [
            (note[0] - start_bar, *note[1:])
            for note in encoding
            if start_bar <= note[0] < start_bar + SEGMENT_BARS
        ]
        if segment:
            segments.append(segment[:MAX_NOTES_PER_SEGMENT])
    return segments


def discover_midis(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"MIDI directory does not exist: {directory}")
    paths = sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    )
    if not paths:
        raise ValueError(f"No .mid or .midi files found under: {directory}")
    return paths


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


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format") != "midi-fst-split-v2":
        raise ValueError(f"Unsupported split manifest: {path}")
    for split_name in ("train", "validation"):
        if not manifest.get(split_name):
            raise ValueError(f"Manifest split is empty: {split_name}")
        for item in manifest[split_name]:
            midi_path = Path(item["path"])
            if not midi_path.is_file():
                raise FileNotFoundError(f"Manifest MIDI file is missing: {midi_path}")
            if item["label"] not in (0, 1):
                raise ValueError(f"Invalid label in manifest: {item}")
    return manifest


def command_make_split(args: argparse.Namespace) -> None:
    if args.val_ratio <= 0 or args.val_ratio >= 0.5:
        raise ValueError("--val-ratio must be greater than 0 and less than 0.5")

    use_per_directory_counts = args.real_counts is not None or args.fake_counts is not None
    if use_per_directory_counts:
        if args.real_counts is None or args.fake_counts is None:
            raise ValueError("--real-counts and --fake-counts must be provided together")
        if len(args.real_counts) != len(args.real_dir):
            raise ValueError(
                f"--real-counts has {len(args.real_counts)} values, but "
                f"--real-dir has {len(args.real_dir)} directories"
            )
        if len(args.fake_counts) != len(args.fake_dir):
            raise ValueError(
                f"--fake-counts has {len(args.fake_counts)} values, but "
                f"--fake-dir has {len(args.fake_dir)} directories"
            )
        if any(count < 1 for count in args.real_counts + args.fake_counts):
            raise ValueError("Every value in --real-counts/--fake-counts must be at least 1")
    else:
        if args.per_class_count is None:
            raise ValueError(
                "Provide either --per-class-count or both --real-counts and --fake-counts"
            )
        if args.per_class_count < 2:
            raise ValueError("--per-class-count must be at least 2")

    rng = random.Random(args.seed)

    def select_usable(
        paths: list[Path], source: str, requested_count: int
    ) -> tuple[list[Path], list[dict[str, str]], int]:
        if requested_count > len(paths):
            raise ValueError(
                f"Requested {requested_count} MIDI files from {source}, but only "
                f"{len(paths)} candidate files are available"
            )

        shuffled = paths[:]
        rng.shuffle(shuffled)
        selected: list[Path] = []
        rejected: list[dict[str, str]] = []
        checked = 0
        for path in shuffled:
            checked += 1
            try:
                midi = miditoolkit.MidiFile(str(path))
                if not any(instrument.notes for instrument in midi.instruments):
                    raise ValueError("MIDI contains no notes")
                selected.append(path)
                if len(selected) == requested_count:
                    break
            except Exception as exc:
                rejected.append({"path": str(path), "source": source, "error": str(exc)})

        if len(selected) < requested_count:
            raise ValueError(
                f"Requested {requested_count} usable MIDI files from {source}, "
                f"but only {len(selected)} passed the open/note check after "
                f"checking all {len(shuffled)} candidates"
            )
        return selected, rejected, checked

    per_directory_selection: list[dict[str, Any]] = []
    rejected_real: list[dict[str, str]] = []
    rejected_fake: list[dict[str, str]] = []
    checked_real = 0
    checked_fake = 0
    available_real = 0
    available_fake = 0

    if use_per_directory_counts:
        real_paths: list[Path] = []
        for directory, requested_count in zip(args.real_dir, args.real_counts):
            candidates = discover_midis(directory)
            selected, rejected, checked = select_usable(
                candidates, f"real:{directory.resolve()}", requested_count
            )
            real_paths.extend(selected)
            rejected_real.extend(rejected)
            checked_real += checked
            available_real += len(candidates)
            per_directory_selection.append(
                {
                    "label": "real",
                    "directory": str(directory.resolve()),
                    "requested": requested_count,
                    "available": len(candidates),
                    "selected": len(selected),
                    "checked": checked,
                    "rejected": len(rejected),
                }
            )

        fake_paths: list[Path] = []
        for directory, requested_count in zip(args.fake_dir, args.fake_counts):
            candidates = discover_midis(directory)
            selected, rejected, checked = select_usable(
                candidates, f"fake:{directory.resolve()}", requested_count
            )
            fake_paths.extend(selected)
            rejected_fake.extend(rejected)
            checked_fake += checked
            available_fake += len(candidates)
            per_directory_selection.append(
                {
                    "label": "fake",
                    "directory": str(directory.resolve()),
                    "requested": requested_count,
                    "available": len(candidates),
                    "selected": len(selected),
                    "checked": checked,
                    "rejected": len(rejected),
                }
            )
    else:
        real_candidates = sorted(
            {path for directory in args.real_dir for path in discover_midis(directory)}
        )
        fake_candidates = sorted(
            {path for directory in args.fake_dir for path in discover_midis(directory)}
        )
        available_real = len(real_candidates)
        available_fake = len(fake_candidates)
        real_paths, rejected_real, checked_real = select_usable(
            real_candidates, "real", args.per_class_count
        )
        fake_paths, rejected_fake, checked_fake = select_usable(
            fake_candidates, "fake", args.per_class_count
        )

    if len(real_paths) < 2 or len(fake_paths) < 2:
        raise ValueError("At least two selected MIDI files are required for each class")

    if len(set(real_paths)) != len(real_paths):
        raise ValueError("The real directories contain overlapping resolved MIDI paths")
    if len(set(fake_paths)) != len(fake_paths):
        raise ValueError("The fake directories contain overlapping resolved MIDI paths")

    overlap = set(real_paths) & set(fake_paths)
    if overlap:
        example = next(iter(overlap))
        raise ValueError(
            "The real and fake directories contain the same resolved MIDI path: "
            f"{example}"
        )

    def divide(paths: list[Path], label: int, source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        shuffled = paths[:]
        rng.shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * args.val_ratio))
        if validation_count >= len(shuffled):
            raise ValueError(f"At least two {source} MIDI files are required")
        validation = shuffled[:validation_count]
        train = shuffled[validation_count:]

        def records(items: Iterable[Path]) -> list[dict[str, Any]]:
            return [{"path": str(path), "label": label, "source": source} for path in items]

        return records(train), records(validation)

    real_train, real_validation = divide(real_paths, 0, "real")
    fake_train, fake_validation = divide(fake_paths, 1, "fake")
    train = real_train + fake_train
    validation = real_validation + fake_validation
    rng.shuffle(train)
    rng.shuffle(validation)

    selected_real = len(real_paths)
    selected_fake = len(fake_paths)
    manifest = {
        "format": "midi-fst-split-v2",
        "created_at_unix": int(time.time()),
        "seed": args.seed,
        "validation_ratio": args.val_ratio,
        "segment_bars": SEGMENT_BARS,
        "selection": {
            "method": (
                "seeded-random-per-directory-after-midi-open-and-note-check"
                if use_per_directory_counts
                else "seeded-random-after-midi-open-and-note-check"
            ),
            "requested_per_class": args.per_class_count,
            "requested_real_counts": args.real_counts,
            "requested_fake_counts": args.fake_counts,
            "per_directory": per_directory_selection,
            "available_real": available_real,
            "available_fake": available_fake,
            "checked_real": checked_real,
            "checked_fake": checked_fake,
            "rejected_real": len(rejected_real),
            "rejected_fake": len(rejected_fake),
            "unexamined_real": available_real - checked_real,
            "unexamined_fake": available_fake - checked_fake,
        },
        "rejected_during_selection": rejected_real + rejected_fake,
        "train": train,
        "validation": validation,
        "counts": {
            "selected_real": selected_real,
            "selected_fake": selected_fake,
            "train_real": len(real_train),
            "train_fake": len(fake_train),
            "validation_real": len(real_validation),
            "validation_fake": len(fake_validation),
        },
    }
    save_json(args.output, manifest)
    print(f"Split manifest saved: {args.output.resolve()}")
    if per_directory_selection:
        for item in per_directory_selection:
            print(
                f"  {item['label']} | {item['directory']} | "
                f"selected {item['selected']}/{item['requested']} "
                f"(available {item['available']}, rejected {item['rejected']})"
            )
    print(f"  available_real: {available_real}")
    print(f"  available_fake: {available_fake}")
    print(f"  checked_real: {checked_real}")
    print(f"  checked_fake: {checked_fake}")
    print(f"  rejected_real: {len(rejected_real)}")
    print(f"  rejected_fake: {len(rejected_fake)}")
    for name, count in manifest["counts"].items():
        print(f"  {name}: {count}")

def command_make_dictionary(args: argparse.Namespace) -> None:
    """Create the deterministic dictionary expected by official MusicBERT checkpoints."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dictionary_path = args.output_dir / "dict.txt"
    tokens: list[str] = []
    tokens.extend(f"<0-{value}>" for value in range(BAR_MAX))
    tokens.extend(
        f"<1-{value}>"
        for value in range(BEAT_NOTE_FACTOR * MAX_NOTES_PER_BAR * POS_RESOLUTION)
    )
    tokens.extend(f"<2-{value}>" for value in range(MAX_INST + 2))
    tokens.extend(f"<3-{value}>" for value in range(2 * MAX_PITCH + 2))
    tokens.extend(f"<4-{value}>" for value in range(DURATION_MAX * POS_RESOLUTION))
    tokens.extend(f"<5-{value}>" for value in range(velocity_to_id(127) + 1))
    tokens.extend(f"<6-{value}>" for value in range(len(TS_LIST)))
    tokens.extend(f"<7-{value}>" for value in range(tempo_to_id(MAX_TEMPO) + 1))
    with dictionary_path.open("w", encoding="utf-8") as stream:
        for token in tokens:
            # Keep the frequency column identical to MusicBERT's generator.
            stream.write(f"{token} 0\n")
    print(f"MusicBERT dictionary saved: {dictionary_path.resolve()}")
    print(f"Dictionary tokens (excluding Fairseq special tokens): {len(tokens)}")


class MusicBERTAdapter(nn.Module):
    """Thin access layer around Microsoft's Fairseq MusicBERT implementation."""

    def __init__(
        self,
        muzic_dir: Path,
        checkpoint: Path,
        data_bin: Path,
        stage1_checkpoint: Path | None = None,
    ) -> None:
        super().__init__()
        musicbert_root = (muzic_dir / "musicbert").resolve()
        user_dir = musicbert_root / "musicbert"
        if not (user_dir / "__init__.py").is_file():
            raise FileNotFoundError(
                f"MusicBERT code was not found. Expected: {user_dir / '__init__.py'}"
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MusicBERT checkpoint not found: {checkpoint}")
        if not (data_bin / "dict.txt").is_file():
            raise FileNotFoundError(
                f"MusicBERT dict.txt not found in: {data_bin}. "
                "Run the make-dictionary command first."
            )

        # MusicBERT imports TransformerSentenceEncoder from the Fairseq 0.10
        # export location. Fairseq 0.12 keeps the class in fairseq.modules.
        try:
            import fairseq.models.roberta as roberta_module
            if not hasattr(roberta_module, "TransformerSentenceEncoder"):
                from fairseq.modules import TransformerSentenceEncoder

                roberta_module.TransformerSentenceEncoder = TransformerSentenceEncoder
            from fairseq import (
                checkpoint_utils,
                tasks as fairseq_tasks,
                utils as fairseq_utils,
            )
            from fairseq.models import ARCH_MODEL_REGISTRY
        except ImportError as exc:
            raise RuntimeError(
                "Fairseq could not be imported. Use Python 3.9 and install "
                "fairseq==0.12.2 from requirements-midi.txt."
            ) from exc

        checkpoint = checkpoint.resolve()
        data_bin = data_bin.resolve()
        self.base_checkpoint_signature = file_signature(checkpoint)

        # Import the Muzic user module first so Fairseq registers
        # MusicBERTModel and the musicbert_base architecture.
        fairseq_utils.import_user_module(argparse.Namespace(user_dir=str(user_dir)))

        # Some released pre-training checkpoints describe the model as
        # roberta_base even though the state dict contains MusicBERT's Octuple
        # down/upsampling layers. The generic Fairseq ensemble loader can still
        # rebuild RobertaModel from that metadata. Instead, instantiate the
        # registered MusicBERTModel directly and then load its weights.
        state = checkpoint_utils.load_checkpoint_to_cpu(str(checkpoint))
        checkpoint_args = state.get("args")
        if checkpoint_args is not None:
            checkpoint_args.arch = "musicbert_base"
            checkpoint_args.data = str(data_bin)

        checkpoint_cfg = state.get("cfg")
        if checkpoint_cfg is not None:
            from omegaconf import OmegaConf, open_dict

            if OmegaConf.is_config(checkpoint_cfg):
                with open_dict(checkpoint_cfg):
                    checkpoint_cfg.model._name = "musicbert_base"
                    checkpoint_cfg.model.arch = "musicbert_base"
                    checkpoint_cfg.task.data = str(data_bin)
            else:
                checkpoint_cfg["model"]["_name"] = "musicbert_base"
                checkpoint_cfg["model"]["arch"] = "musicbert_base"
                checkpoint_cfg["task"]["data"] = str(data_bin)

        if checkpoint_cfg is None:
            raise ValueError("MusicBERT checkpoint does not contain cfg metadata")
        if "musicbert_base" not in ARCH_MODEL_REGISTRY:
            raise RuntimeError("The Muzic user module did not register musicbert_base")

        task = fairseq_tasks.setup_task(checkpoint_cfg.task)
        musicbert_class = ARCH_MODEL_REGISTRY["musicbert_base"]
        self.model = musicbert_class.build_model(checkpoint_cfg.model, task)

        # The released checkpoint was written by a Fairseq revision that named
        # this module layernorm_embedding. Fairseq 0.12.2 calls the same module
        # emb_layer_norm. The tensors and their roles are otherwise identical.
        model_state = dict(state["model"])
        layer_norm_aliases = {
            "encoder.sentence_encoder.layernorm_embedding.weight":
                "encoder.sentence_encoder.emb_layer_norm.weight",
            "encoder.sentence_encoder.layernorm_embedding.bias":
                "encoder.sentence_encoder.emb_layer_norm.bias",
        }
        for old_key, new_key in layer_norm_aliases.items():
            if old_key in model_state:
                if new_key in model_state:
                    raise ValueError(
                        f"Checkpoint contains both LayerNorm aliases: {old_key}, {new_key}"
                    )
                model_state[new_key] = model_state.pop(old_key)

        expected_state = self.model.state_dict()
        missing_keys = sorted(set(expected_state) - set(model_state))
        unexpected_keys = sorted(set(model_state) - set(expected_state))
        shape_mismatches = [
            (
                key,
                tuple(model_state[key].shape),
                tuple(expected_state[key].shape),
            )
            for key in sorted(set(model_state) & set(expected_state))
            if tuple(model_state[key].shape) != tuple(expected_state[key].shape)
        ]
        if missing_keys or unexpected_keys or shape_mismatches:
            raise ValueError(
                "MusicBERT checkpoint compatibility check failed: "
                f"missing={missing_keys}, unexpected={unexpected_keys}, "
                f"shape_mismatches={shape_mismatches}"
            )

        incompatible = nn.Module.load_state_dict(
            self.model,
            model_state,
            strict=True,
            )
        
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                "MusicBERT checkpoint keys did not match the direct model: "
                f"missing={list(incompatible.missing_keys)}, "
                f"unexpected={list(incompatible.unexpected_keys)}"
            )
        if not hasattr(self.model.encoder.sentence_encoder, "downsampling"):
            raise RuntimeError("Loaded backbone is missing OctupleMIDI downsampling")
        print(f"Loaded backbone: {type(self.model).__name__}", flush=True)
        self.dictionary = task.source_dictionary
        self._token_id_cache: dict[str, int] = {}

        layers = self.layers
        model_args = getattr(self.model, "args", None)
        embed_dim = int(getattr(model_args, "encoder_embed_dim", 0))
        if not embed_dim:
            embed_dim = int(
                getattr(self.model.encoder.sentence_encoder, "embedding_dim", 0)
            )
        if len(layers) != MUSICBERT_BASE_LAYERS or embed_dim != MUSICBERT_BASE_DIM:
            raise ValueError(
                "The supplied checkpoint is not MusicBERT-base: "
                f"layers={len(layers)}, embedding_dim={embed_dim}. "
                "Expected 12 layers and 768 dimensions."
            )
        if stage1_checkpoint is not None:
            self.load_stage1_delta(stage1_checkpoint)

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.encoder.sentence_encoder.layers

    @property
    def pad_id(self) -> int:
        return int(self.dictionary.pad())

    def token_id(self, token: str) -> int:
        cached = self._token_id_cache.get(token)
        if cached is not None:
            return cached
        token_id = int(self.dictionary.index(token))
        if token_id == int(self.dictionary.unk()):
            raise KeyError(f"Token is missing from the MusicBERT dictionary: {token}")
        self._token_id_cache[token] = token_id
        return token_id

    def encode_segment(self, segment: Sequence[tuple[int, ...]]) -> torch.Tensor:
        ids = [int(self.dictionary.bos())] * TOKENS_PER_NOTE
        for note in segment[:MAX_NOTES_PER_SEGMENT]:
            ids.extend(self.token_id(f"<{element}-{value}>") for element, value in enumerate(note))
        ids.extend([int(self.dictionary.eos())] * TOKENS_PER_NOTE)
        if len(ids) % TOKENS_PER_NOTE != 0:
            raise AssertionError("OctupleMIDI token length is not divisible by eight")
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, sentence_rep = self.model.encoder.sentence_encoder(
            tokens,
            last_state_only = True,
        )
        return sentence_rep

    def configure_stage1(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for layer in self.layers[-STAGE1_TRAINABLE_LAYERS:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        self.model.train()

    def freeze_for_stage2(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    def trainable_delta(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def load_stage1_delta(self, checkpoint_path: Path) -> dict[str, Any]:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "midi-fst-stage1-v1":
            raise ValueError(f"Unsupported Stage 1 checkpoint: {checkpoint_path}")
        if payload.get("musicbert_base_signature") != self.base_checkpoint_signature:
            raise ValueError(
                "Stage 1 was created from a different MusicBERT-base checkpoint. "
                "Use the same --musicbert-checkpoint for every command."
            )
        incompatible = self.model.load_state_dict(payload["musicbert_delta"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise ValueError(f"Unexpected MusicBERT keys in Stage 1 checkpoint: {unexpected}")
        return payload


class Stage1Head(nn.Module):
    """RoBERTa-style temporary segment classifier used only during Stage 1."""

    def __init__(self, input_dim: int = MUSICBERT_BASE_DIM, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(input_dim, input_dim)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dropout(features)
        features = torch.tanh(self.dense(features))
        features = self.dropout(features)
        return self.output(features).squeeze(-1)


def cache_key(path: Path, extra: str) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{extra}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_signature(path: Path) -> str:
    """Identify checkpoint contents while reading in bounded-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_token_cache(
    items: Sequence[dict[str, Any]],
    adapter: MusicBERTAdapter,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total = len(items)
    for position, item in enumerate(items, start=1):
        source_path = Path(item["path"])
        key = cache_key(source_path, "octuple-4bar-v1")
        cached_path = cache_dir / f"{key}.pt"
        try:
            if cached_path.is_file():
                cached = torch.load(cached_path, map_location="cpu", weights_only=False)
                tokens = cached["tokens"]
            else:
                encoding = midi_to_octuple_encoding(source_path)
                segments = split_into_four_bar_segments(encoding)
                if not segments:
                    raise ValueError("No four-bar segment could be created")
                tokens = [adapter.encode_segment(segment) for segment in segments]
                torch.save(
                    {
                        "format": "midi-fst-token-cache-v1",
                        "source": str(source_path.resolve()),
                        "tokens": tokens,
                    },
                    cached_path,
                )
            for segment_index in range(len(tokens)):
                records.append(
                    {
                        "cache_path": str(cached_path),
                        "segment_index": segment_index,
                        "label": int(item["label"]),
                        "source_path": str(source_path),
                    }
                )
        except Exception as exc:  # Record malformed MIDI without hiding the path/reason.
            failures.append({"path": str(source_path), "error": str(exc)})
        if position == total or position % 100 == 0:
            print(f"Token cache: {position}/{total} MIDI files", flush=True)
    if not records:
        raise RuntimeError("No usable four-bar MIDI segments were produced")
    return records, failures


class SegmentTokenDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = list(records)
        self.labels = [int(record["label"]) for record in self.records]
        self._loaded_path: str | None = None
        self._loaded_tokens: list[torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        cache_path = record["cache_path"]
        if cache_path != self._loaded_path:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self._loaded_path = cache_path
            self._loaded_tokens = payload["tokens"]
        assert self._loaded_tokens is not None
        return self._loaded_tokens[record["segment_index"]], int(record["label"])


def collate_tokens(batch: Sequence[tuple[torch.Tensor, int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    maximum = max(sequence.numel() for sequence in sequences)
    if maximum % TOKENS_PER_NOTE:
        maximum += TOKENS_PER_NOTE - (maximum % TOKENS_PER_NOTE)
    padded = torch.full((len(sequences), maximum), pad_id, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        padded[index, : sequence.numel()] = sequence
    return padded, torch.tensor(labels, dtype=torch.float32)


def balanced_sampler(labels: Sequence[int], seed: int) -> WeightedRandomSampler:
    counts = Counter(labels)
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    balanced_accuracy: float


def metrics_from_outputs(total_loss: float, labels: list[int], predictions: list[int]) -> EpochMetrics:
    if not labels:
        return EpochMetrics(float("nan"), float("nan"), float("nan"))
    correct = sum(int(a == b) for a, b in zip(labels, predictions))
    recalls: list[float] = []
    for label in (0, 1):
        indices = [index for index, value in enumerate(labels) if value == label]
        if indices:
            recalls.append(sum(int(predictions[index] == label) for index in indices) / len(indices))
    return EpochMetrics(
        loss=total_loss / len(labels),
        accuracy=correct / len(labels),
        balanced_accuracy=sum(recalls) / len(recalls),
    )


class TrainingMonitor:
    """Save epoch logs, TensorBoard scalars, and train/validation loss curves."""

    def __init__(self, output_dir: Path, stage_name: str) -> None:
        self.output_dir = output_dir
        self.stage_name = stage_name
        self.log_path = output_dir / f"{stage_name}_training.log"
        self.plot_path = output_dir / f"{stage_name}_loss_curve.png"
        self.train_losses: list[float] = []
        self.validation_losses: list[float] = []

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard logging requires the tensorboard package. "
                "Install it with: pip install tensorboard"
            ) from exc

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Loss-curve plotting requires matplotlib. "
                "Install it with: pip install matplotlib"
            ) from exc

        self.plt = plt
        self.writer = SummaryWriter(log_dir=str(output_dir / "tensorboard" / stage_name))

        with self.log_path.open("w", encoding="utf-8") as stream:
            stream.write(
                "epoch,train_loss,validation_loss,"
                "train_accuracy,validation_accuracy,"
                "train_balanced_accuracy,validation_balanced_accuracy\n"
            )

    def record(
        self, epoch: int, train_metrics: EpochMetrics, validation_metrics: EpochMetrics
    ) -> None:
        self.train_losses.append(train_metrics.loss)
        self.validation_losses.append(validation_metrics.loss)

        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"{epoch},{train_metrics.loss:.8f},{validation_metrics.loss:.8f},"
                f"{train_metrics.accuracy:.8f},{validation_metrics.accuracy:.8f},"
                f"{train_metrics.balanced_accuracy:.8f},"
                f"{validation_metrics.balanced_accuracy:.8f}\n"
            )

        self.writer.add_scalars(
            "Loss",
            {"train": train_metrics.loss, "validation": validation_metrics.loss},
            epoch,
        )
        self.writer.add_scalars(
            "BalancedAccuracy",
            {
                "train": train_metrics.balanced_accuracy,
                "validation": validation_metrics.balanced_accuracy,
            },
            epoch,
        )
        self.writer.flush()
        self._save_loss_curve()

    def _save_loss_curve(self) -> None:
        epochs = range(1, len(self.train_losses) + 1)
        figure = self.plt.figure(figsize=(8, 5))
        self.plt.plot(epochs, self.train_losses, label="Train loss")
        self.plt.plot(epochs, self.validation_losses, label="Validation loss")
        self.plt.xlabel("Epoch")
        self.plt.ylabel("Loss")
        self.plt.title(f"{self.stage_name.upper()} train/validation loss")
        self.plt.legend()
        self.plt.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(self.plot_path, dpi=150)
        self.plt.close(figure)

    def close(self) -> None:
        self.writer.close()


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def stage1_epoch(
    adapter: MusicBERTAdapter,
    head: Stage1Head,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    grad_accum: int,
    amp: bool,
) -> EpochMetrics:
    training = optimizer is not None
    adapter.model.train(training)
    head.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for step, (tokens, labels) in enumerate(loader, start=1):
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with autocast_context(device, amp):
                logits = head(adapter(tokens))
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            if training:
                scaler.scale(loss / grad_accum).backward()
                if step % grad_accum == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for group in optimizer.param_groups for parameter in group["params"]],
                        0.5,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        batch_predictions = (torch.sigmoid(logits) >= 0.5).long()
        total_loss += float(loss.detach()) * labels.numel()
        all_labels.extend(labels.long().detach().cpu().tolist())
        all_predictions.extend(batch_predictions.detach().cpu().tolist())

    return metrics_from_outputs(total_loss, all_labels, all_predictions)


def command_train_stage1(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    manifest = load_manifest(args.split_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = MusicBERTAdapter(
        args.muzic_dir,
        args.musicbert_checkpoint,
        args.musicbert_data_bin,
    ).to(device)
    adapter.configure_stage1()
    head = Stage1Head().to(device)

    cache_root = args.cache_dir or args.output_dir / "cache"
    train_records, train_failures = build_token_cache(
        manifest["train"], adapter, cache_root / "tokens"
    )
    validation_records, validation_failures = build_token_cache(
        manifest["validation"], adapter, cache_root / "tokens"
    )
    failures = train_failures + validation_failures
    if failures:
        save_json(args.output_dir / "midi_preprocess_failures.json", failures)
        print(f"Skipped malformed/unusable MIDI files: {len(failures)}")

    train_dataset = SegmentTokenDataset(train_records)
    validation_dataset = SegmentTokenDataset(validation_records)
    collate = lambda batch: collate_tokens(batch, adapter.pad_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=balanced_sampler(train_dataset.labels, args.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    parameters.extend(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    checkpoint_path = args.output_dir / "musicbert_stage1_best.pt"
    monitor = TrainingMonitor(args.output_dir, "stage1")
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = stage1_epoch(
            adapter,
            head,
            train_loader,
            device,
            optimizer,
            scaler,
            args.grad_accum,
            args.amp,
        )
        validation_metrics = stage1_epoch(
            adapter,
            head,
            validation_loader,
            device,
            None,
            scaler,
            1,
            args.amp,
        )
        print(
            f"Stage 1 epoch {epoch:03d} | "
            f"train loss {train_metrics.loss:.4f}, bal-acc {train_metrics.balanced_accuracy:.4f} | "
            f"val loss {validation_metrics.loss:.4f}, bal-acc {validation_metrics.balanced_accuracy:.4f}",
            flush=True,
        )
        monitor.record(epoch, train_metrics, validation_metrics)
        if validation_metrics.loss < best_loss:
            best_loss = validation_metrics.loss
            stale_epochs = 0
            torch.save(
                {
                    "format": "midi-fst-stage1-v1",
                    "musicbert_variant": "base",
                    "musicbert_layers": MUSICBERT_BASE_LAYERS,
                    "trainable_last_layers": STAGE1_TRAINABLE_LAYERS,
                    "embedding_dim": MUSICBERT_BASE_DIM,
                    "segment_bars": SEGMENT_BARS,
                    "musicbert_base_signature": adapter.base_checkpoint_signature,
                    "musicbert_delta": adapter.trainable_delta(),
                    "stage1_head": head.state_dict(),
                    "best_validation_loss": best_loss,
                    "split_manifest": str(args.split_manifest.resolve()),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Stage 1 early stopping after {epoch} epochs")
                break

    monitor.close()
    print("Stage 1 training finished.")
    print(f"Best checkpoint: {checkpoint_path.resolve()}")
    print(f"Training log: {monitor.log_path.resolve()}")
    print(f"TensorBoard log dir: {(args.output_dir / 'tensorboard' / 'stage1').resolve()}")
    print(f"Loss curve: {monitor.plot_path.resolve()}")



GRAPH_RELATIONS = ("rhythm", "melody", "harmony", "structure")


def build_song_embedding_cache(
    items: Sequence[dict[str, Any]],
    adapter: MusicBERTAdapter,
    device: torch.device,
    token_cache_dir: Path,
    embedding_cache_dir: Path,
    extraction_batch_size: int,
    stage1_signature: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build/reuse the same frozen-MusicBERT song embedding cache as MIDI-FST."""
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    token_records, failures = build_token_cache(items, adapter, token_cache_dir)
    records_by_source: dict[str, list[dict[str, Any]]] = {}
    for record in token_records:
        records_by_source.setdefault(record["source_path"], []).append(record)

    output_records: list[dict[str, Any]] = []
    for position, item in enumerate(items, start=1):
        source_path = str(Path(item["path"]))
        segment_records = records_by_source.get(source_path, [])
        if not segment_records:
            continue
        source = Path(source_path)
        key = cache_key(
            source,
            f"musicbert-stage1-4bar-768-v2|stage1={stage1_signature}",
        )
        output_path = embedding_cache_dir / f"{key}.pt"
        try:
            if not output_path.is_file():
                token_payload = torch.load(
                    segment_records[0]["cache_path"], map_location="cpu", weights_only=False
                )
                token_sequences = token_payload["tokens"][:MAX_SEGMENTS]
                embeddings: list[torch.Tensor] = []
                with torch.no_grad():
                    for start in range(0, len(token_sequences), extraction_batch_size):
                        batch_sequences = token_sequences[start : start + extraction_batch_size]
                        maximum = max(sequence.numel() for sequence in batch_sequences)
                        padded = torch.full(
                            (len(batch_sequences), maximum), adapter.pad_id, dtype=torch.long
                        )
                        for row, sequence in enumerate(batch_sequences):
                            padded[row, : sequence.numel()] = sequence
                        batch_embeddings = adapter(padded.to(device)).float().cpu()
                        embeddings.append(batch_embeddings)
                song_embeddings = torch.cat(embeddings, dim=0)
                torch.save(
                    {
                        "format": "midi-fst-song-embedding-v1",
                        "source": str(source.resolve()),
                        "embeddings": song_embeddings,
                    },
                    output_path,
                )
            output_records.append(
                {
                    "cache_path": str(output_path),
                    "label": int(item["label"]),
                    "source_path": str(source),
                }
            )
        except Exception as exc:
            failures.append({"path": str(source), "error": str(exc)})
        if position == len(items) or position % 100 == 0:
            print(f"Embedding cache: {position}/{len(items)} MIDI files", flush=True)
    if not output_records:
        raise RuntimeError("No song embeddings were produced")
    return output_records, failures


def four_bar_segment_start_bars(
    encoding: Sequence[tuple[int, ...]],
) -> list[int]:
    """Return start-bar indices in exactly the same order as MusicBERT segments."""
    maximum_bar = max(note[0] for note in encoding)
    starts: list[int] = []
    for start_bar in range(0, maximum_bar + 1, SEGMENT_BARS):
        has_note = any(
            start_bar <= note[0] < start_bar + SEGMENT_BARS
            for note in encoding
        )
        if has_note:
            starts.append(start_bar)
    return starts


def align_relation_graph_to_musicbert_segments(
    graph: dict[str, np.ndarray],
    segment_start_bars: Sequence[int],
) -> torch.Tensor:
    """Align raw-MIDI relation matrices to the retained MusicBERT segment order.

    Output shape: [4, T, T] in GRAPH_RELATIONS order.  A MusicBERT segment that
    has no complete graph node (for example a final partial four-bar segment)
    keeps only its self-loop instead of being silently dropped.
    """
    graph_start_bars = [int(value) for value in np.asarray(graph["start_bar"]).tolist()]
    graph_index = {start_bar: index for index, start_bar in enumerate(graph_start_bars)}
    segment_count = len(segment_start_bars)
    adjacency = np.zeros(
        (len(GRAPH_RELATIONS), segment_count, segment_count), dtype=np.float32
    )

    for relation_index, relation in enumerate(GRAPH_RELATIONS):
        relation_matrix = np.asarray(graph[relation], dtype=np.float32)
        if relation_matrix.shape != (len(graph_start_bars), len(graph_start_bars)):
            raise ValueError(
                f"Unexpected {relation} adjacency shape {relation_matrix.shape}; "
                f"expected {(len(graph_start_bars), len(graph_start_bars))}"
            )
        for target_index, target_bar in enumerate(segment_start_bars):
            graph_target = graph_index.get(int(target_bar))
            if graph_target is None:
                continue
            for source_index, source_bar in enumerate(segment_start_bars):
                graph_source = graph_index.get(int(source_bar))
                if graph_source is None:
                    continue
                adjacency[relation_index, target_index, source_index] = relation_matrix[
                    graph_target, graph_source
                ]
        np.fill_diagonal(adjacency[relation_index], 1.0)

    return torch.from_numpy(adjacency)


def build_graph_song_cache(
    embedding_records: Sequence[dict[str, Any]],
    graph_cache_dir: Path,
    recurrence_threshold: float,
    graph_top_k: int,
    stage1_signature: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build/reuse raw-MIDI relation graphs aligned to cached MusicBERT segments."""
    graph_cache_dir.mkdir(parents=True, exist_ok=True)
    output_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for position, record in enumerate(embedding_records, start=1):
        source = Path(record["source_path"])
        key = cache_key(
            source,
            (
                "relation-gat-graph-v1|"
                f"threshold={recurrence_threshold:.8g}|topk={graph_top_k}|"
                f"stage1={stage1_signature}"
            ),
        )
        graph_path = graph_cache_dir / f"{key}.pt"
        try:
            embedding_payload = torch.load(
                record["cache_path"], map_location="cpu", weights_only=False
            )
            embeddings = embedding_payload["embeddings"][:MAX_SEGMENTS].float()
            encoding = midi_to_octuple_encoding(source)
            segment_start_bars = four_bar_segment_start_bars(encoding)[:MAX_SEGMENTS]
            if len(segment_start_bars) != embeddings.shape[0]:
                raise ValueError(
                    "MusicBERT segment/start-bar count mismatch: "
                    f"embeddings={embeddings.shape[0]}, starts={len(segment_start_bars)}"
                )

            if not graph_path.is_file():
                graph = build_relation_graph(
                    source,
                    recurrence_threshold=recurrence_threshold,
                    top_k=graph_top_k,
                )
                adjacency = align_relation_graph_to_musicbert_segments(
                    graph, segment_start_bars
                )
                torch.save(
                    {
                        "format": "graph-midi-fst-relation-cache-v1",
                        "source": str(source.resolve()),
                        "relations": list(GRAPH_RELATIONS),
                        "segment_start_bars": segment_start_bars,
                        "adjacency": adjacency,
                        "recurrence_threshold": float(recurrence_threshold),
                        "graph_top_k": int(graph_top_k),
                    },
                    graph_path,
                )

            output_records.append(
                {
                    "embedding_cache_path": str(record["cache_path"]),
                    "graph_cache_path": str(graph_path),
                    "label": int(record["label"]),
                    "source_path": str(source),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "path": str(source),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if position == len(embedding_records) or position % 100 == 0:
            print(f"Relation graph cache: {position}/{len(embedding_records)} MIDI files", flush=True)

    if not output_records:
        raise RuntimeError("No aligned relation-graph song records were produced")
    return output_records, failures


class GraphSongEmbeddingDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = list(records)
        self.labels = [int(record["label"]) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        record = self.records[index]
        embedding_payload = torch.load(
            record["embedding_cache_path"], map_location="cpu", weights_only=False
        )
        graph_payload = torch.load(
            record["graph_cache_path"], map_location="cpu", weights_only=False
        )
        embeddings = embedding_payload["embeddings"].float()
        adjacency = graph_payload["adjacency"].float()
        if adjacency.shape[0] != len(GRAPH_RELATIONS):
            raise ValueError(
                f"Expected {len(GRAPH_RELATIONS)} graph relations, got {adjacency.shape[0]}"
            )
        if adjacency.shape[1:] != (embeddings.shape[0], embeddings.shape[0]):
            raise ValueError(
                "Cached graph/embedding length mismatch for "
                f"{record['source_path']}: embeddings={embeddings.shape[0]}, "
                f"adjacency={tuple(adjacency.shape)}"
            )
        return embeddings, adjacency, int(record["label"])


def collate_graph_song_embeddings(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings, adjacencies, labels = zip(*batch)
    batch_size = len(embeddings)
    padded_embeddings = torch.zeros(
        (batch_size, MAX_SEGMENTS, MUSICBERT_BASE_DIM), dtype=torch.float32
    )
    padded_adjacency = torch.zeros(
        (batch_size, len(GRAPH_RELATIONS), MAX_SEGMENTS, MAX_SEGMENTS),
        dtype=torch.float32,
    )
    padding_mask = torch.ones((batch_size, MAX_SEGMENTS), dtype=torch.bool)

    for index, (sequence, adjacency) in enumerate(zip(embeddings, adjacencies)):
        length = min(sequence.shape[0], MAX_SEGMENTS)
        if adjacency.shape[1] < length or adjacency.shape[2] < length:
            raise ValueError("Adjacency is shorter than its MusicBERT embedding sequence")
        padded_embeddings[index, :length] = sequence[:length]
        padded_adjacency[index, :, :length, :length] = adjacency[:, :length, :length]
        padding_mask[index, :length] = False

    return (
        padded_embeddings,
        padded_adjacency,
        torch.tensor(labels, dtype=torch.float32),
        padding_mask,
    )


class RelationGATBranch(nn.Module):
    """Dense multi-head GAT over one explicit MIDI relation.

    The adjacency matrix determines allowed neighbors and contributes its
    relation score as a logarithmic attention bias.  Node content comes only
    from MusicBERT embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError("Graph output_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = int(output_dim // num_heads)
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.attention_source = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim)
        )
        self.attention_target = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim)
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)

    def forward(
        self,
        embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = embeddings.shape
        projected = self.projection(embeddings).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )

        source_score = (projected * self.attention_source).sum(dim=-1)
        target_score = (projected * self.attention_target).sum(dim=-1)
        # [B, heads, target, source]
        scores = self.leaky_relu(
            target_score.permute(0, 2, 1).unsqueeze(-1)
            + source_score.permute(0, 2, 1).unsqueeze(-2)
        )

        positive_adjacency = adjacency.clamp_min(0.0)
        valid_node = ~padding_mask
        edge_mask = positive_adjacency > 0
        edge_mask = edge_mask & valid_node.unsqueeze(-1) & valid_node.unsqueeze(-2)

        # Preserve the supplied musical-relation strength without allowing a
        # non-edge to enter attention.
        edge_bias = torch.log(positive_adjacency.clamp_min(1e-6)).unsqueeze(1)
        scores = scores + edge_bias
        scores = scores.masked_fill(~edge_mask.unsqueeze(1), -1e4)

        attention = torch.softmax(scores, dim=-1)
        attention = attention.masked_fill(~edge_mask.unsqueeze(1), 0.0)
        attention = self.attention_dropout(attention)

        # projected: [B, source, heads, head_dim]
        messages = torch.einsum("bhts,bshd->bthd", attention, projected)
        messages = messages.reshape(batch_size, sequence_length, -1)
        messages = messages.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return messages


class MultiRelationGATEncoder(nn.Module):
    """Rhythm/Melody/Harmony/Structure GAT branches with residual fusion."""

    def __init__(
        self,
        input_dim: int = MUSICBERT_BASE_DIM,
        relation_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                RelationGATBranch(input_dim, relation_dim, num_heads, dropout)
                for _ in GRAPH_RELATIONS
            ]
        )
        self.fusion = nn.Linear(relation_dim * len(GRAPH_RELATIONS), input_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if adjacency.ndim != 4 or adjacency.shape[1] != len(GRAPH_RELATIONS):
            raise ValueError(
                "Adjacency must have shape [B, 4, T, T] in "
                "rhythm/melody/harmony/structure order"
            )
        relation_outputs = [
            branch(embeddings, adjacency[:, relation_index], padding_mask)
            for relation_index, branch in enumerate(self.branches)
        ]
        graph_message = torch.cat(relation_outputs, dim=-1)
        graph_message = self.dropout(self.activation(self.fusion(graph_message)))
        augmented = self.layer_norm(embeddings + graph_message)
        augmented = augmented.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return augmented


class GraphAugmentedMIDIFST(nn.Module):
    def __init__(
        self,
        input_dim: int,
        graph_relation_dim: int,
        graph_heads: int,
        graph_dropout: float,
        fst_hidden_dim: int,
        fst_num_heads: int,
        fst_num_layers: int,
        fst_dropout: float,
    ) -> None:
        super().__init__()
        self.graph_encoder = MultiRelationGATEncoder(
            input_dim=input_dim,
            relation_dim=graph_relation_dim,
            num_heads=graph_heads,
            dropout=graph_dropout,
        )
        self.fst = FusionSegmentTransformer(
            input_dim=input_dim,
            hidden_dim=fst_hidden_dim,
            num_heads=fst_num_heads,
            num_layers=fst_num_layers,
            dropout=fst_dropout,
            max_sequence_length=1000,
            num_classes=2,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        augmented = self.graph_encoder(embeddings, adjacency, padding_mask)
        return self.fst(augmented, padding_mask)


def graph_stage2_epoch(
    model: GraphAugmentedMIDIFST,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    amp: bool,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for embeddings, adjacency, labels, padding_mask in loader:
        embeddings = embeddings.to(device, non_blocking=True)
        adjacency = adjacency.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast_context(device, amp):
                logits = model(embeddings, adjacency, padding_mask).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        predictions = (torch.sigmoid(logits) >= 0.5).long()
        total_loss += float(loss.detach()) * labels.numel()
        all_labels.extend(labels.long().detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    return metrics_from_outputs(total_loss, all_labels, all_predictions)


def command_train_stage2(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    manifest = load_manifest(args.split_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter = MusicBERTAdapter(
        args.muzic_dir,
        args.musicbert_checkpoint,
        args.musicbert_data_bin,
        args.stage1_checkpoint,
    ).to(device)
    adapter.freeze_for_stage2()

    cache_root = args.cache_dir or args.output_dir / "cache"
    stage1_signature = file_signature(args.stage1_checkpoint)

    train_embedding_records, train_embedding_failures = build_song_embedding_cache(
        manifest["train"],
        adapter,
        device,
        cache_root / "tokens",
        cache_root / "song_embeddings",
        args.extraction_batch_size,
        stage1_signature,
    )
    validation_embedding_records, validation_embedding_failures = build_song_embedding_cache(
        manifest["validation"],
        adapter,
        device,
        cache_root / "tokens",
        cache_root / "song_embeddings",
        args.extraction_batch_size,
        stage1_signature,
    )

    train_records, train_graph_failures = build_graph_song_cache(
        train_embedding_records,
        cache_root / "relation_graphs",
        args.recurrence_threshold,
        args.graph_top_k,
        stage1_signature,
    )
    validation_records, validation_graph_failures = build_graph_song_cache(
        validation_embedding_records,
        cache_root / "relation_graphs",
        args.recurrence_threshold,
        args.graph_top_k,
        stage1_signature,
    )

    failures = (
        train_embedding_failures
        + validation_embedding_failures
        + train_graph_failures
        + validation_graph_failures
    )
    if failures:
        save_json(args.output_dir / "graph_midi_preprocess_failures.json", failures)
        print(f"Skipped malformed/unusable MIDI files: {len(failures)}")

    del adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_dataset = GraphSongEmbeddingDataset(train_records)
    validation_dataset = GraphSongEmbeddingDataset(validation_records)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=balanced_sampler(train_dataset.labels, args.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_graph_song_embeddings,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_graph_song_embeddings,
    )

    model = GraphAugmentedMIDIFST(
        input_dim=MUSICBERT_BASE_DIM,
        graph_relation_dim=args.graph_relation_dim,
        graph_heads=args.graph_heads,
        graph_dropout=args.graph_dropout,
        fst_hidden_dim=256,
        fst_num_heads=8,
        fst_num_layers=4,
        fst_dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    checkpoint_path = args.output_dir / "graph_midi_fst_stage2_best.pt"
    monitor = TrainingMonitor(args.output_dir, "stage2_graph")
    best_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = graph_stage2_epoch(
            model, train_loader, device, optimizer, scaler, args.amp
        )
        validation_metrics = graph_stage2_epoch(
            model, validation_loader, device, None, scaler, args.amp
        )
        print(
            f"Graph Stage 2 epoch {epoch:03d} | "
            f"train loss {train_metrics.loss:.4f}, "
            f"bal-acc {train_metrics.balanced_accuracy:.4f} | "
            f"val loss {validation_metrics.loss:.4f}, "
            f"bal-acc {validation_metrics.balanced_accuracy:.4f}",
            flush=True,
        )
        monitor.record(epoch, train_metrics, validation_metrics)

        if validation_metrics.loss < best_loss:
            best_loss = validation_metrics.loss
            stale_epochs = 0
            torch.save(
                {
                    "format": "graph-midi-fst-stage2-v1",
                    "model_state": model.state_dict(),
                    "input_dim": MUSICBERT_BASE_DIM,
                    "graph_relations": list(GRAPH_RELATIONS),
                    "graph_type": GRAPH_ARCHITECTURE,
                    "graph_relation_dim": int(args.graph_relation_dim),
                    "graph_heads": int(args.graph_heads),
                    "graph_dropout": float(args.graph_dropout),
                    "graph_top_k": int(args.graph_top_k),
                    "recurrence_threshold": float(args.recurrence_threshold),
                    "fst_hidden_dim": 256,
                    "fst_num_heads": 8,
                    "fst_num_layers": 4,
                    "fst_dropout": 0.1,
                    "max_segments": MAX_SEGMENTS,
                    "threshold": 0.5,
                    "stage1_signature": stage1_signature,
                    "best_validation_loss": best_loss,
                    "split_manifest": str(args.split_manifest.resolve()),
                    "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Graph Stage 2 early stopping after {epoch} epochs")
                break

    monitor.close()
    print("Graph Stage 2 training finished.")
    print(f"Best checkpoint: {checkpoint_path.resolve()}")
    print(f"Training log: {monitor.log_path.resolve()}")
    print(
        "TensorBoard log dir: "
        f"{(args.output_dir / 'tensorboard' / 'stage2_graph').resolve()}"
    )
    print(f"Loss curve: {monitor.plot_path.resolve()}")


def extract_embeddings_and_graph_for_one_midi(
    midi_path: Path,
    adapter: MusicBERTAdapter,
    device: torch.device,
    extraction_batch_size: int,
    recurrence_threshold: float,
    graph_top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
    encoding = midi_to_octuple_encoding(midi_path)
    all_segments = split_into_four_bar_segments(encoding)
    segments = all_segments[:MAX_SEGMENTS]
    if not segments:
        raise ValueError("The MIDI file produced no usable four-bar segments")

    segment_start_bars = four_bar_segment_start_bars(encoding)[:MAX_SEGMENTS]
    if len(segment_start_bars) != len(segments):
        raise ValueError(
            "MusicBERT segment/start-bar mismatch during inference: "
            f"segments={len(segments)}, starts={len(segment_start_bars)}"
        )

    token_sequences = [adapter.encode_segment(segment) for segment in segments]
    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(token_sequences), extraction_batch_size):
            batch = token_sequences[start : start + extraction_batch_size]
            maximum = max(sequence.numel() for sequence in batch)
            padded = torch.full((len(batch), maximum), adapter.pad_id, dtype=torch.long)
            for row, sequence in enumerate(batch):
                padded[row, : sequence.numel()] = sequence
            embeddings.append(adapter(padded.to(device)).float().cpu())

    song_embeddings = torch.cat(embeddings, dim=0)
    graph = build_relation_graph(
        midi_path,
        recurrence_threshold=recurrence_threshold,
        top_k=graph_top_k,
    )
    adjacency = align_relation_graph_to_musicbert_segments(graph, segment_start_bars)
    return song_embeddings, adjacency, len(segments), len(all_segments) > MAX_SEGMENTS


def _predict_one_midi(
    midi_path: Path,
    adapter: MusicBERTAdapter,
    model: GraphAugmentedMIDIFST,
    device: torch.device,
    extraction_batch_size: int,
    recurrence_threshold: float,
    graph_top_k: int,
    threshold: float,
) -> dict[str, Any]:
    song_embeddings, adjacency, length, truncated = extract_embeddings_and_graph_for_one_midi(
        midi_path,
        adapter,
        device,
        extraction_batch_size,
        recurrence_threshold,
        graph_top_k,
    )

    padded_embeddings = torch.zeros(
        (1, MAX_SEGMENTS, MUSICBERT_BASE_DIM), dtype=torch.float32, device=device
    )
    padded_adjacency = torch.zeros(
        (1, len(GRAPH_RELATIONS), MAX_SEGMENTS, MAX_SEGMENTS),
        dtype=torch.float32,
        device=device,
    )
    padding_mask = torch.ones((1, MAX_SEGMENTS), dtype=torch.bool, device=device)
    padded_embeddings[0, :length] = song_embeddings.to(device)
    padded_adjacency[0, :, :length, :length] = adjacency.to(device)
    padding_mask[0, :length] = False

    with torch.no_grad():
        logit = model(padded_embeddings, padded_adjacency, padding_mask).squeeze()
        ai_probability = float(torch.sigmoid(logit).cpu())

    prediction_id = 1 if ai_probability >= threshold else 0
    return {
        "path": midi_path,
        "prediction_id": prediction_id,
        "prediction": "AI" if prediction_id == 1 else "Human",
        "ai_probability": ai_probability,
        "human_probability": 1.0 - ai_probability,
        "segments_used": length,
        "truncated": truncated,
    }


def command_infer(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    stage1_signature = file_signature(args.stage1_checkpoint)

    adapter = MusicBERTAdapter(
        args.muzic_dir,
        args.musicbert_checkpoint,
        args.musicbert_data_bin,
        args.stage1_checkpoint,
    ).to(device)
    adapter.freeze_for_stage2()

    stage2_payload = torch.load(
        args.stage2_checkpoint, map_location="cpu", weights_only=False
    )
    if stage2_payload.get("format") != "graph-midi-fst-stage2-v1":
        raise ValueError(f"Unsupported Graph Stage 2 checkpoint: {args.stage2_checkpoint}")
    checkpoint_graph_type = stage2_payload.get("graph_type")
    if checkpoint_graph_type != GRAPH_ARCHITECTURE:
        raise ValueError(
            f"This script expects a {GRAPH_ARCHITECTURE} checkpoint, but got " 
            f"{checkpoint_graph_type!r}. Use the matching GAT/GCN/R-GCN script."
        )
    if stage2_payload.get("stage1_signature") != stage1_signature:
        raise ValueError(
            "Graph Stage 2 was trained with a different Stage 1 checkpoint. "
            "Use the matching --stage1-checkpoint."
        )

    model = GraphAugmentedMIDIFST(
        input_dim=int(stage2_payload["input_dim"]),
        graph_relation_dim=int(stage2_payload["graph_relation_dim"]),
        graph_heads=int(stage2_payload["graph_heads"]),
        graph_dropout=float(stage2_payload["graph_dropout"]),
        fst_hidden_dim=int(stage2_payload["fst_hidden_dim"]),
        fst_num_heads=int(stage2_payload["fst_num_heads"]),
        fst_num_layers=int(stage2_payload["fst_num_layers"]),
        fst_dropout=float(stage2_payload["fst_dropout"]),
    ).to(device)
    model.load_state_dict(stage2_payload["model_state"])
    model.eval()

    recurrence_threshold = float(stage2_payload["recurrence_threshold"])
    graph_top_k = int(stage2_payload["graph_top_k"])
    threshold = float(stage2_payload.get("threshold", 0.5))

    if args.midi is not None:
        result = _predict_one_midi(
            args.midi,
            adapter,
            model,
            device,
            args.extraction_batch_size,
            recurrence_threshold,
            graph_top_k,
            threshold,
        )
        print(f"MIDI: {result['path'].resolve()}")
        print(f"Prediction: {result['prediction']}")
        print(f"AI probability: {result['ai_probability']:.6f}")
        print(f"Human probability: {result['human_probability']:.6f}")
        print(f"Four-bar segments used: {result['segments_used']}")
        if result["truncated"]:
            print(f"Note: only the first {MAX_SEGMENTS} segments were used")
        return

    if args.label is None:
        raise ValueError("--label is required when using --midi-dir")

    label_id = 1 if args.label == "ai" else 0
    label_name = "AI" if label_id == 1 else "Human"
    midi_paths = discover_midis(args.midi_dir)
    total_found = len(midi_paths)

    excluded_count = 0
    if args.exclude_split is not None:
        excluded_manifest = load_manifest(args.exclude_split)
        excluded_paths = {
            Path(item["path"]).resolve()
            for split_name in ("train", "validation")
            for item in excluded_manifest[split_name]
        }
        filtered_paths = [path for path in midi_paths if path.resolve() not in excluded_paths]
        excluded_count = len(midi_paths) - len(filtered_paths)
        midi_paths = filtered_paths

    if args.count is not None:
        if args.count < 1:
            raise ValueError("--count must be at least 1")
        if len(midi_paths) < args.count:
            raise ValueError(
                f"Requested --count {args.count}, but only {len(midi_paths)} MIDI files "
                "remain after exclusion"
            )

    correct = 0
    evaluated = 0
    failures: list[dict[str, str]] = []

    print(f"Folder: {args.midi_dir.resolve()}")
    print(f"Ground-truth label: {label_name}")
    print(f"MIDI files found: {total_found}")
    if args.exclude_split is not None:
        print(f"Excluded by split manifest: {excluded_count}")
        print(f"Unseen MIDI files available: {total_found - excluded_count}")
    if args.count is not None:
        print(f"Target evaluated MIDI files: {args.count}")

    for index, midi_path in enumerate(midi_paths, start=1):
        if args.count is not None and evaluated >= args.count:
            break
        try:
            result = _predict_one_midi(
                midi_path,
                adapter,
                model,
                device,
                args.extraction_batch_size,
                recurrence_threshold,
                graph_top_k,
                threshold,
            )
            is_correct = result["prediction_id"] == label_id
            correct += int(is_correct)
            evaluated += 1
            print(
                f"[{index}/{len(midi_paths)}] {midi_path.name} | "
                f"Prediction: {result['prediction']} | "
                f"AI: {result['ai_probability']:.6f} | "
                f"Human: {result['human_probability']:.6f} | "
                f"{'CORRECT' if is_correct else 'WRONG'}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"path": str(midi_path), "error": str(exc)})
            print(
                f"[{index}/{len(midi_paths)}] {midi_path.name} | ERROR: {exc}",
                flush=True,
            )

    if evaluated == 0:
        raise RuntimeError("No MIDI files could be evaluated")
    if args.count is not None and evaluated < args.count:
        raise RuntimeError(
            f"Requested {args.count} evaluated MIDI files, but only {evaluated} could be "
            f"evaluated after trying {len(midi_paths)} unseen candidates"
        )

    accuracy = correct / evaluated
    print("")
    print("Folder evaluation finished.")
    print(f"Evaluated: {evaluated}")
    print(f"Correct: {correct}")
    print(f"Wrong: {evaluated - correct}")
    print(f"Accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)")
    print(f"Failed/skipped: {len(failures)}")

    result_file = args.result_file or (args.stage2_checkpoint.parent / "infer_results.txt")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("a", encoding="utf-8") as stream:
        stream.write(
            f"Label: {args.label}\n"
            f"Evaluated: {evaluated}\n"
            f"Correct: {correct}\n"
            f"Accuracy: {accuracy:.6f}\n"
            f"Failed: {len(failures)}\n\n"
        )
    print(f"Result summary saved: {result_file.resolve()}")


def add_musicbert_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--muzic-dir",
        type=Path,
        required=True,
        help="Path to a clone of microsoft/muzic",
    )
    parser.add_argument(
        "--musicbert-checkpoint",
        type=Path,
        required=True,
        help="Official MusicBERT-base .pt checkpoint",
    )
    parser.add_argument(
        "--musicbert-data-bin",
        type=Path,
        required=True,
        help="Directory containing MusicBERT dict.txt",
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--extraction-batch-size",
        type=int,
        default=2,
        help="MusicBERT segment batch size; lower this if GPU memory is insufficient",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"MusicBERT + {GRAPH_DISPLAY_NAME} + Fusion Segment Transformer for MIDI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dictionary = subparsers.add_parser(
        "make-dictionary", help="Create the official deterministic MusicBERT dictionary"
    )
    dictionary.add_argument("--output-dir", type=Path, required=True)
    dictionary.set_defaults(function=command_make_dictionary)

    split = subparsers.add_parser(
        "make-split", help="Split whole real/fake MIDI files before four-bar segmentation"
    )
    split.add_argument(
        "--real-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing real MIDI files",
    )
    split.add_argument(
        "--fake-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing fake MIDI files",
    )
    split.add_argument("--output", type=Path, required=True)
    split.add_argument(
        "--per-class-count",
        type=int,
        help=(
            "Legacy mode: number of usable MIDI songs to select from each class "
            "after combining all directories"
        ),
    )
    split.add_argument(
        "--real-counts",
        type=int,
        nargs="+",
        help="Per-directory counts matching --real-dir order",
    )
    split.add_argument(
        "--fake-counts",
        type=int,
        nargs="+",
        help="Per-directory counts matching --fake-dir order",
    )
    split.add_argument("--val-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(function=command_make_split)

    stage1 = subparsers.add_parser(
        "train-stage1", help="Fine-tune the last six MusicBERT-base layers on four-bar segments"
    )
    add_musicbert_arguments(stage1)
    stage1.add_argument("--split-manifest", type=Path, required=True)
    stage1.add_argument("--output-dir", type=Path, required=True)
    stage1.add_argument("--cache-dir", type=Path)
    stage1.add_argument("--epochs", type=int, default=50)
    stage1.add_argument("--patience", type=int, default=10)
    stage1.add_argument("--batch-size", type=int, default=2)
    stage1.add_argument("--grad-accum", type=int, default=32)
    stage1.add_argument("--learning-rate", type=float, default=5e-6)
    stage1.add_argument("--seed", type=int, default=42)
    stage1.add_argument("--device", default="auto")
    stage1.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    stage1.set_defaults(function=command_train_stage1)

    stage2 = subparsers.add_parser(
        "train-stage2",
        help=f"Freeze MusicBERT and train {GRAPH_DISPLAY_NAME} + FST on songs",
    )
    add_musicbert_arguments(stage2)
    add_runtime_arguments(stage2)
    stage2.add_argument("--stage1-checkpoint", type=Path, required=True)
    stage2.add_argument("--split-manifest", type=Path, required=True)
    stage2.add_argument("--output-dir", type=Path, required=True)
    stage2.add_argument("--cache-dir", type=Path)
    stage2.add_argument("--epochs", type=int, default=200)
    stage2.add_argument("--patience", type=int, default=20)
    stage2.add_argument("--batch-size", type=int, default=8)
    stage2.add_argument("--learning-rate", type=float, default=1e-4)
    stage2.add_argument("--seed", type=int, default=42)
    stage2.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    stage2.add_argument("--recurrence-threshold", type=float, default=0.85)
    stage2.add_argument("--graph-top-k", type=int, default=4)
    stage2.add_argument("--graph-relation-dim", type=int, default=256)
    stage2.add_argument("--graph-heads", type=int, default=4)
    stage2.add_argument("--graph-dropout", type=float, default=0.1)
    stage2.set_defaults(function=command_train_stage2)

    infer = subparsers.add_parser(
        "infer",
        help="Predict one MIDI file or evaluate every MIDI file in a folder",
    )
    add_musicbert_arguments(infer)
    add_runtime_arguments(infer)
    infer.add_argument("--stage1-checkpoint", type=Path, required=True)
    infer.add_argument("--stage2-checkpoint", type=Path, required=True)
    infer_mode = infer.add_mutually_exclusive_group(required=True)
    infer_mode.add_argument(
        "--midi",
        type=Path,
        help="Mode 1: predict one MIDI file",
    )
    infer_mode.add_argument(
        "--midi-dir",
        type=Path,
        help="Mode 2: predict all .mid/.midi files under this folder",
    )
    infer.add_argument(
        "--label",
        choices=("human", "ai"),
        help="Ground-truth label for --midi-dir mode; required for folder accuracy",
    )
    infer.add_argument(
        "--exclude-split",
        type=Path,
        help=(
            "For --midi-dir mode, exclude every MIDI path listed in the train and "
            "validation sections of this split manifest"
        ),
    )
    infer.add_argument(
        "--count",
        type=int,
        help="For --midi-dir mode, evaluate only this many remaining MIDI files",
    )
    infer.add_argument(
        "--result-file",
        type=Path,
        help=(
            "Append the folder-evaluation summary to this text file. "
            "Default: infer_results.txt next to the Stage 2 checkpoint"
        ),
    )
    infer.set_defaults(function=command_infer)
    return parser


def main() -> None:
    print(f"Graph MIDI-FST build: {MIDI_FST_BUILD}", flush=True)
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.function(args)
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user") from None
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
