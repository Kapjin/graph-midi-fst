"""Train and run MuQ-FST on full audio tracks.

This is the audio counterpart of the MIDI-FST training script:

  make-split       Select real/fake audio files (optionally per-directory counts)
                   and split whole tracks into train/validation sets.
  train-stage1     Split tracks into non-overlapping 30-second segments,
                   fine-tune the last six MuQ encoder layers, and train a
                   temporary segment classifier.
  train-stage2     Freeze the Stage-1 MuQ encoder, extract one embedding per
                   30-second segment, and train FusionSegmentTransformer on
                   the resulting full-song segment sequences.
  infer            Classify one audio file, or evaluate a labeled audio folder.

Stage 1 follows the MuQ-FST architecture described in HAIM: audio is resampled
at 24 kHz, divided into non-overlapping 30 s segments, and the last six MuQ
encoder layers are fine-tuned. Stage 2 removes any separate beat branch and
uses the segment embeddings as the input sequence to FusionSegmentTransformer.

Training commands do not automatically start another stage or run inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError as exc:
    raise SystemExit(
        "Missing MuQ-FST dependencies. Activate the project environment and "
        "install torch/numpy first."
    ) from exc

from MIDI_FST.models.model import FusionSegmentTransformer


MUQ_FST_BUILD = "2026-08-26-muq-audio-r1"

# HAIM MuQ-FST settings.
SAMPLE_RATE = 24_000
SEGMENT_SECONDS = 30.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)
MUQ_DIM = 1024
MUQ_ENCODER_LAYERS = 12
MUQ_TRAINABLE_LAYERS = 6
MAX_SEGMENTS = 48
DEFAULT_MUQ_MODEL = "OpenMuQ/MuQ-large-msd-iter"

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
}


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


def file_signature(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(path: Path, extra: str) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{extra}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def discover_audio(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Audio directory does not exist: {directory}")
    paths = sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported audio files found under: {directory}")
    return paths


def _import_audio_libs():
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "Audio loading requires librosa. Install MuQ with: pip install muq"
        ) from exc
    return librosa


def audio_duration_seconds(path: Path) -> float:
    """Read duration without intentionally decoding the full waveform."""
    librosa = _import_audio_libs()
    try:
        # librosa uses ``filename`` for file-based duration lookup across the
        # versions commonly installed with MuQ.
        duration = float(librosa.get_duration(filename=str(path)))
    except Exception as exc:
        raise ValueError(f"Could not read audio duration: {exc}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid audio duration: {duration}")
    return duration


def load_audio_segment(path: Path, start_seconds: float, duration_seconds: float) -> torch.Tensor:
    """Load one mono segment at MuQ's required 24 kHz sample rate."""
    librosa = _import_audio_libs()
    try:
        wav, _ = librosa.load(
            str(path),
            sr=SAMPLE_RATE,
            mono=True,
            offset=float(start_seconds),
            duration=float(duration_seconds),
        )
    except Exception as exc:
        raise ValueError(f"Failed to load audio segment from {path}: {exc}") from exc

    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        raise ValueError("Loaded audio segment is empty")
    tensor = torch.from_numpy(wav)
    if tensor.numel() < SEGMENT_SAMPLES:
        tensor = F.pad(tensor, (0, SEGMENT_SAMPLES - tensor.numel()))
    elif tensor.numel() > SEGMENT_SAMPLES:
        tensor = tensor[:SEGMENT_SAMPLES]
    return tensor.contiguous()


def load_full_audio(path: Path) -> torch.Tensor:
    """Load an entire track as mono 24 kHz float32 audio."""
    librosa = _import_audio_libs()
    try:
        wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise ValueError(f"Failed to load audio file {path}: {exc}") from exc
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        raise ValueError("Audio file is empty")
    return torch.from_numpy(wav).contiguous()


def split_waveform_into_segments(waveform: torch.Tensor) -> list[torch.Tensor]:
    """Create non-overlapping 30 s segments and zero-pad the final partial segment."""
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform [time], got shape {tuple(waveform.shape)}")
    if waveform.numel() == 0:
        raise ValueError("Audio waveform is empty")

    segments: list[torch.Tensor] = []
    for start in range(0, waveform.numel(), SEGMENT_SAMPLES):
        segment = waveform[start : start + SEGMENT_SAMPLES]
        if segment.numel() < SEGMENT_SAMPLES:
            segment = F.pad(segment, (0, SEGMENT_SAMPLES - segment.numel()))
        segments.append(segment.contiguous())
    return segments


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format") != "muq-fst-split-v1":
        raise ValueError(f"Unsupported split manifest: {path}")
    for split_name in ("train", "validation"):
        if not manifest.get(split_name):
            raise ValueError(f"Manifest split is empty: {split_name}")
        for item in manifest[split_name]:
            audio_path = Path(item["path"])
            if not audio_path.is_file():
                raise FileNotFoundError(f"Manifest audio file is missing: {audio_path}")
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
                f"Requested {requested_count} audio files from {source}, but only "
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
                audio_duration_seconds(path)
                selected.append(path)
                if len(selected) == requested_count:
                    break
            except Exception as exc:
                rejected.append({"path": str(path), "source": source, "error": str(exc)})

        if len(selected) < requested_count:
            raise ValueError(
                f"Requested {requested_count} usable audio files from {source}, "
                f"but only {len(selected)} passed the duration/open check after "
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
            candidates = discover_audio(directory)
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
            candidates = discover_audio(directory)
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
            {path for directory in args.real_dir for path in discover_audio(directory)}
        )
        fake_candidates = sorted(
            {path for directory in args.fake_dir for path in discover_audio(directory)}
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
        raise ValueError("At least two selected audio files are required for each class")
    if len(set(real_paths)) != len(real_paths):
        raise ValueError("The real directories contain overlapping resolved audio paths")
    if len(set(fake_paths)) != len(fake_paths):
        raise ValueError("The fake directories contain overlapping resolved audio paths")

    overlap = set(real_paths) & set(fake_paths)
    if overlap:
        example = next(iter(overlap))
        raise ValueError(
            "The real and fake directories contain the same resolved audio path: "
            f"{example}"
        )

    def divide(
        paths: list[Path], label: int, source: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        shuffled = paths[:]
        rng.shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * args.val_ratio))
        if validation_count >= len(shuffled):
            raise ValueError(f"At least two {source} audio files are required")
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

    manifest = {
        "format": "muq-fst-split-v1",
        "created_at_unix": int(time.time()),
        "seed": args.seed,
        "validation_ratio": args.val_ratio,
        "sample_rate": SAMPLE_RATE,
        "segment_seconds": SEGMENT_SECONDS,
        "selection": {
            "method": (
                "seeded-random-per-directory-after-audio-duration-check"
                if use_per_directory_counts
                else "seeded-random-after-audio-duration-check"
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
            "selected_real": len(real_paths),
            "selected_fake": len(fake_paths),
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
    for name, count in manifest["counts"].items():
        print(f"  {name}: {count}")


class MuQAdapter(nn.Module):
    """Thin wrapper around the official OpenMuQ MuQ model."""

    def __init__(self, model_id: str, stage1_checkpoint: Path | None = None) -> None:
        super().__init__()
        try:
            from muq import MuQ
        except ImportError as exc:
            raise RuntimeError(
                "MuQ is not installed. Install the official package with: pip install muq"
            ) from exc

        self.model_id = model_id
        self.model = MuQ.from_pretrained(model_id)

        config = getattr(self.model, "config", None)
        encoder_dim = int(getattr(config, "encoder_dim", 0)) if config is not None else 0
        encoder_depth = int(getattr(config, "encoder_depth", 0)) if config is not None else 0
        if encoder_dim and encoder_dim != MUQ_DIM:
            raise ValueError(
                f"MuQ hidden dimension is {encoder_dim}; this script expects {MUQ_DIM}"
            )
        if encoder_depth and encoder_depth != MUQ_ENCODER_LAYERS:
            raise ValueError(
                f"MuQ encoder depth is {encoder_depth}; this script expects "
                f"{MUQ_ENCODER_LAYERS}"
            )

        if not hasattr(self.model, "model") or not hasattr(self.model.model, "conformer"):
            raise RuntimeError("Loaded MuQ model does not expose model.conformer")
        if not hasattr(self.model.model.conformer, "layers"):
            raise RuntimeError("Loaded MuQ conformer does not expose encoder layers")

        if stage1_checkpoint is not None:
            self.load_stage1_delta(stage1_checkpoint)

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.model.conformer.layers

    def configure_stage1(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if len(self.layers) < MUQ_TRAINABLE_LAYERS:
            raise ValueError(
                f"MuQ exposes only {len(self.layers)} encoder layers; cannot fine-tune "
                f"the last {MUQ_TRAINABLE_LAYERS}"
            )
        for layer in self.layers[-MUQ_TRAINABLE_LAYERS:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        self.model.train()

    def freeze_for_stage2(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # Official MuQ recommends fp32 inference; keep the backbone in fp32.
        outputs = self.model(waveforms.float(), output_hidden_states=True)
        hidden = outputs.last_hidden_state
        if hidden.ndim != 3:
            raise RuntimeError(f"Unexpected MuQ feature shape: {tuple(hidden.shape)}")
        # Same pooling behavior used by the FST repository for transformer features:
        # one vector per short segment.
        return hidden.mean(dim=1)

    def trainable_delta(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def load_stage1_delta(self, checkpoint_path: Path) -> dict[str, Any]:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "muq-fst-stage1-v1":
            raise ValueError(f"Unsupported Stage 1 checkpoint: {checkpoint_path}")
        if payload.get("muq_model_id") != self.model_id:
            raise ValueError(
                "Stage 1 was created from a different MuQ model. "
                "Use the same --muq-model for every command."
            )
        incompatible = self.model.load_state_dict(payload["muq_delta"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                f"Unexpected MuQ keys in Stage 1 checkpoint: {list(incompatible.unexpected_keys)}"
            )
        return payload


class Stage1Head(nn.Module):
    """Temporary binary segment classifier used only during Stage 1."""

    def __init__(self, input_dim: int = MUQ_DIM, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(input_dim, input_dim)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dropout(features)
        features = torch.tanh(self.dense(features))
        features = self.dropout(features)
        return self.output(features).squeeze(-1)


def build_segment_records(
    items: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    total = len(items)
    for position, item in enumerate(items, start=1):
        path = Path(item["path"])
        try:
            duration = audio_duration_seconds(path)
            segment_count = max(1, int(math.ceil(duration / SEGMENT_SECONDS)))
            for segment_index in range(segment_count):
                records.append(
                    {
                        "path": str(path),
                        "start_seconds": segment_index * SEGMENT_SECONDS,
                        "label": int(item["label"]),
                    }
                )
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})
        if position == total or position % 100 == 0:
            print(f"Segment index: {position}/{total} audio files", flush=True)
    if not records:
        raise RuntimeError("No usable 30-second audio segments were produced")
    return records, failures


class AudioSegmentDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = list(records)
        self.labels = [int(record["label"]) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        waveform = load_audio_segment(
            Path(record["path"]),
            float(record["start_seconds"]),
            SEGMENT_SECONDS,
        )
        return waveform, int(record["label"])


def collate_audio_segments(
    batch: Sequence[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    waveforms, labels = zip(*batch)
    return torch.stack(waveforms, dim=0), torch.tensor(labels, dtype=torch.float32)


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
            recalls.append(
                sum(int(predictions[index] == label) for index in indices) / len(indices)
            )
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
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled and device.type == "cuda",
    )


def stage1_epoch(
    adapter: MuQAdapter,
    head: Stage1Head,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_accum: int,
) -> EpochMetrics:
    training = optimizer is not None
    adapter.model.train(training)
    head.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for step, (waveforms, labels) in enumerate(loader, start=1):
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            # Keep MuQ in fp32. The official MuQ repository recommends fp32 to
            # avoid NaNs during feature extraction.
            features = adapter(waveforms)
            logits = head(features)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if training:
                (loss / grad_accum).backward()
                if step % grad_accum == 0 or step == len(loader):
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for group in optimizer.param_groups
                            for parameter in group["params"]
                        ],
                        0.5,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        predictions = (torch.sigmoid(logits) >= 0.5).long()
        total_loss += float(loss.detach()) * labels.numel()
        all_labels.extend(labels.long().detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    return metrics_from_outputs(total_loss, all_labels, all_predictions)


def command_train_stage1(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    manifest = load_manifest(args.split_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter = MuQAdapter(args.muq_model).to(device)
    adapter.configure_stage1()
    head = Stage1Head().to(device)

    train_records, train_failures = build_segment_records(manifest["train"])
    validation_records, validation_failures = build_segment_records(manifest["validation"])
    failures = train_failures + validation_failures
    if failures:
        save_json(args.output_dir / "audio_preprocess_failures.json", failures)
        print(f"Skipped malformed/unusable audio files: {len(failures)}")

    train_dataset = AudioSegmentDataset(train_records)
    validation_dataset = AudioSegmentDataset(validation_records)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=balanced_sampler(train_dataset.labels, args.seed),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_audio_segments,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_audio_segments,
    )

    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    parameters.extend(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)

    checkpoint_path = args.output_dir / "muq_stage1_best.pt"
    monitor = TrainingMonitor(args.output_dir, "stage1")
    best_loss = float("inf")
    stale_epochs = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = stage1_epoch(
                adapter,
                head,
                train_loader,
                device,
                optimizer,
                args.grad_accum,
            )
            validation_metrics = stage1_epoch(
                adapter,
                head,
                validation_loader,
                device,
                None,
                1,
            )
            print(
                f"Stage 1 epoch {epoch:03d} | "
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
                        "format": "muq-fst-stage1-v1",
                        "muq_model_id": args.muq_model,
                        "sample_rate": SAMPLE_RATE,
                        "segment_seconds": SEGMENT_SECONDS,
                        "embedding_dim": MUQ_DIM,
                        "trainable_last_layers": MUQ_TRAINABLE_LAYERS,
                        "muq_delta": adapter.trainable_delta(),
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
    finally:
        monitor.close()

    print("Stage 1 training finished.")
    print(f"Best checkpoint: {checkpoint_path.resolve()}")
    print(f"Training log: {monitor.log_path.resolve()}")
    print(f"TensorBoard log dir: {(args.output_dir / 'tensorboard' / 'stage1').resolve()}")
    print(f"Loss curve: {monitor.plot_path.resolve()}")


def build_song_embedding_cache(
    items: Sequence[dict[str, Any]],
    adapter: MuQAdapter,
    device: torch.device,
    embedding_cache_dir: Path,
    extraction_batch_size: int,
    stage1_signature: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    output_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for position, item in enumerate(items, start=1):
        source = Path(item["path"])
        key = cache_key(
            source,
            f"muq-stage1-30s-1024-v1|stage1={stage1_signature}",
        )
        output_path = embedding_cache_dir / f"{key}.pt"
        try:
            if not output_path.is_file():
                waveform = load_full_audio(source)
                segments = split_waveform_into_segments(waveform)[:MAX_SEGMENTS]
                embeddings: list[torch.Tensor] = []
                with torch.no_grad():
                    for start in range(0, len(segments), extraction_batch_size):
                        batch = torch.stack(
                            segments[start : start + extraction_batch_size], dim=0
                        ).to(device)
                        batch_embeddings = adapter(batch).float().cpu()
                        embeddings.append(batch_embeddings)
                song_embeddings = torch.cat(embeddings, dim=0)
                torch.save(
                    {
                        "format": "muq-fst-song-embedding-v1",
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
            print(f"Embedding cache: {position}/{len(items)} audio files", flush=True)

    if not output_records:
        raise RuntimeError("No song embeddings were produced")
    return output_records, failures


class SongEmbeddingDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = list(records)
        self.labels = [int(record["label"]) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        payload = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
        return payload["embeddings"], int(record["label"])


def collate_song_embeddings(
    batch: Sequence[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings, labels = zip(*batch)
    padded = torch.zeros((len(embeddings), MAX_SEGMENTS, MUQ_DIM), dtype=torch.float32)
    padding_mask = torch.ones((len(embeddings), MAX_SEGMENTS), dtype=torch.bool)
    for index, sequence in enumerate(embeddings):
        length = min(sequence.shape[0], MAX_SEGMENTS)
        padded[index, :length] = sequence[:length]
        padding_mask[index, :length] = False
    return padded, torch.tensor(labels, dtype=torch.float32), padding_mask


def stage2_epoch(
    model: FusionSegmentTransformer,
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

    for embeddings, labels, padding_mask in loader:
        embeddings = embeddings.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast_context(device, amp):
                logits = model(embeddings, padding_mask).squeeze(-1)
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

    adapter = MuQAdapter(args.muq_model, args.stage1_checkpoint).to(device)
    adapter.freeze_for_stage2()

    cache_root = args.cache_dir or args.output_dir / "cache"
    stage1_signature = file_signature(args.stage1_checkpoint)
    train_records, train_failures = build_song_embedding_cache(
        manifest["train"],
        adapter,
        device,
        cache_root / "song_embeddings",
        args.extraction_batch_size,
        stage1_signature,
    )
    validation_records, validation_failures = build_song_embedding_cache(
        manifest["validation"],
        adapter,
        device,
        cache_root / "song_embeddings",
        args.extraction_batch_size,
        stage1_signature,
    )
    failures = train_failures + validation_failures
    if failures:
        save_json(args.output_dir / "audio_preprocess_failures.json", failures)
        print(f"Skipped malformed/unusable audio files: {len(failures)}")

    # Stage 2 uses cached MuQ embeddings only.
    del adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_dataset = SongEmbeddingDataset(train_records)
    validation_dataset = SongEmbeddingDataset(validation_records)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=balanced_sampler(train_dataset.labels, args.seed),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_song_embeddings,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_song_embeddings,
    )

    model = FusionSegmentTransformer(
        input_dim=MUQ_DIM,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        max_sequence_length=1000,
        num_classes=2,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    checkpoint_path = args.output_dir / "muq_fst_stage2_best.pt"
    monitor = TrainingMonitor(args.output_dir, "stage2")
    best_loss = float("inf")
    stale_epochs = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = stage2_epoch(
                model, train_loader, device, optimizer, scaler, args.amp
            )
            validation_metrics = stage2_epoch(
                model, validation_loader, device, None, scaler, args.amp
            )
            print(
                f"Stage 2 epoch {epoch:03d} | "
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
                        "format": "muq-fst-stage2-v1",
                        "model_state": model.state_dict(),
                        "input_dim": MUQ_DIM,
                        "hidden_dim": 256,
                        "num_heads": 8,
                        "num_layers": 4,
                        "dropout": 0.1,
                        "max_segments": MAX_SEGMENTS,
                        "segment_seconds": SEGMENT_SECONDS,
                        "sample_rate": SAMPLE_RATE,
                        "threshold": 0.5,
                        "muq_model_id": args.muq_model,
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
                    print(f"Stage 2 early stopping after {epoch} epochs")
                    break
    finally:
        monitor.close()

    print("Stage 2 training finished.")
    print(f"Best checkpoint: {checkpoint_path.resolve()}")
    print(f"Training log: {monitor.log_path.resolve()}")
    print(f"TensorBoard log dir: {(args.output_dir / 'tensorboard' / 'stage2').resolve()}")
    print(f"Loss curve: {monitor.plot_path.resolve()}")


def _predict_one_audio(
    audio_path: Path,
    adapter: MuQAdapter,
    fst: FusionSegmentTransformer,
    device: torch.device,
    extraction_batch_size: int,
    threshold: float,
) -> dict[str, Any]:
    waveform = load_full_audio(audio_path)
    all_segments = split_waveform_into_segments(waveform)
    segments = all_segments[:MAX_SEGMENTS]
    if not segments:
        raise ValueError("The audio file produced no usable 30-second segments")

    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(segments), extraction_batch_size):
            batch = torch.stack(
                segments[start : start + extraction_batch_size], dim=0
            ).to(device)
            embeddings.append(adapter(batch).float())

    song_embeddings = torch.cat(embeddings, dim=0)
    padded_embeddings = torch.zeros(
        (1, MAX_SEGMENTS, MUQ_DIM), dtype=torch.float32, device=device
    )
    padding_mask = torch.ones((1, MAX_SEGMENTS), dtype=torch.bool, device=device)
    length = song_embeddings.shape[0]
    padded_embeddings[0, :length] = song_embeddings
    padding_mask[0, :length] = False

    with torch.no_grad():
        logit = fst(padded_embeddings, padding_mask).squeeze()
        ai_probability = float(torch.sigmoid(logit).cpu())

    prediction_id = 1 if ai_probability >= threshold else 0
    return {
        "path": audio_path,
        "prediction_id": prediction_id,
        "prediction": "AI" if prediction_id == 1 else "Human",
        "ai_probability": ai_probability,
        "human_probability": 1.0 - ai_probability,
        "segments_used": length,
        "truncated": len(all_segments) > MAX_SEGMENTS,
    }


def command_infer(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    stage1_signature = file_signature(args.stage1_checkpoint)

    adapter = MuQAdapter(args.muq_model, args.stage1_checkpoint).to(device)
    adapter.freeze_for_stage2()

    stage2_payload = torch.load(
        args.stage2_checkpoint, map_location="cpu", weights_only=False
    )
    if stage2_payload.get("format") != "muq-fst-stage2-v1":
        raise ValueError(f"Unsupported Stage 2 checkpoint: {args.stage2_checkpoint}")
    if stage2_payload.get("stage1_signature") != stage1_signature:
        raise ValueError(
            "Stage 2 was trained with a different Stage 1 checkpoint. "
            "Use the matching --stage1-checkpoint."
        )
    if stage2_payload.get("muq_model_id") != args.muq_model:
        raise ValueError(
            "Stage 2 was trained with a different MuQ model. "
            "Use the matching --muq-model."
        )

    fst = FusionSegmentTransformer(
        input_dim=int(stage2_payload["input_dim"]),
        hidden_dim=int(stage2_payload["hidden_dim"]),
        num_heads=int(stage2_payload["num_heads"]),
        num_layers=int(stage2_payload["num_layers"]),
        dropout=float(stage2_payload["dropout"]),
        max_sequence_length=1000,
        num_classes=2,
    ).to(device)
    fst.load_state_dict(stage2_payload["model_state"])
    fst.eval()

    threshold = float(stage2_payload.get("threshold", 0.5))

    if args.audio is not None:
        result = _predict_one_audio(
            args.audio,
            adapter,
            fst,
            device,
            args.extraction_batch_size,
            threshold,
        )
        print(f"Audio: {result['path'].resolve()}")
        print(f"Prediction: {result['prediction']}")
        print(f"AI probability: {result['ai_probability']:.6f}")
        print(f"Human probability: {result['human_probability']:.6f}")
        print(f"30-second segments used: {result['segments_used']}")
        if result["truncated"]:
            print(f"Note: only the first {MAX_SEGMENTS} segments were used")
        return

    if args.label is None:
        raise ValueError("--label is required when using --audio-dir")

    label_id = 1 if args.label == "ai" else 0
    label_name = "AI" if label_id == 1 else "Human"
    audio_paths = discover_audio(args.audio_dir)
    total_found = len(audio_paths)

    excluded_count = 0
    if args.exclude_split is not None:
        excluded_manifest = load_manifest(args.exclude_split)
        excluded_paths = {
            Path(item["path"]).resolve()
            for split_name in ("train", "validation")
            for item in excluded_manifest[split_name]
        }
        filtered_paths = [
            path for path in audio_paths if path.resolve() not in excluded_paths
        ]
        excluded_count = len(audio_paths) - len(filtered_paths)
        audio_paths = filtered_paths

    if args.count is not None:
        if args.count < 1:
            raise ValueError("--count must be at least 1")
        if len(audio_paths) < args.count:
            raise ValueError(
                f"Requested --count {args.count}, but only {len(audio_paths)} audio files "
                "remain after exclusion"
            )

    correct = 0
    evaluated = 0
    failures: list[dict[str, str]] = []

    print(f"Folder: {args.audio_dir.resolve()}")
    print(f"Ground-truth label: {label_name}")
    print(f"Audio files found: {total_found}")
    if args.exclude_split is not None:
        print(f"Excluded by split manifest: {excluded_count}")
        print(f"Unseen audio files available: {total_found - excluded_count}")
    if args.count is not None:
        print(f"Target evaluated audio files: {args.count}")

    for index, audio_path in enumerate(audio_paths, start=1):
        if args.count is not None and evaluated >= args.count:
            break
        try:
            result = _predict_one_audio(
                audio_path,
                adapter,
                fst,
                device,
                args.extraction_batch_size,
                threshold,
            )
            is_correct = result["prediction_id"] == label_id
            correct += int(is_correct)
            evaluated += 1
            print(
                f"[{index}/{len(audio_paths)}] {audio_path.name} | "
                f"Prediction: {result['prediction']} | "
                f"AI: {result['ai_probability']:.6f} | "
                f"Human: {result['human_probability']:.6f} | "
                f"{'CORRECT' if is_correct else 'WRONG'}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"path": str(audio_path), "error": str(exc)})
            print(
                f"[{index}/{len(audio_paths)}] {audio_path.name} | ERROR: {exc}",
                flush=True,
            )

    if evaluated == 0:
        raise RuntimeError("No audio files could be evaluated")
    if args.count is not None and evaluated < args.count:
        raise RuntimeError(
            f"Requested {args.count} evaluated audio files, but only {evaluated} could be evaluated "
            f"after trying {len(audio_paths)} unseen candidates"
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


def add_muq_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--muq-model",
        default=DEFAULT_MUQ_MODEL,
        help=(
            "MuQ Hugging Face model ID or local pretrained directory "
            f"(default: {DEFAULT_MUQ_MODEL})"
        ),
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--extraction-batch-size",
        type=int,
        default=2,
        help="MuQ 30-second segment batch size; lower this if GPU memory is insufficient",
    )
    parser.add_argument("--num-workers", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MuQ + Fusion Segment Transformer for full-audio AI music detection"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser(
        "make-split",
        help="Split whole real/fake audio files before 30-second segmentation",
    )
    split.add_argument(
        "--real-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing human/real audio files",
    )
    split.add_argument(
        "--fake-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing AI/fake audio files",
    )
    split.add_argument("--output", type=Path, required=True)
    split.add_argument(
        "--per-class-count",
        type=int,
        help=(
            "Legacy mode: number of usable audio tracks to select from each class "
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
        "train-stage1",
        help="Fine-tune the last six MuQ encoder layers on 30-second audio segments",
    )
    add_muq_argument(stage1)
    stage1.add_argument("--split-manifest", type=Path, required=True)
    stage1.add_argument("--output-dir", type=Path, required=True)
    # Kept for command compatibility with the MIDI script. Stage 1 does not
    # cache decoded waveforms; Stage 2 uses this directory for embeddings.
    stage1.add_argument("--cache-dir", type=Path)
    stage1.add_argument("--epochs", type=int, default=50)
    stage1.add_argument("--patience", type=int, default=50)
    stage1.add_argument("--batch-size", type=int, default=2)
    stage1.add_argument("--grad-accum", type=int, default=32)
    stage1.add_argument("--learning-rate", type=float, default=5e-6)
    stage1.add_argument("--seed", type=int, default=42)
    stage1.add_argument("--device", default="auto")
    stage1.add_argument("--num-workers", type=int, default=0)
    stage1.set_defaults(function=command_train_stage1)

    stage2 = subparsers.add_parser(
        "train-stage2",
        help="Freeze Stage-1 MuQ, extract song segment embeddings, and train FST",
    )
    add_muq_argument(stage2)
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
    stage2.set_defaults(function=command_train_stage2)

    infer = subparsers.add_parser(
        "infer",
        help="Predict one audio file or evaluate every supported audio file in a folder",
    )
    add_muq_argument(infer)
    add_runtime_arguments(infer)
    infer.add_argument("--stage1-checkpoint", type=Path, required=True)
    infer.add_argument("--stage2-checkpoint", type=Path, required=True)
    infer_mode = infer.add_mutually_exclusive_group(required=True)
    infer_mode.add_argument(
        "--audio",
        type=Path,
        help="Mode 1: predict one audio file",
    )
    infer_mode.add_argument(
        "--audio-dir",
        type=Path,
        help="Mode 2: predict all supported audio files under this folder",
    )
    infer.add_argument(
        "--label",
        choices=("human", "ai"),
        help="Ground-truth label for --audio-dir mode; required for folder accuracy",
    )
    infer.add_argument(
        "--exclude-split",
        type=Path,
        help=(
            "For --audio-dir mode, exclude every audio path listed in the train and "
            "validation sections of this split manifest"
        ),
    )
    infer.add_argument(
        "--count",
        type=int,
        help="For --audio-dir mode, evaluate only this many remaining audio files",
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
    print(f"MuQ-FST build: {MUQ_FST_BUILD}", flush=True)
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
