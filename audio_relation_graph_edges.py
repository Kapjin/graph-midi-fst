#!/usr/bin/env python3
"""
audio_relation_graph_edges.py

Build relation graphs directly from audio for Graph MuQ-FST.

Graph node
----------
One node = one non-overlapping 30-second audio segment, matching MuQ-FST.

Relations
---------
rhythm:
    Similarity of onset-strength patterns.

melody:
    Similarity of transposition-invariant dominant-chroma interval histograms.

harmony:
    Similarity of segment-level chroma statistics.

structure:
    Non-adjacent recurrence of a combined audio descriptor
    (MFCC statistics + chroma + onset pattern), thresholded by cosine similarity.

The returned graph contains four [T, T] weighted adjacency matrices:
    rhythm, melody, harmony, structure

This is the audio-domain counterpart of midi_relation_graph_edges.py.  It keeps
the same relation-specific graph interface while replacing MIDI-only symbolic
features with audio-derived features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


EPS = 1e-12
GRAPH_RELATIONS = ("rhythm", "melody", "harmony", "structure")


def _import_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "Audio relation graph construction requires librosa. "
            "Install it with: pip install librosa"
        ) from exc
    return librosa


def _cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got shape {x.shape}")
    if len(x) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    norm = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = x / np.maximum(norm, EPS)
    similarity = normalized @ normalized.T

    # Relation weights used by GAT are non-negative.
    return np.clip(similarity, 0.0, 1.0).astype(np.float32)


def _resample_vector(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if bins < 1:
        raise ValueError("bins must be positive")
    if len(values) == 0:
        return np.zeros(bins, dtype=np.float32)
    if len(values) == 1:
        return np.full(bins, float(values[0]), dtype=np.float32)

    source_x = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, bins, dtype=np.float32)
    return np.interp(target_x, source_x, values).astype(np.float32)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _topk_symmetric(
    score: np.ndarray,
    top_k: int,
    min_score: float = 0.0,
) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError(f"Expected square score matrix, got {score.shape}")

    n = score.shape[0]
    out = np.zeros_like(score, dtype=np.float32)

    if top_k <= 0:
        out = np.where(score > min_score, score, 0.0).astype(np.float32)
        np.fill_diagonal(out, 0.0)
        return np.maximum(out, out.T).astype(np.float32)

    for i in range(n):
        candidates = [
            (float(score[i, j]), j)
            for j in range(n)
            if j != i and float(score[i, j]) > min_score
        ]
        candidates.sort(key=lambda item: item[0], reverse=True)
        for value, j in candidates[:top_k]:
            out[i, j] = value

    out = np.maximum(out, out.T)
    np.fill_diagonal(out, 0.0)
    return out.astype(np.float32)


def _segment_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    segment_seconds: float,
    max_segments: int,
) -> list[np.ndarray]:
    segment_samples = int(round(sample_rate * segment_seconds))
    if segment_samples < 1:
        raise ValueError("segment_seconds produces an empty segment")

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise ValueError("Audio contains no samples")

    segments = [
        waveform[start : start + segment_samples]
        for start in range(0, len(waveform), segment_samples)
        if waveform[start : start + segment_samples].size > 0
    ]
    return segments[:max_segments]


def _dominant_chroma_interval_histogram(chroma: np.ndarray) -> np.ndarray:
    """
    Build a 12-bin pitch-class interval histogram from dominant chroma frames.

    Because only pitch-class differences are retained, the representation is
    invariant to global transposition.
    """
    chroma = np.asarray(chroma, dtype=np.float32)
    if chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.shape[1] < 2:
        return np.zeros(12, dtype=np.float32)

    frame_energy = chroma.sum(axis=0)
    if not np.any(frame_energy > EPS):
        return np.zeros(12, dtype=np.float32)

    threshold = max(float(np.median(frame_energy)) * 0.25, EPS)
    valid = frame_energy > threshold
    dominant = np.argmax(chroma[:, valid], axis=0)
    if dominant.size < 2:
        return np.zeros(12, dtype=np.float32)

    intervals = (np.diff(dominant) % 12).astype(int)
    histogram = np.bincount(intervals, minlength=12).astype(np.float32)
    return _l2_normalize(histogram)


def _segment_features(
    segment: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    librosa = _import_librosa()

    segment = np.asarray(segment, dtype=np.float32)
    if segment.size == 0:
        raise ValueError("Empty audio segment")

    # Avoid numerical problems on completely silent segments while preserving
    # the segment count used by MuQ-FST.
    if float(np.max(np.abs(segment))) <= EPS:
        return (
            np.zeros(64, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
            np.zeros(24, dtype=np.float32),
            np.zeros(67, dtype=np.float32),
        )

    hop_length = 512

    onset = librosa.onset.onset_strength(
        y=segment,
        sr=sample_rate,
        hop_length=hop_length,
    )
    rhythm = _l2_normalize(_resample_vector(onset, 64))

    chroma = librosa.feature.chroma_stft(
        y=segment,
        sr=sample_rate,
        n_fft=2048,
        hop_length=hop_length,
    ).astype(np.float32)

    melody = _dominant_chroma_interval_histogram(chroma)

    chroma_mean = np.nan_to_num(chroma.mean(axis=1), nan=0.0).astype(np.float32)
    chroma_std = np.nan_to_num(chroma.std(axis=1), nan=0.0).astype(np.float32)
    harmony = _l2_normalize(np.concatenate([chroma_mean, chroma_std]))

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=sample_rate,
        n_mfcc=13,
        n_fft=2048,
        hop_length=hop_length,
    ).astype(np.float32)
    mfcc_mean = np.nan_to_num(mfcc.mean(axis=1), nan=0.0).astype(np.float32)
    mfcc_std = np.nan_to_num(mfcc.std(axis=1), nan=0.0).astype(np.float32)

    structure_onset = _resample_vector(onset, 16)
    structure = _l2_normalize(
        np.concatenate(
            [
                mfcc_mean,          # 13
                mfcc_std,           # 13
                chroma_mean,        # 12
                chroma_std,         # 12
                structure_onset,    # 16
                np.asarray([float(np.sqrt(np.mean(segment**2) + EPS))], dtype=np.float32),
            ]
        )
    )
    # 13 + 13 + 12 + 12 + 16 + 1 = 67
    return rhythm, melody, harmony, structure


def build_audio_relation_graph(
    audio_path: Path,
    sample_rate: int = 24_000,
    segment_seconds: float = 30.0,
    recurrence_threshold: float = 0.85,
    top_k: int = 4,
    max_segments: int = 48,
) -> dict[str, np.ndarray]:
    """
    Construct rhythm/melody/harmony/structure relations aligned one-to-one
    with MuQ-FST 30-second segments.
    """
    librosa = _import_librosa()
    waveform, _ = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True,
    )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError("Audio contains no samples")

    segments = _segment_waveform(
        waveform,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        max_segments=max_segments,
    )
    if not segments:
        raise ValueError("Audio produced no graph nodes")

    rhythm_features: list[np.ndarray] = []
    melody_features: list[np.ndarray] = []
    harmony_features: list[np.ndarray] = []
    structure_features: list[np.ndarray] = []

    for segment in segments:
        rhythm, melody, harmony, structure = _segment_features(
            segment,
            sample_rate,
        )
        rhythm_features.append(rhythm)
        melody_features.append(melody)
        harmony_features.append(harmony)
        structure_features.append(structure)

    rhythm_score = _cosine_matrix(np.vstack(rhythm_features))
    melody_score = _cosine_matrix(np.vstack(melody_features))
    harmony_score = _cosine_matrix(np.vstack(harmony_features))
    structure_similarity = _cosine_matrix(np.vstack(structure_features))

    rhythm = _topk_symmetric(rhythm_score, top_k=top_k, min_score=0.0)
    melody = _topk_symmetric(melody_score, top_k=top_k, min_score=0.0)
    harmony = _topk_symmetric(harmony_score, top_k=top_k, min_score=0.0)

    # Preserve the MIDI graph's non-adjacent structural-recurrence idea:
    # only segments at least two nodes apart and above the recurrence threshold.
    structure_score = np.zeros_like(structure_similarity, dtype=np.float32)
    n = structure_similarity.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) < 2:
                continue
            value = float(structure_similarity[i, j])
            if value >= recurrence_threshold:
                structure_score[i, j] = value
                structure_score[j, i] = value

    structure = _topk_symmetric(
        structure_score,
        top_k=top_k,
        min_score=0.0,
    )

    combined = (rhythm + melody + harmony + structure).astype(np.float32)

    return {
        "rhythm": rhythm,
        "melody": melody,
        "harmony": harmony,
        "structure": structure,
        "combined": combined,
        "segment_index": np.arange(len(segments), dtype=np.int32),
        "segment_start_seconds": (
            np.arange(len(segments), dtype=np.float32) * float(segment_seconds)
        ),
    }
