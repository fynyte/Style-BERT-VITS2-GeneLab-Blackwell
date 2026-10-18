"""WORLD targets aligned with the padded, center=False training spectrogram."""
import math
import os
from pathlib import Path
import tempfile

import numpy as np


def extract_prosody(wav, sampling_rate, hop_length, filter_length, frame_count):
    import pyworld

    audio = np.asarray(wav, dtype=np.float64).reshape(-1)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN/Inf")
    period = hop_length / sampling_rate * 1000
    f0, time = pyworld.dio(audio, sampling_rate, frame_period=period,
                         f0_floor=50.0, f0_ceil=1100.0)
    f0 = pyworld.stonemask(audio, f0, time, sampling_rate)
    # Padded STFT frame centers are (k + 0.5) * hop, not k * hop.
    centers = (np.arange(frame_count) + 0.5) * hop_length / sampling_rate
    nearest = np.clip(np.searchsorted(time, centers), 1, len(time) - 1)
    nearest -= (np.abs(time[nearest - 1] - centers) < np.abs(time[nearest] - centers))
    voiced = (f0[nearest] > 0).astype(np.float32)
    valid = f0 > 0
    pitch = np.zeros(frame_count, dtype=np.float32)
    if valid.any():
        pitch = ((np.interp(centers, time[valid], np.log(f0[valid])) - math.log(180)) / 0.5).astype(np.float32)
        pitch *= voiced
    padding = (filter_length - hop_length) // 2
    padded = np.pad(audio, (padding, padding), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, filter_length)[::hop_length][:frame_count]
    if len(frames) != frame_count:
        raise ValueError("Prosody/spectrogram frame count mismatch")
    energy = (np.log(np.sqrt(np.mean(frames ** 2, axis=-1)) + 1e-5) + 4.0) / 2.0
    return np.stack((pitch, energy, voiced)).astype(np.float32)


def load_or_extract_prosody(filename, wav, sampling_rate, hop_length, filter_length, frame_count):
    source = Path(filename)
    cache = Path(str(source) + ".prosody.npz")
    metadata = np.array([2, sampling_rate, hop_length, filter_length, frame_count,
                         source.stat().st_size, source.stat().st_mtime_ns], dtype=np.int64)
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as data:
                features = data["features"]
                if (np.array_equal(data["metadata"], metadata)
                        and features.shape == (3, frame_count) and np.isfinite(features).all()):
                    return features.copy()
        except (ValueError, OSError, KeyError):
            pass
    features = extract_prosody(wav, sampling_rate, hop_length, filter_length, frame_count)
    # DDP ranks/workers can create the same cache; atomic replacement avoids
    # reading a partial ZIP. Failure is explicit (a read-only dataset is fixable).
    fd, temporary = tempfile.mkstemp(prefix=cache.name, suffix=".tmp", dir=cache.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, features=features, metadata=metadata)
        os.replace(temporary, cache)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return features
