"""Audio loading for laptop-side inference.

Loads an audio file (any format ffmpeg can read) and returns a
mono float32 waveform at 16 kHz ready to feed the FastConformer
encoder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio_16k_mono(path: str | Path) -> np.ndarray:
    """Load audio, resample to 16 kHz mono, return float32 waveform.

    Uses ffmpeg for resampling because it handles a much wider range
    of input formats than soundfile alone.
    """
    import subprocess

    import soundfile as sf

    out_wav = Path("/tmp") / ("_bim_asr_" + Path(path).stem + "_16k.wav")
    if not out_wav.exists():
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(out_wav),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    wav, sr = sf.read(str(out_wav), dtype="float32")
    if sr != 16000:
        raise RuntimeError("ffmpeg did not produce 16 kHz output")
    return wav


def chunk_waveform(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16000,
    window_seconds: float = 30.0,
    overlap_seconds: float = 5.0,
) -> list[np.ndarray]:
    """Split a long waveform into overlapping windows.

    Default 30 s windows with 5 s overlap. This is the chunking that
    BimOmni's documented path used and what Canary-Qwen-2.5B was
    trained on.
    """
    win = int(window_seconds * sample_rate)
    hop = int((window_seconds - overlap_seconds) * sample_rate)
    if hop <= 0:
        raise ValueError("overlap must be smaller than window")
    if len(waveform) <= win:
        return [waveform]
    chunks: list[np.ndarray] = []
    for start in range(0, len(waveform) - win + 1, hop):
        chunks.append(waveform[start : start + win])
    if (len(waveform) - win) % hop != 0:
        chunks.append(waveform[-win:])
    return chunks

