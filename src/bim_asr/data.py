"""Dataset utilities for Bim-ASR.

We do not write a custom audio preprocessing pipeline here. The NeMo
SALM trainer brings its own audio preprocessor (it knows about
FastConformer features). For the laptop-side MLX inference path we
just need to load an audio file, resample to 16 kHz mono, and feed the
raw waveform to the encoder.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class AudioSample:
    """One audio clip with metadata.

    Matches the NeMo manifest format with a few extras for our own
    bookkeeping. The NeMo trainer reads audio_filepath, duration, and
    text. The rest is informational.
    """

    audio_filepath: str
    duration: float
    text: str
    source_dataset: str = ""
    source_split: str = "train"
    language: str = "en"
    speaker_id: str = ""
    whisper_logprob: float | None = None
    audio_sha256: str = ""

    def to_jsonl_line(self) -> str:
        d = asdict(self)
        d["audio_filepath"] = str(d["audio_filepath"])
        return json.dumps(d, ensure_ascii=False)


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl_manifest(path: str | Path) -> Iterator[AudioSample]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield AudioSample(**obj)


def write_jsonl_manifest(samples: Iterable[AudioSample], path: str | Path) -> int:
    n = 0
    with open(path, "w") as f:
        for s in samples:
            f.write(s.to_jsonl_line() + "\n")
            n += 1
    return n


def probe_duration_seconds(path: str | Path) -> float:
    """Return audio duration using ffprobe. Falls back to soundfile."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return float(out.decode().strip())
    except Exception:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)


def resample_to_16k_mono(
    in_path: str | Path, out_path: str | Path, *, overwrite: bool = False
) -> Path:
    """Resample an audio file to 16 kHz mono PCM using ffmpeg."""
    in_path = Path(in_path)
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

