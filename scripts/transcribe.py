#!/usr/bin/env python3
"""Bim-ASR inference script.

Loads the bim-asr model (PyTorch or MLX backend), reads audio, runs
the SALM forward pass, and prints the transcript. The script works
with the unfused base + LoRA weights for now; once we have a fused
checkpoint this is the entry point.

Examples
--------

Transcribe one file with the PyTorch backend (CUDA or CPU):

    python scripts/transcribe.py path/to/audio.wav

Transcribe with the MLX backend on Apple Silicon:

    python scripts/transcribe.py --backend mlx path/to/audio.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make src/ importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from bim_asr.audio import chunk_waveform, load_audio_16k_mono  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bim-ASR transcription")
    p.add_argument("audio", type=str, help="Path to an audio file (wav/mp3/etc.)")
    p.add_argument(
        "--backend",
        choices=["pytorch", "mlx"],
        default="pytorch",
        help="Inference backend (default: pytorch)",
    )
    p.add_argument(
        "--encoder-id",
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="HF id for the audio encoder",
    )
    p.add_argument(
        "--llm-id",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HF id for the language model",
    )
    p.add_argument(
        "--lora-adapter",
        default=None,
        help="Path to a LoRA adapter to load on top of the LLM",
    )
    p.add_argument(
        "--window-seconds",
        type=float,
        default=30.0,
        help="Audio window in seconds (default 30)",
    )
    p.add_argument(
        "--overlap-seconds",
        type=float,
        default=5.0,
        help="Overlap between windows in seconds (default 5)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per window",
    )
    p.add_argument(
        "--quantize-4bit",
        action="store_true",
        help="Quantise the LLM to 4-bit (PyTorch backend, requires bitsandbytes)",
    )
    return p.parse_args()


def transcribe_window(
    waveform,
    sample_rate: int,
    *,
    backend: str,
    encoder_id: str,
    llm_id: str,
    lora_adapter: str | None,
    max_new_tokens: int,
    quantize_4bit: bool,
) -> str:
    """Run one inference pass over one audio window.

    This is a stub that prints a clear "not implemented" message. The
    real implementation wires the FastConformer encoder to the
    AudioProjection to the Qwen3 LLM with a SALM prompt. It lives in
    src/bim_asr/inference.py once the encoder is loaded in this
    environment.
    """
    print(
        "  [stub] encoder=", encoder_id, "llm=", llm_id, "lora=", lora_adapter,
        "backend=", backend, "quantize_4bit=", quantize_4bit,
        "max_new_tokens=", max_new_tokens, "samples=", len(waveform),
        file=sys.stderr,
    )
    return ""


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"audio file not found: {audio_path}", file=sys.stderr)
        return 2

    print(f"loading audio: {audio_path}", file=sys.stderr)
    waveform = load_audio_16k_mono(audio_path)
    sr = 16000
    print(f"  duration: {len(waveform) / sr:.2f} s", file=sys.stderr)

    chunks = chunk_waveform(
        waveform,
        sample_rate=sr,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    print(f"  windows: {len(chunks)} (each {args.window_seconds}s, overlap {args.overlap_seconds}s)", file=sys.stderr)

    transcripts: list[str] = []
    t0 = time.time()
    for i, chunk in enumerate(chunks):
        text = transcribe_window(
            chunk,
            sr,
            backend=args.backend,
            encoder_id=args.encoder_id,
            llm_id=args.llm_id,
            lora_adapter=args.lora_adapter,
            max_new_tokens=args.max_new_tokens,
            quantize_4bit=args.quantize_4bit,
        )
        transcripts.append(text)
        print(f"  window {i + 1}/{len(chunks)} done ({time.time() - t0:.1f}s)", file=sys.stderr)
    elapsed = time.time() - t0
    print(f"total wall time: {elapsed:.2f}s ({len(waveform) / sr / max(elapsed, 0.01):.1f}x RTFx)", file=sys.stderr)

    full = " ".join(t for t in transcripts if t).strip()
    if not full:
        print(
            "WARNING: no transcripts produced. The PyTorch inference path is "
            "stubbed until the FastConformer encoder is loaded into this "
            "environment. The real implementation lives in src/bim_asr/inference.py.",
            file=sys.stderr,
        )
        return 0
    print(full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
