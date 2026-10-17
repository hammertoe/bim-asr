#!/usr/bin/env python3
"""Download the datasets Bim-ASR uses.

We do not commit any audio. This script is the only sanctioned way
to bring audio onto the training box. Each download step has a guard
that skips the step if the data is already on disk.

Datasets
--------
- LibriSpeech (CC-BY-4.0): train-clean-100, dev-clean, test-clean
- Common Voice 17 (CC-0): the en/ subset for the English baseline,
  plus the Caribbean speaker filter for the Caribbean training
- GigaSpeech (research only, gated): XS subset only for now
- Barbados Parliament: yt-dlp against the official channel

Example
-------

    python scripts/download_data.py --datasets librispeech-100,cv17-en
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Make src/ importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bim_asr.data import write_jsonl_manifest, AudioSample, probe_duration_seconds  # noqa: E402

DATA_ROOT = Path(os.environ.get("BIM_ASR_DATA", ROOT / "data"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Bim-ASR datasets")
    p.add_argument(
        "--datasets",
        default="librispeech-100",
        help="Comma-separated list of datasets. One or more of: "
             "librispeech-100, librispeech-dev, cv17-en, gigaspeech-xs, parliament",
    )
    p.add_argument(
        "--parliament-videos",
        default="",
        help="Comma-separated list of YouTube or Vimeo URLs for the parliament step",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Where to put the audio (default: ./data or $BIM_ASR_DATA)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip datasets that are already on disk (default: true)",
    )
    return p.parse_args()


def download_librispeech_100(root: Path, skip_existing: bool) -> Path:
    target = root / "librispeech" / "train-clean-100"
    if skip_existing and target.exists() and any(target.iterdir()):
        print(f"  already on disk: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "huggingface-cli", "download",
        "openslr/librispeech_asr",
        "--repo-type", "dataset",
        "--include", "train.clean.100/*",
        "--local-dir", str(target),
    ]
    print("  running:", " ".join(cmd))
    subprocess.check_call(cmd)
    return target


def download_librispeech_dev(root: Path, skip_existing: bool) -> Path:
    target = root / "librispeech" / "dev-clean"
    if skip_existing and target.exists() and any(target.iterdir()):
        print(f"  already on disk: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "huggingface-cli", "download",
        "openslr/librispeech_asr",
        "--repo-type", "dataset",
        "--include", "dev.clean/*",
        "--local-dir", str(target),
    ]
    print("  running:", " ".join(cmd))
    subprocess.check_call(cmd)
    return target


def download_cv17_en(root: Path, skip_existing: bool) -> Path:
    target = root / "common_voice_17" / "en"
    if skip_existing and target.exists() and any(target.iterdir()):
        print(f"  already on disk: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "huggingface-cli", "download",
        "mozilla-foundation/common_voice_17_0",
        "--repo-type", "dataset",
        "--include", "en/*",
        "--local-dir", str(target),
    ]
    print("  running:", " ".join(cmd))
    subprocess.check_call(cmd)
    return target


def download_gigaspeech_xs(root: Path, skip_existing: bool) -> Path:
    target = root / "gigaspeech" / "xs"
    if skip_existing and target.exists() and any(target.iterdir()):
        print(f"  already on disk: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "huggingface-cli", "download",
        "speechcolab/gigaspeech",
        "--repo-type", "dataset",
        "--include", "xs/*",
        "--local-dir", str(target),
    ]
    print("  running:", " ".join(cmd))
    subprocess.check_call(cmd)
    return target


def download_parliament(root: Path, urls: list[str]) -> list[Path]:
    target = root / "parliament"
    target.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for url in urls:
        out_wav = target / (url.split("=")[-1] + ".wav")
        if out_wav.exists():
            print(f"  already on disk: {out_wav}")
            out.append(out_wav)
            continue
        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", str(target / "%(id)s.%(ext)s"),
            url,
        ]
        print("  running:", " ".join(cmd))
        subprocess.check_call(cmd)
        # yt-dlp will name the file by id; we record the most recent
        # wav it produced.
        wvs = sorted(target.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if wvs:
            out.append(wvs[-1])
    return out


def build_librispeech_manifest(target: Path) -> Path:
    """Walk the LibriSpeech tree and produce a JSONL manifest."""
    out = target.parent / "manifest.jsonl"
    samples = []
    transcripts_dir = target / "LibriSpeech" / "train-clean-100"
    if not transcripts_dir.exists():
        transcripts_dir = target / "train-clean-100"
    for chap_dir in sorted(transcripts_dir.glob("*")):
        for speaker_dir in sorted(chap_dir.glob("*")):
            trans_file = speaker_dir / f"{speaker_dir.name}.trans.txt"
            if not trans_file.exists():
                continue
            with open(trans_file) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2:
                        continue
                    utt_id, text = parts
                    audio_path = speaker_dir / f"{utt_id}.flac"
                    if not audio_path.exists():
                        continue
                    samples.append(
                        AudioSample(
                            audio_filepath=str(audio_path),
                            duration=probe_duration_seconds(audio_path),
                            text=text,
                            source_dataset="librispeech",
                            source_split="train",
                            language="en-US",
                            speaker_id=speaker_dir.name,
                        )
                    )
    n = write_jsonl_manifest(samples, out)
    print(f"  wrote {n} samples to {out}")
    return out


def main() -> int:
    args = parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    print(f"data root: {args.data_root}")

    wanted = set(s.strip() for s in args.datasets.split(",") if s.strip())
    if "librispeech-100" in wanted:
        print("LibriSpeech train-clean-100:")
        p = download_librispeech_100(args.data_root, args.skip_existing)
        build_librispeech_manifest(p)
    if "librispeech-dev" in wanted:
        print("LibriSpeech dev-clean:")
        download_librispeech_dev(args.data_root, args.skip_existing)
    if "cv17-en" in wanted:
        print("Common Voice 17 en:")
        download_cv17_en(args.data_root, args.skip_existing)
    if "gigaspeech-xs" in wanted:
        print("GigaSpeech XS:")
        download_gigaspeech_xs(args.data_root, args.skip_existing)
    if "parliament" in wanted:
        urls = [u.strip() for u in args.parliament_videos.split(",") if u.strip()]
        if not urls:
            print("  no parliament videos specified; pass --parliament-videos")
        else:
            print("Barbados Parliament:")
            download_parliament(args.data_root, urls)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
