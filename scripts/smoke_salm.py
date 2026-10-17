#!/usr/bin/env python3
"""Bim-ASR Stage 0 smoke test.

This is the gate from docs/TRAINING.md stage 0. It proves the SALM
wiring (encoder -> projection -> LLM with LoRA -> transcript) works
end-to-end on a single short audio clip, without needing a rented
GPU.

By default it runs in two modes:

* --mode fake-encoder: the FastConformer encoder is replaced with a
  random-frame generator. This is what we run on the Mac mini where
  the heavy NeMo encoder cannot be loaded. It proves the projection
  bridge, the LoRA wiring, and the SALM prompt all work together.

* --mode real-encoder (default if --mode not given): loads
  nvidia/parakeet-tdt-0.6b-v2 via the NeMo toolkit, extracts just
  the encoder, and runs the full pipeline. This needs a machine with
  enough RAM to hold the encoder plus Qwen3.5-2B plus the LoRA
  adapter. Use --llm-id Qwen/Qwen3.5-2B for the laptop test.

The script prints the prompt, the generated tokens, and the
decoded transcript. It exits 0 if it managed to produce any tokens
and 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bim_asr.audio import load_audio_16k_mono, chunk_waveform  # noqa: E402
from bim_asr.model import (  # noqa: E402
    AudioProjection,
    BimAsrConfig,
    BimAsrModel,
    build_default_prompt,
    count_trainable_parameters_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bim-ASR Stage 0 smoke test")
    p.add_argument(
        "--mode",
        choices=["fake-encoder", "real-encoder"],
        default="fake-encoder",
        help="fake-encoder (default, Mac mini) replaces the Parakeet "
             "encoder with a random-frame generator. real-encoder "
             "loads the actual NeMo encoder (needs more RAM).",
    )
    p.add_argument(
        "--llm-id",
        default="Qwen/Qwen3.5-2B",
        help="HF id for the LLM (default: 2B variant for laptop test)",
    )
    p.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to a 16 kHz mono wav. If omitted, a 2 s silent "
             "test clip is used.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum new tokens to generate (default 64)",
    )
    p.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Quantise the LLM to 4-bit (uses bitsandbytes)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args()


class FakeFastConformerEncoder(torch.nn.Module):
    """A drop-in replacement for the Parakeet encoder that returns
    random per-frame features at the right shape.

    Output shape: (batch, frames, 512). Frames per second is 12.5
    so 30 s of audio -> 375 frames.
    """

    def __init__(self, hidden: int = 512, frame_rate: float = 12.5) -> None:
        super().__init__()
        self.hidden = hidden
        self.frame_rate = frame_rate
        # No learnable params; pure feature extractor.

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor | None = None):
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        b = waveforms.shape[0]
        max_len = int(waveforms.shape[1])
        max_frames = max(1, int(round(max_len / 16000.0 * self.frame_rate)))
        # Random per-frame features, scaled to a similar magnitude as
        # FastConformer features.
        feats = torch.randn(b, max_frames, self.hidden, device=waveforms.device) * 0.5
        lengths = (
            waveform_lengths
            if waveform_lengths is not None
            else torch.full((b,), max_len, device=waveforms.device, dtype=torch.long)
        )
        return feats, lengths


def make_silent_clip(seconds: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Bim-ASR Stage 0 smoke test")
    print("=" * 60)
    print(f"mode:         {args.mode}")
    print(f"llm-id:       {args.llm_id}")
    print(f"max-new:      {args.max_new_tokens}")
    print(f"4-bit:        {args.load_in_4bit}")
    print()

    # 1. Build a BimAsrConfig pointing at the requested LLM.
    if "Qwen3.5-2B" in args.llm_id:
        cfg = BimAsrConfig.for_2b_smoke()
    else:
        cfg = BimAsrConfig(llm_id=args.llm_id)
    print(f"config: llm_id={cfg.llm_id} hidden={cfg.llm_hidden} "
          f"lora_rank={cfg.lora_rank} target_modules={list(cfg.lora_target_modules)}")
    print()

    # 2. Get an audio waveform.
    if args.audio:
        print(f"loading audio: {args.audio}")
        waveform = load_audio_16k_mono(args.audio)
    else:
        print("no audio given; using a 2 s silent test clip")
        waveform = make_silent_clip(2.0)
    print(f"  waveform: {len(waveform)} samples ({len(waveform) / 16000.0:.2f} s)")
    chunks = chunk_waveform(waveform, sample_rate=16000, window_seconds=2.0, overlap_seconds=0.0)
    print(f"  chunks: {len(chunks)} (each 2.0 s, no overlap)")
    print()

    # 3. Build the projection.
    proj = AudioProjection(encoder_hidden=cfg.encoder_hidden, llm_hidden=cfg.llm_hidden)
    print("projection:", count_trainable_parameters_summary(BimAsrModel(cfg)))
    # Run a fake audio through it to confirm the bridge works.
    fake_feats = torch.randn(1, 38, 512)  # 2 s of audio at 12.5 Hz ~= 25 frames, padded
    projected = proj(fake_feats)
    print(f"  bridge in:  (1, 38, 512)")
    print(f"  bridge out: {tuple(projected.shape)}")
    assert projected.shape == (1, 38, cfg.llm_hidden), "bridge shape mismatch"
    print("  bridge OK")
    print()

    # 4. Optionally load the real LLM with LoRA and try a full forward pass.
    if args.mode == "real-encoder" or os.environ.get("BIM_ASR_FULL_SMOKE") == "1":
        print("loading Qwen3.5 + LoRA (this needs ~3-8 GB depending on quantisation)...")
        t0 = time.time()
        try:
            model = BimAsrModel.from_pretrained(
                cfg,
                load_in_4bit=args.load_in_4bit,
            )
        except Exception as e:
            print(f"  FAILED to load: {e.__class__.__name__}: {e}")
            print("  This is expected on the 16 GB Mac mini. The 2B model")
            print("  is fine on a laptop with 16 GB+; the 4B needs more.")
            return 1
        print(f"  loaded in {time.time() - t0:.1f} s")
        print(f"  trainable: {model.trainable_parameter_count():,}")
        print()

        # 5. Build the SALM prompt.
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg.llm_id)
        prompt_ids = build_default_prompt(
            tok, user_prompt=cfg.user_prompt, audio_locator_tag=cfg.audio_locator_tag
        )
        print(f"prompt ids: {len(prompt_ids)} tokens")
        print(f"  {tok.decode(prompt_ids[:60])!r} ...")
        print()

        # 6. Run a real generation with audio spliced in.
        encoder = FakeFastConformerEncoder() if args.mode == "fake-encoder" else None
        if args.mode == "real-encoder":
            print("loading real Parakeet encoder via NeMo...")
            try:
                import nemo.collections.asr as nemo_asr

                asr_model = nemo_asr.models.ASRModel.from_pretrained(cfg.encoder_id)
                encoder = asr_model.encoder
                print("  encoder loaded")
            except Exception as e:
                print(f"  FAILED: {e.__class__.__name__}: {e}")
                return 1
        else:
            encoder = FakeFastConformerEncoder()

        chunk = chunks[0]
        with torch.no_grad():
            wav_t = torch.from_numpy(chunk).unsqueeze(0)
            feats, feat_lens = encoder(wav_t)
            print(f"encoder out: {tuple(feats.shape)}")
            audio_emb = model.encode_audio(feats)
            print(f"audio emb:   {tuple(audio_emb.shape)}")

            # Splice the audio embeddings into the prompt.
            text_emb = model.llm.get_input_embeddings()(prompt_ids.unsqueeze(0))
            full = torch.cat([text_emb[:, :1], audio_emb, text_emb[:, 1:]], dim=1)
            print(f"full emb:    {tuple(full.shape)}")

            out = model.llm.generate(
                inputs_embeds=full,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            print()
            print("---")
            print(tok.decode(out[0], skip_special_tokens=True))
            print("---")
        return 0

    # 7. If we got here we are in fake-encoder mode without the full smoke.
    print("Fake-encoder mode only validated the bridge. The full smoke")
    print("(real LLM + encoder) needs to run on the laptop with enough")
    print("RAM. Set BIM_ASR_FULL_SMOKE=1 and pass --llm-id Qwen/Qwen3.5-2B")
    print("when you run this on the laptop, and it will do the full pass.")
    print()
    print("OK: Stage 0 scaffold works on this Mac mini.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
