# OpenCode build brief: Bim-ASR SALM wiring

> **Why this file exists.** The OpenClaw sessions_spawn tool on this
> Gateway is locked to `agentId: "main"`; you cannot dispatch directly
> to the OpenCode engineering manager from a subagent. This file is the
> build brief for the next OpenCode session, and the run command at the
> bottom will start it.

## Project

/Users/matt/Development/bim-asr

Bim-ASR is a Speech-Augmented Language Model (SALM) that fuses a
frozen FastConformer acoustic encoder with a frozen Qwen3-4B-Instruct
LLM through a small linear projection plus a LoRA adapter. Only the
projection and LoRA are trained.

The reference is nvidia/canary-qwen-2.5b. We use a 4B Qwen3-Instruct
base instead of 1.7B, and a Parakeet TDT 0.6B v2 encoder instead of
Canary.

## What already exists

- `src/bim_asr/model.py` - `BimAsrConfig`, `AudioProjection`, `BimAsrModel` (PyTorch). The projection is 1.31M params, encoder_hidden=512 -> llm_hidden=2560.
- `src/bim_asr/data.py` - NeMo manifest writer/reader with audio SHA-256, duration probe, ffmpeg resample.
- `src/bim_asr/audio.py` - 16 kHz mono loader, 30 s/5 s overlap chunker.
- `src/bim_asr/mlx_model.py` - MLX mirror of the projection. Reports `is_mlx_available()`.
- `src/bim_asr/__init__.py`
- `scripts/transcribe.py` - inference entry point (currently a stub that prints "no transcripts produced").
- `scripts/download_data.py` - dataset downloader for LibriSpeech, Common Voice 17, GigaSpeech XS, and Barbados Parliament YouTube/Vimeo.
- `scripts/eval_proper_nouns.py` - proper-noun evaluator using the BimOmni 10-noun list. Tested on a 4-clip set (got 0.075).
- `tests/test_smoke.py` - 8 tests, all passing.
- `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, `docs/DATA.md`
- `pyproject.toml` and `.venv` (mlx 0.32.4, torch 2.14.1, transformers 5.20.1, peft 0.21.2)
- `ffmpeg` 9.0.2 and `yt-dlp` 2026.8.19 installed via Homebrew.

## What to do (in this order)

### 1. Real SALM forward pass in `src/bim_asr/inference.py`

- Load `nvidia/parakeet-tdt-0.6b-v2` via the NeMo toolkit (`nemo.collections.asr.models.ASRModel.from_pretrained`) and run only the encoder to get per-frame features.
- Project the features through `BimAsrModel.projection`.
- Build the SALM prompt using `build_default_prompt` from `src/bim_asr/model.py`. Tokenise. Splice the projected audio embeddings in at the position of the audio locator token.
- Call the Qwen3 LLM with the LoRA adapter loaded, decoding the generated ids back to text.
- `scripts/transcribe.py` should call this for every 30 s window and stitch outputs. Remove the stub warning.

### 2. NeMo SALM training config at `configs/bim_asr_v0.yaml`

- Based on the stock nvidia/canary-qwen-2.5b `salm.yaml`.
- Encoder: nvidia/parakeet-tdt-0.6b-v2
- LLM: Qwen/Qwen3-4B-Instruct-2507
- Projection shape: (512, 2560)
- LoRA: rank 32 on q/k/v/o_proj
- Frozen encoder, trainable projection + LoRA
- Audio window 30 s, max tokens 1024
- LR 1e-4, cosine, warmup 500 steps, bf16
- Reference: https://github.com/NVIDIA/NeMo/blob/main/examples/speechlm2/conf/salm.yaml

### 3. `scripts/prepare_parakeet_encoder.py`

- Extract just the encoder from the Parakeet TDT 0.6B v2 NeMo checkpoint.
- Save it in a form the MLX conversion can consume.
- Document the CC-BY-4.0 licence at the top of the file.

### 4. `scripts/smoke_salm.py` (Stage 0)

- Build a tiny BimAsrModel.
- Patch in a fake encoder that returns random 512-dim frames.
- Patch in a small Qwen3 (Qwen/Qwen3-0.6B or whatever fits in 16 GB M4) with a fresh LoRA.
- Run one forward pass and print a generated token.
- This is the stage 0 smoke test from `docs/TRAINING.md`. Goal is to prove the SALM wiring works on a 16 GB Mac before we rent a GPU.

### 5. Update `README.md`

- Point at the new files.
- Add a short "Next steps" section listing what remains.

## Constraints

- Do not edit `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, or `docs/DATA.md` without checking the user first. They were written overnight and may need to stay as-is.
- The projection is already implemented in `src/bim_asr/model.py`. Do not duplicate it.
- Keep the SALM prompt format from `build_default_prompt` in `src/bim_asr/model.py`.
- Work in a feature branch of the git repo, not on main.
- For any opencode work, the user wants the model
  `minimax-coding-plan/MiniMax-M3.1-Flash-Preview`. Set this with the
  opencode `--model` flag or the relevant provider override.

## Run command

The user starts the build session from the shell with:

    cd /Users/matt/Development/bim-asr
    git checkout -b feature/salm-wiring
    opencode \
      --model minimax-coding-plan/MiniMax-M3.1-Flash-Preview \
      --prompt "Read OPENCODE_BRIEF.md at the repo root and execute the five tasks it lists. Summarise the diff and any blockers when done." \
      --print-logs

## When you finish

Report:

- Files added or changed (one line each).
- A diff summary (`git diff main`) so the user can review.
- Open questions or blockers.
- What the next dispatch should cover (e.g. data prep on rented H100, full training run, leaderboard submission).

