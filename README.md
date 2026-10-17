# Bim-ASR

A small, fast, Caribbean-aware speech recognition stack. It fuses a
strong acoustic encoder (FastConformer) with a small dense language model
(Qwen3-4B-Instruct) so the transcription head can lean on local context the
way a human listener does.

This is the second-generation follow-up to
`hammertoe/BimOmni-30B-A3B`. BimOmni handled Barbadian proper nouns much
better than its base, but it was a 30B-A3B MoE that was too big and too
slow for everyday use. Bim-ASR keeps the local-aware behaviour while
fitting on a laptop and finishing transcription well under real time.

## Status

Research, design, and initial scaffolding. Built overnight in the run-up
to an NVIDIA hackathon.

## Layout

- `docs/ARCHITECTURE.md` — design rationale, why each piece was chosen
- `docs/TRAINING.md` — staged training plan, data, hyperparameters
- `docs/DATA.md` — dataset sources and licenses
- `src/bim_asr/` — the model code
- `scripts/` — small helper scripts
- `notebooks/` — exploratory notebooks
- `tests/` — smoke tests

## Components

- **Acoustic encoder** — FastConformer from NVIDIA Canary/Parakeet,
  ~600M params, 80 ms frame rate.
- **Language model** — Qwen3-4B-Instruct-2507, dense, 256K context.
- **Bridge** — linear projection from audio embeddings to LLM token
  embedding space, plus a LoRA adapter on the LLM.
- **Inference** — MLX on Apple Silicon (laptop) and NeMo on rented
  H100/H200 (training).

## Naming

`Bim` is the local nickname for Barbados. The merged model name will
probably be `hammertoe/Bim-ASR-1B`.

## Provenance

- Author: Matt Hamilton (`@hammertoe`)
- Project: Future Caribbean
- Previous model: `hammertoe/BimOmni-30B-A3B`
- Stack: NVIDIA NeMo, MLX, Hugging Face Transformers, mlx-lm

