# Training

This is the staged training plan for Bim-ASR. Each stage is independent and
the artefacts are saved so we can re-run, compare, and roll back.

## Stages at a glance

| Stage | What it trains                | What data                      | Where          | Target metric                  |
| ----- | ----------------------------- | ------------------------------ | -------------- | ------------------------------ |
| 0     | smoke test (tiny subset)      | LibriSpeech dev-clean          | local M4       | pipeline runs end to end       |
| 1     | projection + LoRA             | LibriSpeech train-clean-100    | 1-2x H100      | WER < 8 % on dev-clean         |
| 2     | projection + LoRA             | + GigaSpeech XS subset         | 1-2x H100      | WER < 6 % on dev-clean         |
| 3     | projection + LoRA             | + Common Voice 17 (Caribbean)  | 1-2x H100      | Caribbean WER within 2 % of US |
| 4     | projection + LoRA + enc LoRA  | + Parliament audio (paired)    | 1-2x H100      | proper-noun F1 within 5 % of BimOmni |
| 5     | full-stack evaluation         | held-out Parliament clips      | local M4 + HF | publish to leaderboard + HF   |

Stages 0-2 are general English. Stage 3 adds Caribbean English. Stage 4
adds paired Barbadian Parliament audio. Stage 5 is the evaluation and
publication step.

## Stage 0 — smoke test (local M4)

Goal: prove the SALM scaffolding works end to end on the laptop. We
should see a non-zero loss go down across a few hundred steps.

- Data: 100 utterances from LibriSpeech dev-clean.
- Compute: Mac M4, 16 GB unified memory, MLX backend.
- Steps: 200, batch size 1, gradient accumulation 4.
- Expected loss curve: noisy but trending down; no NaNs.

If this fails, do not start renting GPUs.

## Stage 1 — general English baseline (rented GPU)

Goal: a working SALM with Parakeet encoder + Qwen3-4B + projection +
LoRA, trained on a small English corpus. This is the reference run.

- Data: LibriSpeech train-clean-100, 100 hours of read English.
  Manifest in NeMo JSONL format with `audio_filepath`,
  `duration`, and `text`.
- Compute: 1x NVIDIA H100 80GB on Hugging Face Jobs.
- Hyperparameters (initial):
  - Optimiser: AdamW, lr 1e-4, weight decay 0.0
  - LR schedule: cosine, warmup 500 steps
  - Batch size 8, grad accumulation 4 → effective 32
  - Max audio length 30 s, max tokens 1024
  - LoRA rank 32, alpha 64, on q/k/v/o_proj
  - Precision: bfloat16
  - Steps: 5 000 (one epoch over train-clean-100)
- Evaluation: WER on LibriSpeech dev-clean and dev-other every 500 steps.
- Expected outcome: WER close to or under Canary-Qwen-2.5B at 1.7B, with
  the projection doing the heavy lifting. (Rough target: 5-8 % on
  dev-clean, 12-18 % on dev-other.)

## Stage 2 — wider general English (rented GPU)

Goal: close the gap to Canary-Qwen-2.5B on a broader training set.

- Data: GigaSpeech XS subset (~ 250 hours curated, balanced across
  audiobook, podcast, YouTube).
- Compute: 1x NVIDIA H100 80GB, ~ 24 hours.
- Hyperparameters: same as Stage 1, with 2 epochs.
- Evaluation: WER on LibriSpeech dev-clean and dev-other, plus a
  held-out GigaSpeech dev slice.

## Stage 3 — Caribbean English (rented GPU)

Goal: the Caribbean-specific lift. The model should now start picking up
local names, accents, and prosody.

- Data: Common Voice 17 English speakers from Caribbean islands
  (Jamaica, Trinidad, Barbados, Guyana, Belize). The CV 17 dataset has
  speaker metadata we can filter on. Plus any CV Caribbean clips that
  the project's existing data pipeline has already produced.
- Compute: 1x NVIDIA H100 80GB, ~ 12 hours.
- Recipe: continue from Stage 2 checkpoint, lower lr to 5e-5, 2 epochs.
- Evaluation: a small Caribbean-specific WER set we curate from
  Common Voice 17 Caribbean speakers. Compare against the unadapted
  baseline.

## Stage 4 — Barbadian Parliament (rented GPU)

Goal: this is the model that ships. The model should produce strong
Barbadian proper-noun transcripts.

- Data: a small, paired Barbadian Parliament audio dataset. We need
  (a) audio of the House of Assembly or Senate, (b) transcripts that
  match the audio. The transcripts come from a frontier model first
  (Whisper-large-v3 or Qwen2-Audio), then are corrected by hand.
- For the first run, even 5-10 hours of paired audio is enough to
  shift the model's behaviour. BimOmni trained on text only and still
  moved the needle on proper nouns; we expect the audio supervision to
  move it further.
- Compute: 1x NVIDIA H100 80GB, ~ 6 hours.
- Recipe: continue from Stage 3, lower lr to 2e-5, 1 epoch.
- Optional sub-stage: also unfreeze a low-rank LoRA on the
  FastConformer encoder (rank 8 on the conv + attn layers). This is
  where the model can pick up Barbadian phonetics directly from
  audio without forgetting the English pretraining.

## Stage 5 — evaluation and release (local + HF)

Goal: prove the model is good and publish it.

- Evaluation sets:
  - LibriSpeech test-clean and test-other (sanity check)
  - GigaSpeech test (sanity check)
  - Internal Caribbean test set from Common Voice 17
  - Held-out Barbadian Parliament clips (10-20 minutes, hand-transcribed)
  - The BimOmni 10-proper-noun table (legacy comparison)
- Metrics: WER, proper-noun F1, RTFx on Mac M4.
- Publication: push adapter weights to
  `hammertoe/bim-asr-1b` (or similar), with a model card that
  mirrors the BimOmni layout.
- Leaderboard: request a slot on the Open ASR Leaderboard via PR. If
  accepted, that is the strongest single-number signal.

## Hyperparameter ledger

| Symbol             | Value  | Why                                       |
| ------------------ | ------ | ----------------------------------------- |
| lora_rank          | 32     | Matches Canary-Qwen-2.5B config           |
| lora_alpha         | 64     | 2x rank is the standard recipe            |
| lora_dropout       | 0.05   | Mild regularisation                       |
| encoder_lr_mult    | 0.0    | Encoder is frozen in stages 1-3           |
| projection_lr_mult | 1.0    | Projection is the bridge, full speed      |
| lora_lr_mult       | 1.0    | LoRA is the new knowledge, full speed     |
| audio_window       | 30 s   | Canary-Qwen-2.5B used 40 s; 30 is safer   |
| max_tokens         | 1024   | Matches Canary-Qwen-2.5B                  |
| label_smoothing    | 0.1    | Standard for seq-to-seq                   |
| warmup_steps       | 500    | Roughly 10 % of Stage 1 schedule          |
| weight_decay       | 0.0    | LoRA does not need regularisation         |
| precision          | bf16   | H100 has good bf16 throughput             |
| grad_clip          | 1.0    | Standard                                  |

## Loss curves we expect

Stage 1 should show a clean monotonic decline in cross-entropy loss on
the train set, with the eval WER falling roughly in step. Stages 3-4
may have a brief uptick at the start (the Caribbean data is
out-of-distribution for the Stage 2 model) before the loss re-converges.

If we see divergence at any stage, halve the learning rate and resume
from the previous checkpoint.

## What "done" looks like

- Stage 0: smoke test runs without errors on M4.
- Stage 1: dev-clean WER <= 8 % within 5 000 steps.
- Stage 2: dev-clean WER <= 6 %.
- Stage 3: Caribbean WER within 2 % of the US WER.
- Stage 4: proper-noun F1 on the BimOmni 10-proper-noun set >= 9/10.
- Stage 5: model card published, leaderboard entry requested, M4 RTFx
  published.
