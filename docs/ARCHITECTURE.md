# Architecture

This document explains why Bim-ASR looks the way it does, and how each
piece relates to the prior art.

## TL;DR

Bim-ASR is a **Speech-Augmented Language Model** in the same family as
NVIDIA's Canary-Qwen-2.5B. A FastConformer acoustic encoder turns audio
into a sequence of embedding frames. A small **linear projection** maps
those frames into the embedding space of a dense LLM (Qwen3-4B-Instruct).
A **LoRA adapter** on the LLM lets the language model speak with
Barbadian vocabulary and style without retraining the whole 4B
parameter base. The acoustic encoder and LLM base stay frozen; only the
projection and LoRA adapters are trained.

    16 kHz audio
         |
         v
    +-----------------+         +-------+         +-----------------+
    | FastConformer   |  -->    |  Lin  |  -->    | Qwen3-4B + LoRA |
    | (frozen)        | frames  | proj  | tokens  | (frozen + LoRA) |
    +-----------------+         +-------+         +-----------------+
                                                          |
                                                          v
                                                   transcript text

## Why this pattern

Three reference designs cover the space:

1. **Pure encoder-decoder ASR (Parakeet TDT, Canary).** Fast and
   accurate for general English, but the language prior is whatever
   lived in the decoder at pretrain time. Adding Barbadian vocabulary
   means re-training the decoder head, which is expensive and easy to
   over-fit.

2. **End-to-end audio LLM (LFM2-Audio-1.5B, Qwen2.5-Omni).** A single
   model is trained from scratch to consume audio and emit text. The
   strong language prior is built in, but the model is large and the
   pretraining cost is high. BimOmni-30B-A3B is in this family and was
   the lesson: it works, but it is too big and too slow.

3. **Speech-augmented LLM (SALM, used by Canary-Qwen-2.5B).** A small
   projection layer feeds audio embeddings into an LLM, and a LoRA
   adapter teaches the LLM to use them. The base LLM keeps its full
   language ability, the projection is cheap, and the LoRA is
   parameter-efficient. NVIDIA's Canary-Qwen-2.5B sits at the top of
   the Open ASR Leaderboard using exactly this recipe, and it is small
   enough to run on one or two GPUs.

Bim-ASR uses pattern 3. It is the same approach as
[nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b),
swapped from a 1.7B Qwen3 base to a 4B Qwen3-Instruct base so that the
Barbados-aware training (DAPT plus LoRA) can lean on a stronger,
already-instruction-tuned language model.

## Pieces

### 1. Acoustic encoder: FastConformer

We use the **FastConformer encoder** from NVIDIA's Parakeet TDT 0.6B v2
([model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)). The
choice between Parakeet TDT 0.6B v2 (English-only) and Canary 1B Flash
(4 languages) is a tradeoff:

- **Parakeet TDT 0.6B v2** — slightly faster, 80 ms frame rate, English
  only. Source weights are CC-BY-4.0.
- **Canary 1B Flash** — 883 M params, English/German/French/Spanish,
  encoder is FastConformer. Slightly heavier but the encoder weights are
  pretrained on more languages.

For an English-anchored Caribbean project, **Parakeet TDT 0.6B v2** is
the better starting point. We only need the **encoder** (not the TDT
decoder), so we discard the decoder and treat the encoder as a
frozen feature extractor.

Hidden size: **512** (FastConformer-XL config). Output: roughly 12.5
frames per second of audio, each 512-dim.

The encoder is frozen for the main Bim-ASR run. Optional stage (see
TRAINING.md) does a small LoRA on the encoder to absorb Barbadian
phonetics.

### 2. Language model: Qwen3-4B-Instruct-2507

We use [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
as the dense backbone. Key facts:

- 4.0 B parameters (3.6 B non-embedding)
- 36 transformer layers
- Grouped-query attention: 32 Q heads, 8 KV heads
- Hidden size: **2560**
- Native context: 262 144 tokens
- **Non-thinking** mode only (we do not need chain-of-thought for ASR)
- Pretrained on 36 T tokens across 119 languages
- Released under Apache 2.0

Why not the 30B-A3B MoE we used for BimOmni? Because BimOmni was
already too big and too slow, and an MoE forces routing decisions that
make a small ASR head feel out of place. A dense 4B fits comfortably in
16 GB of unified memory after 4-bit quantisation (about 2.5 GB), and
runs on Metal with mlx-lm.

Why not Nemotron-H-4B? Nemotron's hybrid Mamba-2/attention design is
attractive, but it is licensed under NVIDIA's internal R&D licence,
which rules out open redistribution. Qwen3-4B is Apache 2.0.

Why not Qwen2.5-Omni-3B? That model already has a built-in audio
tower. The whole point of Bim-ASR is to test the projection + LoRA
recipe at a smaller scale than the leaderboard SOTA. Using a fresh
small dense base also keeps the comparison clean.

### 3. Bridge: linear projection + LoRA

The bridge has two parts.

**Linear projection.** The FastConformer outputs 512-dim frames at
12.5 Hz. The Qwen3 embedding space is 2560-dim. A single
`nn.Linear(512, 2560)` maps each frame to a token-embedding-shaped
vector. After the projection, the audio frames are interleaved with
text token embeddings and fed to the LLM exactly the way SALM does it.

The projection is **trained from scratch** (no warm-up). It is small
(1.3 M parameters) and converges quickly.

**LoRA adapter on the LLM.** A rank-32 LoRA on the LLM's attention
projection layers (q_proj, k_proj, v_proj, o_proj). Roughly 27 M
trainable parameters, or about 0.7 % of the 4B base. The MLP and
embedding layers stay frozen.

The LoRA is what gives the model its Caribbean accent. The same Qwen3
base, with a different LoRA, could be turned into a code completion
model, a customer-service agent, or anything else. That is the point of
using LoRA instead of full fine-tuning.

### 4. Training framework

We train in **NVIDIA NeMo** (`nemo.collections.speechlm2`) because
NVIDIA has already open-sourced the SALM training script and base
config, and because NeMo handles audio preprocessing, FSDP, and
gradient accumulation out of the box.

The relevant files are:
- Training script: `examples/speechlm2/salm_train.py`
- Base config: `examples/speechlm2/conf/salm.yaml`

We adapt those to:
- Use `nvidia/parakeet-tdt-0.6b-v2` for the encoder (only)
- Use `Qwen/Qwen3-4B-Instruct-2507` for the LLM
- Use a fresh projection of shape `(512, 2560)`
- Freeze the encoder and most of the LLM, train projection + LoRA
- Add Caribbean data after the initial English run

For laptop-side development, we mirror the architecture in
**MLX + mlx-lm** so the model can be loaded and run on the M4
unified memory. The laptop path is **inference only**; full training
lives on a rented H100 or H200.

## Why a SALM, not a full audio LLM

BimOmni used an existing audio LLM (Qwen3-Omni-30B-A3B) and adapted it
with LoRA on Barbados newspaper text. That gave us a 13.3-point lift on
Barbados proper nouns in the model card's evaluation, and the
visual/audio towers were useful for follow-on multimodal work. But the
model was 30B parameters of MoE, ran at 25-30 seconds of wall time for
30 seconds of audio on a Mac, and did not actually learn Barbadian
phonetics because the adaptation was text-only.

The hypothesis for Bim-ASR is:

1. **Most of the lift BimOmni got was from local language priors, not
   from the audio tower.** A 4B instruct model with a strong LoRA
   should get most of the same benefit at a fraction of the size.
2. **A small projection layer + small LoRA is a much faster iteration
   loop.** We can run new Caribbean training rounds in hours, not
   days.
3. **Inference stays under real time on a laptop.** This matters for
   Pulse-style ingestion of broadcast audio, where a 30 s per 30 s
   pipeline blocks the next step.

## Open questions we will answer in training

- Does a 4B instruct base really beat a 1.7B base on Caribbean
  proper-noun WER? (Compare against Canary-Qwen-2.5B at 1.7B.)
- How much does adding Common Voice Caribbean speakers help versus
  a clean English baseline?
- Is a 30-40 second audio window enough for Parliament-style speech,
  or do we need to extend it? Canary-Qwen-2.5B was trained at 40 s.
- Can the projection + LoRA absorb a real Barbadian accent (vowel
  shifts, dropped final consonants) with a small fine-tune on paired
  Parliament audio?
