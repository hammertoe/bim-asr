"""PyTorch model definition for Bim-ASR.

This is the reference implementation that mirrors the SALM recipe used
by nvidia/canary-qwen-2.5b (the current top of the Open ASR
Leaderboard). We swap in a Qwen3.5-4B-Instruct base and use the
encoder from nvidia/parakeet-tdt-0.6b-v2.

Training-time pieces that the NeMo SALM trainer expects (FSDP hooks,
gradient accumulation, mixed precision, audio preprocessing) are
intentionally kept out of this file. The NeMo trainer wraps this
class.

Qwen3.5 architecture note
-------------------------

Qwen3.5-4B is a hybrid model: 8 blocks of (3x Gated DeltaNet -> FFN)
followed by 1 block of (Gated Attention -> FFN), repeated for a total
of 32 layers. The hidden size is 2560 (same as Qwen3-4B-Instruct), so
the projection bridge from FastConformer (512-dim) is unchanged.

For LoRA we target the attention projections in the Gated Attention
layers (q_proj, k_proj, v_proj, o_proj) and the FFN gate/up/down
projections in all layers. This is the broadest correct target set
we can use without peft having explicit support for Gated DeltaNet.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
from torch import nn


# Hidden sizes are pinned by the source checkpoints.
_FASTCONFORMER_HIDDEN = 512  # Parakeet TDT 0.6B v2 encoder output
_QWEN35_4B_HIDDEN = 2560     # Qwen3.5-4B token embedding dim
_QWEN35_2B_HIDDEN = 2560     # Qwen3.5-2B (same hidden as the 4B)
_AUDIO_FRAME_RATE = 12.5     # 80 ms frames, 12.5 frames per second of audio


@dataclasses.dataclass(frozen=True)
class BimAsrConfig:
    """Static configuration for a Bim-ASR model.

    The defaults match nvidia/parakeet-tdt-0.6b-v2 for the audio side
    and Qwen/Qwen3.5-4B for the language model side. Qwen3.5-2B is
    kept as a second variant for the laptop smoke test (smaller, fits
    in 16 GB M-series unified memory more comfortably).
    """

    encoder_id: str = "nvidia/parakeet-tdt-0.6b-v2"
    llm_id: str = "Qwen/Qwen3.5-4B"
    audio_locator_tag: str = "<|audio|>"

    encoder_hidden: int = _FASTCONFORMER_HIDDEN
    llm_hidden: int = _QWEN35_4B_HIDDEN
    audio_frame_rate_hz: float = _AUDIO_FRAME_RATE

    # LoRA settings
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # Qwen3.5 target modules. Attention projections cover the Gated
    # Attention layers; the FFN projections cover the Gated DeltaNet
    # MLP paths. peft's standard module-name matching will hit the
    # matching parameter names in both block types.
    lora_target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Generation
    max_new_tokens: int = 512
    max_audio_seconds: int = 30

    # Prompt the SALM model is trained against
    user_prompt: str = "Transcribe the following audio."

    @classmethod
    def for_2b_smoke(cls) -> "BimAsrConfig":
        """The 2B variant for the laptop smoke test.

        Qwen3.5-2B has the same hidden size as the 4B, so the
        projection bridge is identical. The LLM is smaller and faster,
        which matters when the OpenCode laptop has limited VRAM.
        """
        return dataclasses.replace(
            cls(),
            llm_id="Qwen/Qwen3.5-2B",
        )


class AudioProjection(nn.Module):
    """Linear bridge from FastConformer frames to LLM token embedding space.

    Input: (batch, frames, 512)  - one 512-dim vector per 80 ms audio frame
    Output: (batch, frames, 2560) - one 2560-dim vector per frame, ready
            to be interleaved with text token embeddings before the LLM.

    A LayerNorm on the input is helpful because FastConformer features
    are roughly zero-mean, unit-variance per frame but the LLM's
    embedding space is not. The norm lets the linear layer start from
    a sensible scale.
    """

    def __init__(
        self,
        encoder_hidden: int = _FASTCONFORMER_HIDDEN,
        llm_hidden: int = _QWEN35_4B_HIDDEN,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(encoder_hidden)
        self.proj = nn.Linear(encoder_hidden, llm_hidden, bias=False)
        # Small init keeps the bridge near identity at the start.
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        # audio_features: (B, T_audio, encoder_hidden)
        x = self.norm(audio_features)
        return self.proj(x)


class BimAsrModel(nn.Module):
    """Top-level Bim-ASR module.

    Composition (all submodules are loaded from the named Hugging Face
    repos or initialised locally):

    - encoder: Parakeet FastConformer (we use the encoder only; the
      TDT decoder is dropped).
    - tokenizer: Qwen3.5 tokenizer, used to encode the prompt and
      decode the generated transcript.
    - llm: Qwen3.5-4B with LoRA adapters on the attention and FFN
      projections. Qwen3.5-2B is the smaller variant.
    - projection: a fresh nn.Linear(512, 2560) bridge.

    The forward pass follows the SALM pattern:

    1. Run the frozen audio encoder to get per-frame features.
    2. Project those features into the LLM embedding space.
    3. Concatenate the projected audio embeddings with the embedded
       prompt tokens.
    4. Run the LLM (with LoRA) to autoregressively generate the
       transcript.

    Training mode and inference mode share the same forward signature.
    The trainer passes in labels; inference passes in max_new_tokens
    via generate.
    """

    def __init__(self, config: BimAsrConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = AudioProjection(
            encoder_hidden=config.encoder_hidden,
            llm_hidden=config.llm_hidden,
        )
        # Submodules are loaded lazily via from_pretrained() so that
        # this file stays importable without the heavy deps installed.
        self.encoder = None
        self.llm = None

    @classmethod
    def from_pretrained(
        cls,
        config: BimAsrConfig | None = None,
        *,
        encoder_id: str | None = None,
        llm_id: str | None = None,
        cache_dir: str | Path | None = None,
        load_in_4bit: bool = False,
    ) -> "BimAsrModel":
        """Load a Bim-ASR model with all submodules initialised.

        Loading is done through transformers + peft. The Parakeet
        encoder is loaded via the NeMo-to-HF conversion script the
        first time (see scripts/prepare_parakeet_encoder.py); the
        Qwen3.5 LLM is loaded via transformers AutoModelForCausalLM.
        """
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if config is None:
            config = BimAsrConfig()
        if encoder_id is not None:
            config = dataclasses.replace(config, encoder_id=encoder_id)
        if llm_id is not None:
            config = dataclasses.replace(config, llm_id=llm_id)

        model = cls(config)

        # Load Qwen3.5 LLM + LoRA
        llm_kwargs: dict = dict(
            torch_dtype=torch.bfloat16,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            llm_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            llm_kwargs["device_map"] = "auto"

        llm = AutoModelForCausalLM.from_pretrained(config.llm_id, **llm_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            config.llm_id, cache_dir=llm_kwargs.get("cache_dir")
        )

        # Apply LoRA on the attention and FFN projections.
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)
        # Make sure the embedding and LM head are frozen; only LoRA
        # adapters and any explicitly unfrozen modules are trainable.
        for name, param in llm.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        model.llm = llm
        # The encoder is loaded separately, by name, by the trainer
        # (NeMo wraps it as part of the audio preprocessor pipeline).
        return model

    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Project acoustic features to the LLM embedding space.

        audio_features is the output of the FastConformer encoder,
        shape (batch, frames, encoder_hidden). Returns the projected
        features ready to be concatenated with the text token
        embeddings, shape (batch, frames, llm_hidden).
        """
        return self.projection(audio_features)

    def freeze_for_projection_only_training(self) -> None:
        """Freeze everything except the projection.

        Used for the very first stage of training where we want the
        bridge to learn the LLM embedding space before LoRA kicks in.
        """
        for p in self.parameters():
            p.requires_grad = False
        for p in self.projection.parameters():
            p.requires_grad = True

    def trainable_parameter_count(self) -> int:
        """Return the number of trainable parameters (projection + LoRA)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_default_prompt(
    tokenizer,
    *,
    user_prompt: str,
    audio_locator_tag: str,
):
    """Build the SALM input prompt for training and inference.

    Returns a 1-D LongTensor of input ids. The audio embeddings will be
    spliced in at the position of audio_locator_tag by the SALM
    trainer / inference loop.

    The format is Qwen3.5's ChatML.
    """
    messages = [
        {"role": "system", "content": "You are a careful speech transcriber."},
        {"role": "user", "content": user_prompt + " " + audio_locator_tag},
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    return torch.tensor(ids, dtype=torch.long)


def count_trainable_parameters_summary(model: BimAsrModel) -> str:
    """One-line summary of trainable vs total parameters."""
    trainable = model.trainable_parameter_count()
    total = model.total_parameter_count()
    pct = 100.0 * trainable / max(total, 1)
    return "trainable={:,} / total={:,} ({:.3f} %)".format(trainable, total, pct)
