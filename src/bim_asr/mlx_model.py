"""MLX mirror of the Bim-ASR model for Apple Silicon inference.

This is the laptop path. It mirrors the PyTorch model definition so
that the same weights (after conversion) can run on the M4 unified
memory without a CUDA GPU.

The MLX conversion of the FastConformer encoder and Qwen3 LLM
happens elsewhere (see scripts/convert_to_mlx.py). This file is
just the bridge and the SALM forward loop in MLX.
"""

from __future__ import annotations

try:
    import mlx.core as mx
    from mlx import nn as mlxnn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore
    mlxnn = None  # type: ignore


if HAS_MLX:

    class MlxAudioProjection(mlxnn.Module):
        """MLX version of the FastConformer -> Qwen3 projection bridge."""

        def __init__(self, encoder_hidden: int = 512, llm_hidden: int = 2560) -> None:
            super().__init__()
            self.norm = mlxnn.LayerNorm(encoder_hidden)
            self.proj = mlxnn.Linear(encoder_hidden, llm_hidden, bias=False)

        def __call__(self, audio_features):
            return self.proj(self.norm(audio_features))

else:

    class MlxAudioProjection:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("MLX is not available on this machine")


def is_mlx_available() -> bool:
    """Return True if the MLX backend can be used."""
    return HAS_MLX

