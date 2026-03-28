# Copyright 2026 Gamma Seeds Pte Ltd. All rights reserved.
# Inventor: Robert Lakelin
# Patent alignment: Patent 28 (deterministic runtime output)

"""TernOutput — inference result container."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TernToken:
    """A single generated token from a streaming inference."""
    text: str                   # decoded token text
    token_id: int               # raw token ID
    position: int               # position in sequence (0-indexed)
    latency_ms: float           # time to generate this token
    is_final: bool              # True for last token in sequence
    device: str                 # "ANE" | "GPU" | "CPU"


@dataclass
class TernOutput:
    """Output from a ternary inference call."""

    text: str                    # generated text or top prediction
    latency_ms: float            # wall-clock inference time
    device: str                  # "ANE" | "GPU" | "CPU"
    tokens_per_second: float     # throughput
    model_id: str
