# Copyright 2026 Gamma Seeds Pte Ltd. All rights reserved.
# Inventor: Robert Lakelin
# Patent alignment: Patent 28 (deterministic runtime output)

"""TernOutput — inference result container."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TernOutput:
    """Output from a ternary inference call."""

    text: str                    # generated text or top prediction
    latency_ms: float            # wall-clock inference time
    device: str                  # "ANE" | "GPU" | "CPU"
    tokens_per_second: float     # throughput
    model_id: str
