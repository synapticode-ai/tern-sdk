# Copyright 2026 Gamma Seeds Pte Ltd. All rights reserved.
# Inventor: Robert Lakelin
# Patent portfolio: Synapticode Patents 1-98

"""
tern -- Ternary AI for Apple Silicon

Quick start:
    import tern
    model = tern.convert("distilgpt2")
    runtime = tern.deploy(model)
    output = runtime.infer("Hello, ternary world")
"""

from tern.api.convert import convert, TernModel
from tern.api.deploy import deploy, TernRuntime
from tern.api.output import TernOutput

__version__ = "0.1.0"
__all__ = ["convert", "deploy", "TernModel", "TernRuntime", "TernOutput"]
