"""
tern — Synapticode Developer SDK
CNS Synaptic™ by Synapticode

The developer interface for ternary AI deployment.
Convert models, deploy to NPU hardware, and integrate
ternary inference into your applications.

Quick Start:
    import tern

    model = tern.convert("meta-llama/Llama-3-8B", target="atom")
    runtime = tern.deploy(model, device="npu:0")
    output = runtime.infer("Explain ternary computing.")

Licence: Apache 2.0 (SDK only)
Core engine: proprietary — partnerships@synapticode.ai
"""

__version__ = "0.1.0-dev"
__author__ = "Synapticode Co., Ltd."


class TernModel:
    """A ternary-quantised model ready for deployment."""

    def __init__(self, path=None, metadata=None):
        self.path = path
        self.metadata = metadata or {}

    @property
    def stats(self):
        """Compression and accuracy statistics."""
        return self.metadata.get("stats", {})

    def save(self, path):
        """Save .tern-model to disk."""
        raise NotImplementedError("Pre-development stub")

    @classmethod
    def load(cls, path):
        """Load .tern-model from disk."""
        raise NotImplementedError("Pre-development stub")


class TernOutput:
    """Output from a ternary inference call."""

    def __init__(self, text=None, tokens=None, metadata=None):
        self.text = text
        self.tokens = tokens
        self.metadata = metadata or {}


class TernTrace:
    """Deterministic decision trace for an inference output."""

    def __init__(self, output=None, trace_data=None):
        self.output = output
        self.trace_data = trace_data or {}

    @property
    def top_contributors(self):
        """Top contributing weight groups by layer."""
        return self.trace_data.get("contributors", [])


class TernRuntime:
    """Deployed ternary model runtime on NPU hardware."""

    def __init__(self, model, device="npu:0", deterministic=True):
        self.model = model
        self.device = device
        self.deterministic = deterministic

    def infer(self, input, max_tokens=256, temperature=0.0):
        """Run inference on deployed model.

        Args:
            input: Text, image path, or multimodal input dict.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            TernOutput with inference result.
        """
        raise NotImplementedError("Pre-development stub")

    def explain(self, output, depth="layer"):
        """Get deterministic decision trace.

        Args:
            output: TernOutput from infer().
            depth: Trace depth — "layer", "weight", or "full".

        Returns:
            TernTrace with decision attribution.
        """
        raise NotImplementedError("Pre-development stub")

    def benchmark(self, input, iterations=100):
        """Run performance benchmark.

        Returns dict with latency_ms, throughput_tokens_per_sec,
        power_watts, and comparison_vs_fp16.
        """
        raise NotImplementedError("Pre-development stub")


def convert(model, target="atom", sensitivity=0.95, progressive=False):
    """Convert a standard model to ternary format.

    Args:
        model: Model path, HuggingFace ID, or loaded model object.
        target: Target NPU — "atom", "hexagon", "tnpu", "generic".
        sensitivity: Accuracy retention target (0.0-1.0).
        progressive: Enable multi-level compression output.

    Returns:
        TernModel ready for deployment.
    """
    raise NotImplementedError("Pre-development stub")


def deploy(model, device="npu:0", replicas=1, deterministic=True):
    """Deploy a ternary model to hardware.

    Args:
        model: TernModel or path to .tern-model file.
        device: Device identifier (e.g., "npu:0", "npu:0,1,2").
        replicas: Number of model replicas for throughput.
        deterministic: Enforce deterministic execution.

    Returns:
        TernRuntime with deployed model.
    """
    if isinstance(model, str):
        model = TernModel.load(model)
    return TernRuntime(model, device=device, deterministic=deterministic)


def version():
    """Return SDK version string."""
    return __version__
