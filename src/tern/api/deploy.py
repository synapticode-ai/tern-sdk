# Copyright 2026 Gamma Seeds Pte Ltd. All rights reserved.
# Inventor: Robert Lakelin
# Patent alignment: Patent 27 (NPU orchestration), Patent 28 (deterministic runtime)

"""tern.deploy() and TernRuntime — deploy and run inference on ternary models."""

from __future__ import annotations

import time
from pathlib import Path

from tern.api.output import TernOutput


class TernRuntime:
    """A deployed ternary model ready for inference."""

    def __init__(self, tern_model: "TernModel"):
        self._tern_model = tern_model
        self._loaded_model = None
        self._tokenizer = None
        self._device = "CPU"
        self._is_healthy = True
        self._is_unloaded = False
        self._load()

    def _load(self):
        """Load the model for inference -- CoreML if available, PyTorch fallback."""
        mlpackage_path = Path(self._tern_model.mlpackage_path)

        # Try CoreML first (real .mlpackage from stages 4-5)
        coreml_loaded = False
        if mlpackage_path.exists() and not (mlpackage_path / "tern_fallback.json").exists():
            try:
                import coremltools as ct
                self._loaded_model = ct.models.MLModel(str(mlpackage_path))
                self._device = "ANE"
                coreml_loaded = True
            except Exception:
                pass

        # Fallback: use original PyTorch model from convert()
        if not coreml_loaded:
            if self._tern_model._original_model is not None:
                self._loaded_model = self._tern_model._original_model
                self._device = "CPU"
            else:
                # Last resort: reload from HuggingFace
                try:
                    import torch
                    from transformers import AutoModelForCausalLM
                    self._loaded_model = AutoModelForCausalLM.from_pretrained(
                        self._tern_model.model_id, torch_dtype=torch.float32
                    )
                    self._loaded_model.eval()
                    self._device = "CPU"
                except Exception:
                    self._is_healthy = False

        # Load tokenizer
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._tern_model.model_id)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        except Exception:
            self._is_healthy = False

    def infer(
        self,
        input,                        # str | list[str] | dict
        max_tokens: int = 256,
        temperature: float = 0.0,     # 0.0 = deterministic
    ) -> TernOutput:
        """
        Run inference. Returns TernOutput with text, latency, and device info.

        If a real .mlpackage is loaded, uses CoreML prediction.
        Otherwise falls back to PyTorch generation on CPU.

        Patents: 27 (dispatch), 28 (determinism), 36 (deterministic guarantee).
        """
        if self._is_unloaded:
            raise RuntimeError("Runtime has been unloaded. Deploy again to infer.")

        prompt = input if isinstance(input, str) else str(input)

        t0 = time.time()

        if self._device == "ANE":
            # CoreML inference path
            generated_text = self._infer_coreml(prompt, max_tokens)
        else:
            # PyTorch fallback
            generated_text = self._infer_pytorch(prompt, max_tokens, temperature)

        latency_ms = (time.time() - t0) * 1000

        # Estimate tokens per second
        num_tokens = len(self._tokenizer.encode(generated_text)) if self._tokenizer else len(generated_text.split())
        tokens_per_second = (num_tokens / (latency_ms / 1000)) if latency_ms > 0 else 0.0

        return TernOutput(
            text=generated_text,
            latency_ms=round(latency_ms, 2),
            device=self._device,
            tokens_per_second=round(tokens_per_second, 1),
            model_id=self._tern_model.model_id,
        )

    def _infer_coreml(self, prompt: str, max_tokens: int) -> str:
        """Run inference via CoreML model."""
        import numpy as np
        tokens = self._tokenizer.encode(prompt, return_tensors="np")
        prediction = self._loaded_model.predict({"input_ids": tokens})
        # Decode output -- shape depends on model
        output_ids = prediction.get("logits", prediction.get("output", None))
        if output_ids is not None:
            output_ids = np.argmax(output_ids, axis=-1)
            return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return prompt

    def _infer_pytorch(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Run inference via PyTorch model (fallback)."""
        import torch
        tokens = self._tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self._loaded_model.generate(
                tokens,
                max_new_tokens=min(max_tokens, 50),  # limit for speed
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True)

    def health(self) -> dict:
        """Return runtime health stats."""
        return {
            "is_healthy": self._is_healthy and not self._is_unloaded,
            "device": self._device,
            "model_id": self._tern_model.model_id,
            "model_loaded": self._loaded_model is not None,
            "tokenizer_loaded": self._tokenizer is not None,
            "stages_completed": self._tern_model.stats.get("stages_completed", []),
        }

    def unload(self):
        """Explicitly unload model and free memory."""
        self._loaded_model = None
        self._tokenizer = None
        self._is_unloaded = True
        self._is_healthy = False


def deploy(tern_model: "TernModel", device: str = "ane") -> TernRuntime:
    """
    Deploy a TernModel for inference.

    Example:
        runtime = tern.deploy(model)
        output = runtime.infer("Explain ternary computing")
        print(output.text)
    """
    return TernRuntime(tern_model)
