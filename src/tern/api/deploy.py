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
        self._coreml_model = None
        self._pytorch_model = None
        self._tokenizer = None
        self._device = "CPU"
        self._seq_len = 512
        self._is_healthy = True
        self._is_unloaded = False
        self._load()

    def _load(self):
        """Load the model for inference -- CoreML if available, PyTorch fallback."""
        mlpackage_path = Path(self._tern_model.mlpackage_path)

        # Always keep PyTorch model for fallback
        if getattr(self._tern_model, "_original_model", None) is not None:
            self._pytorch_model = self._tern_model._original_model
        else:
            try:
                import torch
                from transformers import AutoModelForCausalLM
                self._pytorch_model = AutoModelForCausalLM.from_pretrained(
                    self._tern_model.model_id, torch_dtype=torch.float32
                )
                self._pytorch_model.eval()
            except Exception:
                pass

        # Try CoreML (real .mlpackage from stages 4-5)
        if mlpackage_path.exists() and not (mlpackage_path / "tern_fallback.json").exists():
            try:
                import coremltools as ct
                self._coreml_model = ct.models.MLModel(str(mlpackage_path))
                self._device = "ANE"
            except Exception:
                pass

        if self._coreml_model is None and self._pytorch_model is None:
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

        if self._coreml_model is not None:
            generated_text = self._infer_coreml(prompt, max_tokens)
        elif self._pytorch_model is not None:
            generated_text = self._infer_pytorch(prompt, max_tokens, temperature)
        else:
            generated_text = prompt

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
        """Run inference via CoreML model. Pads input to fixed sequence length."""
        import numpy as np
        token_ids = self._tokenizer.encode(prompt)
        num_tokens = len(token_ids)
        # Pad to model's fixed sequence length
        padded = token_ids + [self._tokenizer.pad_token_id or 0] * (self._seq_len - num_tokens)
        padded = padded[:self._seq_len]
        input_array = np.array([padded], dtype=np.int32)
        try:
            prediction = self._coreml_model.predict({"input_ids": input_array})
        except RuntimeError:
            if self._pytorch_model is not None:
                return self._infer_pytorch(prompt, max_tokens, 0.0)
            return prompt
        # Find logits output — may be named "logits", "var_NNN", or "output"
        output_val = None
        for key in prediction:
            val = prediction[key]
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                output_val = val
                break
        if output_val is not None:
            # Take logits for the actual token positions (not padding)
            logits = output_val[0, :num_tokens]
            next_token = int(np.argmax(logits[-1]))
            result_ids = token_ids + [next_token]
            return self._tokenizer.decode(result_ids, skip_special_tokens=True)
        return prompt

    def _infer_pytorch(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Run inference via PyTorch model (fallback)."""
        import torch
        tokens = self._tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self._pytorch_model.generate(
                tokens,
                max_new_tokens=min(max_tokens, 50),
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
            "model_loaded": self._coreml_model is not None or self._pytorch_model is not None,
            "tokenizer_loaded": self._tokenizer is not None,
            "stages_completed": self._tern_model.stats.get("stages_completed", []),
        }

    def unload(self):
        """Explicitly unload model and free memory."""
        self._coreml_model = None
        self._pytorch_model = None
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
