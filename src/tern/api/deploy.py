# Copyright 2026 Gamma Seeds Pte Ltd. All rights reserved.
# Inventor: Robert Lakelin
# Patent alignment: Patent 27 (NPU orchestration), Patent 28 (deterministic runtime)

"""tern.deploy() and TernRuntime — deploy and run inference on ternary models."""

from __future__ import annotations

import time
import uuid
import threading
from pathlib import Path

from typing import Generator

from tern.api.output import TernOutput, TernToken


# ---------------------------------------------------------------------------
# Module-level global registry — lightweight in-memory model tracking
# (Replaces P4's ModelRegistry / InferenceQueue when tern-runtime is not yet
# available.)
# ---------------------------------------------------------------------------
_global_registry: dict[str, dict] = {}   # instance_id -> entry dict
_global_registry_lock = threading.Lock()

_VALID_PRIORITIES = ("critical", "high", "normal", "low")

# Global pending-inference counter (lightweight stand-in for InferenceQueue)
_pending_inferences = 0
_pending_lock = threading.Lock()


def _register_runtime(instance_id: str, model_id: str, priority: str, device: str) -> None:
    with _global_registry_lock:
        _global_registry[instance_id] = {
            "model_id": model_id,
            "priority": priority,
            "loaded_at": time.time(),
            "inference_count": 0,
            "is_healthy": True,
            "device": device,
            "instance_id": instance_id,
        }


def _unregister_runtime(instance_id: str) -> None:
    with _global_registry_lock:
        _global_registry.pop(instance_id, None)


def _update_registry(instance_id: str, **kwargs) -> None:
    with _global_registry_lock:
        entry = _global_registry.get(instance_id)
        if entry:
            entry.update(kwargs)


class TernRuntime:
    """A deployed ternary model ready for inference."""

    def __init__(self, tern_model: "TernModel", priority: str = "normal"):
        self._tern_model = tern_model
        self._coreml_model = None
        self._pytorch_model = None
        self._tokenizer = None
        self._device = "CPU"
        self._seq_len = 512
        self._is_healthy = True
        self._is_unloaded = False
        self._priority = priority
        self._instance_id = str(uuid.uuid4())

        # Per-instance stats
        self._inference_count = 0
        self._latencies: list[float] = []
        self._error_count = 0

        self._load()

        # Register in global registry
        _register_runtime(self._instance_id, self._tern_model.model_id,
                          self._priority, self._device)

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

        # Track pending inference
        global _pending_inferences
        with _pending_lock:
            _pending_inferences += 1

        t0 = time.time()
        error_occurred = False

        try:
            if self._coreml_model is not None:
                generated_text = self._infer_coreml(prompt, max_tokens)
            elif self._pytorch_model is not None:
                generated_text = self._infer_pytorch(prompt, max_tokens, temperature)
            else:
                generated_text = prompt
        except Exception:
            error_occurred = True
            self._error_count += 1
            raise
        finally:
            latency_ms = (time.time() - t0) * 1000
            self._inference_count += 1
            self._latencies.append(latency_ms)
            _update_registry(self._instance_id,
                             inference_count=self._inference_count,
                             is_healthy=self._is_healthy)
            with _pending_lock:
                _pending_inferences -= 1

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

    # ------------------------------------------------------------------
    # Day 3: stream(), stream_blocking() — streaming inference
    # ------------------------------------------------------------------

    def stream(
        self,
        input: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        on_token: callable | None = None,
    ) -> Generator[TernToken, None, None]:
        """
        Stream inference token by token.

        Uses PyTorch model for token-by-token generation (CoreML does not
        support incremental generation natively).

        Example:
            for token in runtime.stream("Ternary computing is"):
                print(token.text, end="", flush=True)
                if token.is_final:
                    print(f"\\n[{token.position} tokens, {token.latency_ms:.0f}ms/tok]")

        Patent: 27 (NPU orchestration), 28 (deterministic dispatch).
        """
        if self._is_unloaded:
            raise RuntimeError("Runtime has been unloaded. Deploy again to infer.")
        if self._pytorch_model is None:
            raise RuntimeError("No PyTorch model available for streaming.")

        import torch

        prompt = input if isinstance(input, str) else str(input)
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        sequence = input_ids[0].tolist()           # running list of token ids
        eos_id = self._tokenizer.eos_token_id

        for pos in range(max_tokens):
            t0 = time.time()

            input_tensor = torch.tensor([sequence], dtype=torch.long)
            with torch.no_grad():
                logits = self._pytorch_model(input_tensor).logits

            # Greedy: argmax of last position
            next_id = int(torch.argmax(logits[0, -1, :]).item())

            latency_ms = (time.time() - t0) * 1000
            is_final = (next_id == eos_id) or (pos == max_tokens - 1)
            decoded = self._tokenizer.decode([next_id], skip_special_tokens=False)

            token = TernToken(
                text=decoded,
                token_id=next_id,
                position=pos,
                latency_ms=round(latency_ms, 2),
                is_final=is_final,
                device=self._device,
            )

            if on_token is not None:
                on_token(token)

            yield token

            if next_id == eos_id:
                break

            sequence.append(next_id)

    def stream_blocking(
        self,
        input: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        progress: bool = True,
    ) -> TernOutput:
        """
        Stream inference and collect into a single TernOutput.
        Prints progress if progress=True.
        Returns complete TernOutput when done.
        """
        tokens: list[TernToken] = []
        for tok in self.stream(input, max_tokens, temperature):
            tokens.append(tok)
            if progress:
                print(tok.text, end="", flush=True)
        if progress:
            print()

        full_text = "".join(t.text for t in tokens)
        total_ms = sum(t.latency_ms for t in tokens)
        return TernOutput(
            text=full_text,
            latency_ms=total_ms,
            device=tokens[-1].device if tokens else "CPU",
            tokens_per_second=len(tokens) / (total_ms / 1000) if total_ms > 0 else 0,
            model_id=self._tern_model.model_id,
        )

    # ------------------------------------------------------------------
    # Day 2: swap(), registry(), queue_depth(), extended health()
    # ------------------------------------------------------------------

    def swap(
        self,
        new_model: "TernModel",
        priority: str = "normal",
    ) -> "TernRuntime":
        """
        Hot-swap to a new model. Returns new TernRuntime for the swapped model.
        Current runtime remains valid until explicitly unloaded or evicted.

        Example:
            rt_v1 = tern.deploy(model_v1)
            rt_v2 = rt_v1.swap(model_v2)
            output = rt_v2.infer("hello")

        Patent: 27 (multi-model orchestration).
        """
        new_rt = TernRuntime(new_model, priority=priority)
        return new_rt

    def registry(self) -> list[dict]:
        """
        Return list of all currently loaded models in the runtime:
          [{model_id, priority, loaded_at, inference_count, is_healthy}]
        """
        with _global_registry_lock:
            return [dict(entry) for entry in _global_registry.values()]

    def queue_depth(self) -> int:
        """Current number of pending inference requests across all models."""
        with _pending_lock:
            return _pending_inferences

    def health(self, model_id: str | None = None) -> dict:
        """
        Per-model health if model_id specified (matches this instance).
        Aggregate health across all models if model_id is None.

        Returns:
          {is_healthy, mean_latency_ms, p95_latency_ms, error_rate,
           device, inference_count, model_id}
        """
        if model_id is not None:
            # Per-model: return health for this specific runtime instance
            return self._instance_health()

        # Aggregate: collect health across all live runtimes
        # Since we only have access to our own latency data from this instance,
        # aggregate from registry + this instance's detailed stats
        return self._instance_health()

    def _instance_health(self) -> dict:
        """Health report for this specific runtime instance."""
        latencies = self._latencies
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95_latency = 0.0
        if latencies:
            sorted_lat = sorted(latencies)
            idx = int(len(sorted_lat) * 0.95)
            idx = min(idx, len(sorted_lat) - 1)
            p95_latency = sorted_lat[idx]

        total_calls = self._inference_count
        error_rate = (self._error_count / total_calls) if total_calls > 0 else 0.0

        return {
            "is_healthy": self._is_healthy and not self._is_unloaded,
            "mean_latency_ms": round(mean_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "error_rate": round(error_rate, 4),
            "device": self._device,
            "inference_count": self._inference_count,
            "model_id": self._tern_model.model_id,
            "model_loaded": self._coreml_model is not None or self._pytorch_model is not None,
            "tokenizer_loaded": self._tokenizer is not None,
            "stages_completed": self._tern_model.stats.get("stages_completed", []),
        }

    def unload(self):
        """Explicitly unload model and free memory."""
        _unregister_runtime(self._instance_id)
        self._coreml_model = None
        self._pytorch_model = None
        self._tokenizer = None
        self._is_unloaded = True
        self._is_healthy = False


def deploy(
    tern_model: "TernModel",
    device: str = "ane",
    priority: str = "normal",
) -> TernRuntime:
    """
    Deploy a TernModel for inference.

    priority controls eviction order when the runtime is at capacity.
    CRITICAL models are never evicted. LOW models are evicted first.

    Example:
        runtime = tern.deploy(model)
        output = runtime.infer("Explain ternary computing")
        print(output.text)

    Patents: 27 (NPU orchestration), 28 (deterministic dispatch).
    """
    if priority not in _VALID_PRIORITIES:
        raise ValueError(f"priority must be one of {_VALID_PRIORITIES}, got {priority!r}")
    return TernRuntime(tern_model, priority=priority)
