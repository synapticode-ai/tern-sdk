# tern-sdk

**Synapticode Developer SDK**
CNS Synaptic™ by Synapticode

The developer interface for ternary AI deployment. Convert models, deploy to NPU hardware, and integrate ternary inference into your applications.

## Overview

tern-sdk is the public entry point to the Synapticode ternary stack. It provides APIs for model conversion, deployment, inference, and governance — without exposing the internal architecture of the core engine, compiler, or runtime.

```
Your Application
      │
      ▼
┌─────────────┐
│   tern-sdk  │  ← You are here
├─────────────┤
│  (internal) │  tern-runtime, tern-compiler, tern-core
└─────────────┘
      │
      ▼
   NPU Hardware
```

## Quick Start

```python
import tern

# Convert a model to ternary
model = tern.convert("meta-llama/Llama-3-8B", target="atom")

# Check compression stats
print(model.stats)
# → TernModel: 70MB (was 13.4GB), 192x compression, 97.2% accuracy retained

# Deploy to NPU
runtime = tern.deploy(model, device="npu:0")

# Run inference
output = runtime.infer("Explain ternary computing in one sentence.")
# → "Ternary computing replaces multiplication with comparison by reducing
#    every neural network weight to one of three values."

# Governance: get decision trace
trace = runtime.explain(output)
print(trace.top_contributors)
# → [Layer 12 Attn +1 weights: 847, Layer 14 FFN -1 weights: 312, ...]
```

## API Reference

### tern.convert()

Convert a standard model to ternary format.

```python
tern.convert(
    model,              # Model path, HuggingFace ID, or loaded model object
    target="atom",      # Target NPU: "atom", "hexagon", "tnpu", "generic"
    sensitivity=0.95,   # Accuracy retention target (0.0–1.0)
    progressive=False,  # Enable multi-level compression
)
# Returns: TernModel
```

### tern.deploy()

Deploy a ternary model to hardware.

```python
tern.deploy(
    model,              # TernModel from convert() or .tern-model path
    device="npu:0",     # Device identifier
    replicas=1,         # Number of model replicas for throughput
    deterministic=True, # Enforce deterministic execution
)
# Returns: TernRuntime
```

### TernRuntime.infer()

Run inference on deployed model.

```python
runtime.infer(
    input,              # Text, image, or multimodal input
    max_tokens=256,     # Maximum output tokens
    temperature=0.0,    # Sampling temperature (0.0 = deterministic)
)
# Returns: TernOutput
```

### TernRuntime.explain()

Get deterministic decision trace for any output.

```python
runtime.explain(
    output,             # TernOutput from infer()
    depth="layer",      # Trace depth: "layer", "weight", "full"
)
# Returns: TernTrace
```

## Architecture

```
tern-sdk/
├── src/
│   ├── api/             # Public API surface
│   ├── examples/        # Integration examples, tutorials
│   └── bindings/        # Language bindings (Python, C++, Rust, JS)
├── tests/               # SDK integration tests
└── docs/
    ├── guides/          # Getting started, tutorials, recipes
    └── api-reference/   # Complete API documentation
```

## Language Support

| Language | Status | Package |
|---|---|---|
| Python | Primary | `pip install tern` |
| C++ | Planned | `libtern` |
| Rust | Planned | `tern-rs` |
| JavaScript | Planned | `@synapticode/tern` |

## Hardware Support

| NPU | SDK Integration | Status |
|---|---|---|
| Rebellions ATOM | ATOM SDK | Primary target |
| Qualcomm Hexagon | Hexagon SDK | Fallback |
| Synapticode TNPU | Native | Future (Patent 27) |
| Generic CPU | SIMD optimised | Development/testing |

## Examples

See `src/examples/` for complete integration examples:

- `basic_inference.py` — Convert and run a model in 5 lines
- `multi_npu.py` — Scale inference across multiple NPUs
- `governance_audit.py` — Generate compliance reports
- `edge_deploy.py` — Deploy to edge device via CNS Edge
- `streaming.py` — Streaming inference for generative models
- `benchmark.py` — Compare ternary vs FP16 performance

## Licence

Apache 2.0 — SDK and examples.
Ternary core engine, compiler, and runtime are proprietary.

## Links

- Website: [synapticode.ai](https://synapticode.ai)
- Korea: [synapticode.io](https://synapticode.io)
- Contact: partnerships@synapticode.ai (NDA required for core technology access)

---

CNS Synaptic™ by Synapticode
