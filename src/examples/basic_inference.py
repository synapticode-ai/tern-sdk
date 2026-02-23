"""
Basic Inference Example
tern-sdk by Synapticode

Convert a model to ternary and run inference in 5 lines.
"""

import tern

# 1. Convert a model to ternary format
model = tern.convert("meta-llama/Llama-3-8B", target="atom")

# 2. Check compression statistics
print(model.stats)
# Expected: TernModel: 70MB (was 13.4GB), 192x compression, 97.2% accuracy retained

# 3. Deploy to NPU
runtime = tern.deploy(model, device="npu:0")

# 4. Run inference
output = runtime.infer("Explain ternary computing in one sentence.")
print(output.text)

# 5. Get decision trace (governance)
trace = runtime.explain(output)
print(trace.top_contributors)
