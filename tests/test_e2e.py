# Copyright 2026 Gamma Seeds Pte Ltd. All rights reserved.
# Inventor: Robert Lakelin

"""End-to-end tests for the tern SDK."""

from pathlib import Path


def test_full_pipeline_distilgpt2():
    """
    Full tern-core -> tern-compiler -> tern-runtime -> tern-sdk pipeline.
    Uses distilgpt2 (small, fast). Verifies each stage output is non-null.
    Target: completes in under 120 seconds on M4 Pro.
    """
    import tern

    # Convert
    model = tern.convert("distilgpt2", verify=True)
    assert model.stats["compression_ratio"] >= 4.0
    assert model.stats["ternary_ratio"] >= 0.70
    assert Path(model.mlpackage_path).exists()

    # Deploy
    runtime = tern.deploy(model)

    # Infer
    output = runtime.infer("Ternary computing uses")
    assert isinstance(output.text, str)
    assert len(output.text) > 0
    assert output.latency_ms > 0
    assert output.device in ("ANE", "GPU", "CPU")

    # Health
    health = runtime.health()
    assert health["is_healthy"] is True

    # Clean up
    runtime.unload()


def test_stats_output():
    """Compression stats are logged correctly."""
    import tern

    model = tern.convert("distilgpt2", verify=False)
    assert "compression_ratio" in model.stats
    assert "ternary_ratio" in model.stats
    assert "file_size_mb" in model.stats
    assert model.compile_time_s > 0
