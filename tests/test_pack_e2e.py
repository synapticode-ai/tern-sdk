# Copyright 2026 Gamma Seeds Pte Ltd. All rights reserved.
# Inventor: Robert Lakelin
# Day 4 — P5 SDK Integration: pack/inspect end-to-end tests

"""End-to-end tests for tern.pack() and tern.inspect()."""


def test_pack_and_inspect_pipeline():
    """Full pipeline: convert → pack → inspect → verify."""
    import tern
    from pathlib import Path

    # Convert
    model = tern.convert("distilgpt2", verify=False)

    # Pack
    pkg = tern.pack(
        model,
        package_id="distilgpt2_ternary_test",
        version="0.3.0",
        output_dir="/tmp",
        grantee="KAIST Test",
        valid_until="2026-12-31",
    )

    assert pkg.valid, f"Package invalid: {pkg}"
    assert Path(pkg.path).exists()
    assert pkg.size_mb > 0
    assert "MANIFEST.json" in pkg.contents
    assert "model_card.see3" in pkg.contents
    assert "LICENSE.tern" in pkg.contents

    # Inspect the built package
    verified = tern.inspect(pkg.path)
    assert verified.valid
    assert verified.manifest["schema"] == "tern-pkg/1.0"
    assert verified.manifest["inventor"] == "Robert Lakelin"
    assert verified.manifest["assignee"] == "Gamma Seeds Pte Ltd"


def test_inspect_reports_model_stats():
    """inspect() manifest contains compression stats."""
    import tern
    model = tern.convert("distilgpt2", verify=False)
    pkg = tern.pack(model, output_dir="/tmp", grantee="Test")
    verified = tern.inspect(pkg.path)
    assert "compression_ratio" in verified.manifest.get("stats", {})


def test_package_is_standard_zip():
    """Any zip tool can open a .tern-pkg file."""
    import tern, zipfile
    model = tern.convert("distilgpt2", verify=False)
    pkg = tern.pack(model, output_dir="/tmp", grantee="Test")
    assert zipfile.is_zipfile(pkg.path)
    with zipfile.ZipFile(pkg.path) as zf:
        assert len(zf.namelist()) >= 5
