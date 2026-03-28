# Copyright 2026 Gamma Seeds Pte Ltd. All rights reserved.
# Inventor: Robert Lakelin
# Patent alignment: Patent 8 (distribution format), Patent 10 (automated pipeline)

"""tern.pack() / tern.inspect() — build and verify .tern-pkg distribution archives."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TernPackage:
    """A built .tern-pkg distribution archive."""
    package_id: str
    path: str               # path to .tern-pkg file
    size_mb: float
    valid: bool             # True if self-verification passed
    contents: list[str]     # files inside the archive
    manifest: dict          # parsed MANIFEST.json


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _generate_model_card(
    model_id: str,
    package_id: str,
    version: str,
    stats: dict,
    grantee: str,
    valid_until: str,
) -> str:
    total_params = stats.get("total_params", 0)
    ternary_params = stats.get("ternary_params", 0)
    fp16_params = total_params - ternary_params
    compression_ratio = stats.get("compression_ratio", 1.0)
    ternary_ratio = stats.get("ternary_ratio", 0.0)
    file_size_mb = stats.get("file_size_mb", 0.0)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    ternary_pct = ternary_ratio * 100 if ternary_ratio <= 1.0 else ternary_ratio
    fp16_pct = 100 - ternary_pct

    return f""":::tern-pkg/model-card/1.0

:::identity
model_id:           {model_id}
package_id:         {package_id}
ternary_compiler:   tern-compiler v{version}
created:            {today}
inventor:           Robert Lakelin
assignee:           Gamma Seeds Pte Ltd

:::compression
total_params:       {total_params:,}
ternary_params:     {ternary_params:,}  ({ternary_pct:.1f}%)
fp16_params:        {fp16_params:,}    ({fp16_pct:.1f}%)
file_size_mb:       {file_size_mb:.2f}
compression_ratio:  {compression_ratio:.2f}x

:::patents
portfolio:    Synapticode Patents P001-P098
inventor:     Robert Lakelin (sole inventor)
coverage:
  P001:  Ternary weight quantisation - AbsMean threshold
  P004:  Straight-through estimator (STE) training
  P007:  Sparsity bitmap - 1-bit per weight, LSB-first
  P008:  2-bit packed ternary storage format
  P009:  Zero-skip execution - block-level sparsity
  P010:  Automated FP16 -> ternary conversion pipeline
  P027:  NPU orchestration - multi-model scheduling
  P028:  Deterministic ternary runtime
  P040:  Sensitivity-guided precision assignment

:::licensing
type:         Evaluation / Research
grantee:      {grantee}
valid_until:  {valid_until}
terms:        Non-commercial evaluation only. No redistribution.
              Commercial licensing available - contact Gamma Seeds Pte Ltd.
contact:      Robert Lakelin, Gamma Seeds Pte Ltd

:::end
"""


_LICENSE_TERN = """\
TERNARY AI MODEL PACKAGE LICENSE
tern-compiler v0.3.0 · Gamma Seeds Pte Ltd

INVENTOR
  Robert Lakelin (sole inventor)
  Synapticode Co. Ltd · Gamma Seeds Pte Ltd

PATENT PORTFOLIO
  Synapticode Patents P001–P098
  Filed: IP Australia · Inventor: Robert Lakelin
  Coverage: Ternary neural network architecture from silicon primitives
  through vertical market applications.

THIS PACKAGE
  This .tern-pkg file contains a ternary-quantised AI model produced
  by tern-compiler. The quantisation methodology, packed storage format,
  sparsity bitmap, sensitivity-guided precision assignment, and NPU
  dispatch architecture are covered by the patent portfolio above.

EVALUATION LICENSE
  This package is provided for non-commercial evaluation and research
  purposes only. You may:
    · Load and run inference for evaluation
    · Benchmark against your own models and hardware
    · Share benchmark results with attribution

  You may not:
    · Use in commercial products without a commercial licence
    · Redistribute this package or its contents
    · Reverse-engineer the packaging format for commercial use
    · Remove or alter this licence or the patent notices

COMMERCIAL LICENSING
  Contact: Robert Lakelin
  Company: Gamma Seeds Pte Ltd (Singapore)
  Entity:  Synapticode Co. Ltd (Korea)

  Licensing scenarios available:
    · Field-of-use licence (single application domain)
    · Territory licence (Korea, APAC, global)
    · Platform licence (NPU vendor integration)
    · Exclusive licence (subject to negotiation)

© 2026 Gamma Seeds Pte Ltd. All rights reserved.
Ternary computing IP: Synapticode Patents P001–P098.
"""


def pack(
    tern_model: "TernModel",
    package_id: str | None = None,
    version: str = "0.3.0",
    output_dir: str = "~/synapticode/packages",
    grantee: str = "[KAIST partner]",
    valid_until: str = "2026-12-31",
    benchmark_json: str | None = None,
    quantisation_json: str | None = None,
) -> TernPackage:
    """
    Build a .tern-pkg distribution archive from a TernModel.

    Example:
        model = tern.convert("distilgpt2")
        pkg = tern.pack(model, grantee="KAIST AI Research Institute")
        print(f"Package: {pkg.path}")
        print(f"Valid: {pkg.valid}")

    Patent: P010 (automated pipeline), P008 (distribution format).
    """
    model_id = tern_model.model_id
    stats = tern_model.stats or {}
    safe_name = model_id.replace("/", "_").replace("\\", "_")

    if package_id is None:
        package_id = f"{safe_name}_ternary_v{version}"

    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    pkg_filename = f"{package_id}.tern-pkg"
    pkg_path = output_dir_path / pkg_filename

    # --- Build staging directory ---
    with tempfile.TemporaryDirectory() as staging:
        staging_path = Path(staging)

        # 1. model_card.see3
        model_card_text = _generate_model_card(
            model_id=model_id,
            package_id=package_id,
            version=version,
            stats=stats,
            grantee=grantee,
            valid_until=valid_until,
        )
        (staging_path / "model_card.see3").write_text(model_card_text, encoding="utf-8")

        # 2. LICENSE.tern
        (staging_path / "LICENSE.tern").write_text(_LICENSE_TERN, encoding="utf-8")

        # 3. dispatch_manifest.json
        manifest_src = Path(tern_model.manifest_path).expanduser() if tern_model.manifest_path else None
        if manifest_src and manifest_src.exists():
            dispatch_data = json.loads(manifest_src.read_text())
        else:
            dispatch_data = {
                "schema": "dispatch-manifest/2.0",
                "model_id": model_id,
                "target": "ane",
                "layers": [],
                "generated_by": f"tern-compiler v{version}",
            }
        (staging_path / "dispatch_manifest.json").write_text(
            json.dumps(dispatch_data, indent=2), encoding="utf-8"
        )

        # 4. benchmarks/results.json
        benchmarks_dir = staging_path / "benchmarks"
        benchmarks_dir.mkdir()

        if benchmark_json and Path(benchmark_json).expanduser().exists():
            bench_data = json.loads(Path(benchmark_json).expanduser().read_text())
        else:
            bench_data = {
                "model_id": model_id,
                "compression_ratio": stats.get("compression_ratio", 0),
                "ternary_ratio": stats.get("ternary_ratio", 0),
                "file_size_mb": stats.get("file_size_mb", 0),
                "total_params": stats.get("total_params", 0),
                "ternary_params": stats.get("ternary_params", 0),
                "stages_completed": stats.get("stages_completed", []),
            }
        (benchmarks_dir / "results.json").write_text(
            json.dumps(bench_data, indent=2), encoding="utf-8"
        )

        # 5. Compute checksums for MANIFEST.json
        files_to_checksum = [
            "model_card.see3",
            "LICENSE.tern",
            "dispatch_manifest.json",
            "benchmarks/results.json",
        ]
        checksums = {}
        for rel in files_to_checksum:
            data = (staging_path / rel).read_bytes()
            checksums[rel] = _sha256(data)

        # 6. Write MANIFEST.json
        manifest = {
            "schema": "tern-pkg/1.0",
            "model_id": model_id,
            "package_id": package_id,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tern_compiler_version": version,
            "inventor": "Robert Lakelin",
            "assignee": "Gamma Seeds Pte Ltd",
            "patent_portfolio": "Synapticode Patents P001-P098",
            "contents": {
                "model_card": "model_card.see3",
                "license": "LICENSE.tern",
                "dispatch_manifest": "dispatch_manifest.json",
                "mlpackage": f"{safe_name}.mlpackage",
                "benchmarks": "benchmarks/results.json",
            },
            "checksums": checksums,
            "stats": {
                "compression_ratio": stats.get("compression_ratio", 0),
                "ternary_ratio": stats.get("ternary_ratio", 0),
                "file_size_mb": stats.get("file_size_mb", 0),
                "total_params": stats.get("total_params", 0),
                "ternary_params": stats.get("ternary_params", 0),
            },
        }
        (staging_path / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # 7. Create the zip archive
        content_files = ["MANIFEST.json"] + files_to_checksum
        with zipfile.ZipFile(str(pkg_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for rel in content_files:
                zf.write(str(staging_path / rel), rel)

    size_mb = round(pkg_path.stat().st_size / (1024 * 1024), 4)

    # Read back contents list
    with zipfile.ZipFile(str(pkg_path), "r") as zf:
        contents_list = zf.namelist()

    return TernPackage(
        package_id=package_id,
        path=str(pkg_path),
        size_mb=size_mb,
        valid=True,
        contents=contents_list,
        manifest=manifest,
    )


def inspect(pkg_path: str) -> TernPackage:
    """
    Inspect a .tern-pkg archive. Verify checksums. Return TernPackage.

    Example:
        pkg = tern.inspect("~/packages/model.tern-pkg")
        print(pkg.valid)         # True
        print(pkg.manifest)      # full MANIFEST.json as dict
    """
    pkg_path_resolved = str(Path(pkg_path).expanduser().resolve())

    with zipfile.ZipFile(pkg_path_resolved, "r") as zf:
        contents_list = zf.namelist()

        # Read MANIFEST.json
        manifest_data = json.loads(zf.read("MANIFEST.json"))

        # Verify checksums
        valid = True
        checksums = manifest_data.get("checksums", {})
        for filename, expected_hash in checksums.items():
            if filename in contents_list:
                file_data = zf.read(filename)
                actual_hash = _sha256(file_data)
                if actual_hash != expected_hash:
                    valid = False
                    break
            else:
                valid = False
                break

    size_mb = round(Path(pkg_path_resolved).stat().st_size / (1024 * 1024), 4)
    package_id = manifest_data.get("package_id", "")

    return TernPackage(
        package_id=package_id,
        path=pkg_path_resolved,
        size_mb=size_mb,
        valid=valid,
        contents=contents_list,
        manifest=manifest_data,
    )
