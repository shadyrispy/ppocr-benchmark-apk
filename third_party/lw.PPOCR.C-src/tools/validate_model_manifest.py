#!/usr/bin/env python3
"""Validate a PP-OCR model-package manifest and its asset hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_STATUSES = {"primary", "supported", "analysis-only"}


def _safe_relative(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must stay inside the model directory")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "model.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"unable to read {manifest_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("model manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("model manifest schema_version must be 1")
    if manifest.get("family") != "PP-OCRv6":
        raise ValueError("model manifest family must be PP-OCRv6")
    if not isinstance(manifest.get("variant"), str) or not manifest["variant"]:
        raise ValueError("model manifest variant must be non-empty")
    if manifest.get("runtime_status") not in RUNTIME_STATUSES:
        raise ValueError("model manifest runtime_status is invalid")

    models = manifest.get("models")
    if not isinstance(models, dict) or set(models) != {"det", "cls", "rec"}:
        raise ValueError("model manifest models must contain det, cls, and rec")
    asset_paths = [_safe_relative(models[name], f"models.{name}") for name in ("det", "cls", "rec")]
    asset_paths.append(_safe_relative(manifest.get("dictionary"), "dictionary"))

    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict) or set(checksums) != set(asset_paths):
        raise ValueError("checksums must cover every model and dictionary asset exactly once")
    for relative in asset_paths:
        expected = checksums.get(relative)
        if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
            raise ValueError(f"invalid SHA-256 for {relative}")
        asset = model_dir / relative
        if not asset.is_file():
            raise ValueError(f"manifest asset is missing: {asset}")
        actual = _sha256(asset)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args(argv)
    manifest = validate_manifest(args.model_dir)
    print(json.dumps({
        "family": manifest["family"],
        "variant": manifest["variant"],
        "runtime_status": manifest["runtime_status"],
        "assets": len(manifest["checksums"]),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
