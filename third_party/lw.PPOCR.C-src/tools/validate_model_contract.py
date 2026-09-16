#!/usr/bin/env python3
"""Validate model manifest identities and REC/dictionary shape contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.validate_model_manifest import validate_manifest


def output_classes(model_path: Path) -> int:
    model = onnx.load(str(model_path), load_external_data=True)
    if len(model.graph.output) != 1:
        raise ValueError("REC model must have exactly one output")
    dimensions = model.graph.output[0].type.tensor_type.shape.dim
    if not dimensions or not dimensions[-1].HasField("dim_value"):
        raise ValueError("REC output class dimension must be static")
    classes = int(dimensions[-1].dim_value)
    if classes <= 0:
        raise ValueError("REC output class dimension must be positive")
    return classes


def shape(value: Any) -> list[int | str]:
    dimensions = value.type.tensor_type.shape.dim
    result: list[int | str] = []
    for dimension in dimensions:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append("?")
    return result


def validate_model_contract(model_dir: Path) -> dict[str, Any]:
    manifest = validate_manifest(model_dir)
    rec_path = model_dir / manifest["models"]["rec"]
    dictionary_path = model_dir / manifest["dictionary"]
    rec_model = onnx.load(str(rec_path), load_external_data=True)
    classes = output_classes(rec_path)
    entries = dictionary_path.read_text(encoding="utf-8").splitlines()
    if classes != len(entries) + 2:
        raise ValueError(
            f"REC classes {classes} do not equal dictionary entries + 2 ({len(entries) + 2})"
        )
    return {
        "status": "ok",
        "schema_version": manifest["schema_version"],
        "family": manifest["family"],
        "variant": manifest["variant"],
        "runtime_status": manifest["runtime_status"],
        "assets": dict(manifest["checksums"]),
        "rec_output_shape": shape(rec_model.graph.output[0]),
        "rec_classes": classes,
        "dictionary_entries": len(entries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_model_contract(args.model_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
