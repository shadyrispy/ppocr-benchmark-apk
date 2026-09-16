#!/usr/bin/env python3
"""Compare a candidate CLS ONNX asset with the bundled Tiny CLS contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shape(value: Any) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append("?")
    return result


def inspect(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=True)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "ir_version": model.ir_version,
        "opset": [
            {"domain": item.domain, "version": item.version}
            for item in model.opset_import
        ],
        "inputs": [
            {"name": value.name, "shape": shape(value), "dtype": value.type.tensor_type.elem_type}
            for value in model.graph.input
        ],
        "outputs": [
            {"name": value.name, "shape": shape(value), "dtype": value.type.tensor_type.elem_type}
            for value in model.graph.output
        ],
        "node_count": len(model.graph.node),
        "op_counts": {
            op: sum(1 for node in model.graph.node if node.op_type == op)
            for op in sorted({node.op_type for node in model.graph.node})
        },
    }


def compare(reference: Path, candidate: Path) -> dict[str, Any]:
    reference_info = inspect(reference)
    candidate_info = inspect(candidate)
    exact_bytes = reference_info["sha256"] == candidate_info["sha256"]
    structural_match = all(
        reference_info[key] == candidate_info[key]
        for key in ("ir_version", "opset", "inputs", "outputs", "node_count", "op_counts")
    )
    return {
        "status": "ok",
        "shared_asset": exact_bytes,
        "structural_match": structural_match,
        "reference": reference_info,
        "candidate": candidate_info,
        "recommendation": (
            "reuse-tiny-cls-contract"
            if exact_bytes
            else "candidate-requires-independent-cls-conversion-and-gates"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cls", type=Path, required=True)
    parser.add_argument("--candidate-cls", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = compare(args.reference_cls, args.candidate_cls)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
