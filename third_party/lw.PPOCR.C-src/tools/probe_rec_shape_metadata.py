#!/usr/bin/env python3
"""Probe dynamic REC Shape/Slice metadata with ONNX Runtime.

This is a converter-development aid, not a runtime converter.  It appends
Shape/Slice outputs to a copy of an ONNX graph and records the values observed
at representative input widths.  The resulting report is evidence for a
future static metadata lowering; it must not be treated as LWM support.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.ppocr_contracts import PP_OCRV6_REC_WIDTHS

DEFAULT_WIDTHS = PP_OCRV6_REC_WIDTHS
METADATA_OPS = {"Shape", "Slice"}


def metadata_nodes(model: onnx.ModelProto) -> list[dict[str, Any]]:
    """Return Shape-derived Shape/Slice nodes in graph order.

    REC also uses Slice for ordinary floating-point data.  Those nodes are
    deliberately excluded: only a Slice whose first input is derived from a
    Shape value belongs to the metadata lowering problem.
    """
    shape_values = {
        item.name
        for item in model.graph.initializer
        if item.data_type in (TensorProto.INT32, TensorProto.INT64)
    }
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            inputs = [name for name in node.input if name]
            outputs = [name for name in node.output if name]
            if not outputs:
                continue
            is_shape = node.op_type == "Shape"
            is_shape = is_shape or (
                node.op_type in {"Identity", "Cast"}
                and inputs
                and inputs[0] in shape_values
            )
            is_shape = is_shape or (
                node.op_type in {"Concat", "Slice", "Squeeze", "Unsqueeze", "Reshape"}
                and inputs
                and inputs[0] in shape_values
            )
            if is_shape:
                for output in outputs:
                    if output not in shape_values:
                        shape_values.add(output)
                        changed = True

    result = []
    for index, node in enumerate(model.graph.node):
        if node.op_type == "Slice" and (
            not node.input or node.input[0] not in shape_values
        ):
            continue
        if node.op_type not in METADATA_OPS:
            continue
        result.append({
            "index": index,
            "name": node.name,
            "op": node.op_type,
            "inputs": list(node.input),
            "outputs": [name for name in node.output if name],
        })
    return result


def _input_name_and_shape(model: onnx.ModelProto) -> tuple[str, list[int]]:
    initializers = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    if len(inputs) != 1:
        raise ValueError("REC metadata probe requires exactly one graph input")
    value = inputs[0]
    dimensions = value.type.tensor_type.shape.dim
    if len(dimensions) != 4:
        raise ValueError("REC metadata probe requires a 4-D [N,C,H,W] input")
    shape = [int(dimension.dim_value) if dimension.dim_value > 0 else 1 for dimension in dimensions]
    return value.name, shape


def _instrument(model: onnx.ModelProto, outputs: list[str]) -> onnx.ModelProto:
    instrumented = copy.deepcopy(model)
    existing = {item.name for item in instrumented.graph.output}
    inferred = shape_inference.infer_shapes(instrumented, strict_mode=True, data_prop=False)
    value_info = {
        item.name: item
        for item in (
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        )
    }
    for name in outputs:
        if not name or name in existing:
            continue
        info = value_info.get(name)
        if info is not None:
            instrumented.graph.output.append(copy.deepcopy(info))
        else:
            instrumented.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.INT64, None)
            )
        existing.add(name)
    return instrumented


def probe(model_path: Path, widths: list[int]) -> dict[str, Any]:
    model = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(model)
    nodes = metadata_nodes(model)
    output_names = [name for node in nodes for name in node["outputs"]]
    input_name, base_shape = _input_name_and_shape(model)
    instrumented = _instrument(model, output_names)
    session = ort.InferenceSession(
        instrumented.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    probes: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for width in widths:
        if width <= 0:
            raise ValueError("probe widths must be positive")
        shape = list(base_shape)
        shape[3] = width
        sample = rng.standard_normal(shape, dtype=np.float32)
        values = session.run(output_names, {input_name: sample})
        values_by_name = dict(zip(output_names, values))
        metadata = {
            name: np.asarray(values_by_name[name]).reshape(-1).tolist()
            for name in output_names
        }
        probes.append({"width": width, "input_shape": shape, "metadata": metadata})

    return {
        "schema_version": 1,
        "tool": "tools/probe_rec_shape_metadata.py",
        "model": str(model_path),
        "input": {"name": input_name, "base_shape": base_shape, "widths": widths},
        "metadata_nodes": nodes,
        "probes": probes,
    }


def staticize(model_path: Path, width: int, output_path: Path) -> dict[str, Any]:
    """Replace Shape-derived metadata with constants and verify final outputs."""
    if width <= 0:
        raise ValueError("width must be positive")
    model = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(model)
    nodes = metadata_nodes(model)
    output_names = [name for node in nodes for name in node["outputs"]]
    input_name, base_shape = _input_name_and_shape(model)
    input_shape = list(base_shape)
    input_shape[-1] = width
    sample = np.random.default_rng(0).standard_normal(input_shape, dtype=np.float32)

    instrumented = _instrument(model, output_names)
    metadata_values = dict(zip(
        output_names,
        ort.InferenceSession(
            instrumented.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(output_names, {input_name: sample}),
    ))

    static_model = copy.deepcopy(model)
    metadata_indices = {node["index"] for node in nodes}
    retained_nodes = [
        node for index, node in enumerate(static_model.graph.node)
        if index not in metadata_indices
    ]
    del static_model.graph.node[:]
    static_model.graph.node.extend(retained_nodes)
    existing_initializers = {item.name for item in static_model.graph.initializer}
    for name in output_names:
        if name in existing_initializers:
            raise ValueError(f"metadata output already has an initializer: {name}")
        static_model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(metadata_values[name]), name=name)
        )

    static_model = shape_inference.infer_shapes(static_model)
    onnx.checker.check_model(static_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(static_model, str(output_path))

    original_outputs = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {input_name: sample})
    static_outputs = ort.InferenceSession(
        static_model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {input_name: sample})
    if len(original_outputs) != len(static_outputs):
        raise ValueError("staticized graph changed the number of outputs")
    max_abs_error = 0.0
    for original, static in zip(original_outputs, static_outputs):
        np.testing.assert_allclose(original, static, rtol=1e-4, atol=1e-5)
        if original.size:
            max_abs_error = max(
                max_abs_error,
                float(np.max(np.abs(np.asarray(original) - np.asarray(static)))),
            )

    return {
        "schema_version": 1,
        "tool": "tools/probe_rec_shape_metadata.py",
        "mode": "staticize-and-compare",
        "model": str(model_path),
        "output": str(output_path),
        "input": {"name": input_name, "shape": input_shape},
        "removed_nodes": nodes,
        "materialized_outputs": {
            name: np.asarray(metadata_values[name]).reshape(-1).tolist()
            for name in output_names
        },
        "max_abs_error": max_abs_error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--width", action="append", type=int, dest="widths")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--staticize-output",
        type=Path,
        help="write a fixed-width ONNX prototype with Shape-derived metadata materialized",
    )
    args = parser.parse_args(argv)
    widths = args.widths or list(DEFAULT_WIDTHS)
    report = probe(args.model, widths)
    static_report = None
    if args.staticize_output:
        static_report = staticize(args.model, widths[0], args.staticize_output)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8", newline="\n")
    if static_report:
        static_report_path = (
            args.json_output.with_name(args.json_output.stem + "-static.json")
            if args.json_output
            else args.staticize_output.with_suffix(".json")
        )
        static_report_path.write_text(
            json.dumps(static_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps({
        "metadata_nodes": len(report["metadata_nodes"]),
        "probes": len(report["probes"]),
        "widths": widths,
        "staticized": bool(static_report),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
