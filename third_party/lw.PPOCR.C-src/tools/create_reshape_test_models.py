#!/usr/bin/env python3
"""Create tiny LWM fixtures for direct runtime Reshape contract tests."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.lwm_v0 import _write_model


FIXTURES = {
    "valid": [2, -1, 2],
    "two-unknown": [-1, -1],
    "non-divisible": [5, -1],
    "zero": [0, -1],
    "overflow": [2147483647, 2147483647, 2147483647, 2147483647],
}


def make_fixture(shape: list[int], output: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Reshape", ["x", "target"], ["y"], name="reshape")],
        "dynamic-reshape-contract",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 6, 2])],
        initializer=[helper.make_tensor("target", TensorProto.INT64, [len(shape)], shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    inferred = copy.deepcopy(model)
    dimensions = inferred.graph.output[0].type.tensor_type.shape.dim
    del dimensions[:]
    for value in shape:
        dimension = dimensions.add()
        if value > 0:
            dimension.dim_value = value
        elif value == 0:
            dimension.dim_value = 0
        else:
            dimension.dim_param = "inferred"
    _write_model(model, output, inferred)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, shape in FIXTURES.items():
        make_fixture(shape, args.output_dir / f"{name}.lwm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
