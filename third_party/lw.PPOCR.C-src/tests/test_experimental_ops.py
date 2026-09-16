from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from converter.lwm_v0 import OP_IDS, _materialize_slice_inputs, _write_model
from tools.convert_small_rec_experimental import lower_dynamic_metadata


class ExperimentalOperatorEncodingTests(unittest.TestCase):
    def test_small_dynamic_lowering_removes_only_reshape_metadata(self) -> None:
        graph = helper.make_graph(
            [
                helper.make_node("Shape", ["x"], ["shape"], name="shape"),
                helper.make_node(
                    "Reshape", ["x", "shape"], ["y"], name="reshape"
                ),
            ],
            "dynamic-lowering",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, "W"])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, "W"])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10
        lowered = lower_dynamic_metadata(model)
        self.assertEqual([node.op_type for node in lowered.graph.node], ["Reshape"])
        self.assertEqual(list(lowered.graph.node[0].input), ["x"])

    def test_small_dynamic_lowering_rejects_numeric_metadata_use(self) -> None:
        graph = helper.make_graph(
            [
                helper.make_node("Shape", ["x"], ["shape"], name="shape"),
                helper.make_node("Add", ["shape", "shape"], ["y"], name="add"),
            ],
            "dynamic-lowering-reject",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, "W"])],
            [helper.make_tensor_value_info("y", TensorProto.INT64, [4])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10
        with self.assertRaisesRegex(ValueError, "numeric operator Add"):
            lower_dynamic_metadata(model)

    def test_sub_pow_sqrt_are_encoded_as_zero_parameter_nodes(self) -> None:
        graph = helper.make_graph(
            [
                helper.make_node("Sub", ["x", "bias"], ["sub"], name="sub"),
                helper.make_node("Pow", ["sub", "exponent"], ["pow"], name="pow"),
                helper.make_node("Sqrt", ["pow"], ["y"], name="sqrt"),
            ],
            "experimental-ops",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
            initializer=[
                helper.make_tensor("bias", TensorProto.FLOAT, [1], [1.0]),
                helper.make_tensor("exponent", TensorProto.FLOAT, [1], [2.0]),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        model.ir_version = 10
        inferred = onnx.shape_inference.infer_shapes(model)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "experimental.lwm"
            info = _write_model(model, output, inferred)
            encoded = output.read_bytes()

        self.assertEqual(info.node_count, 3)
        node_offset = struct.unpack_from("<4sHH6I13Q3Q", encoded)[12]
        ops = [
            struct.unpack_from("<H", encoded, node_offset + index * 72)[0]
            for index in range(info.node_count)
        ]
        self.assertEqual(ops, [OP_IDS["Sub"], OP_IDS["Pow"], OP_IDS["Sqrt"]])
        for index in range(info.node_count):
            self.assertEqual(struct.unpack_from("<I", encoded, node_offset + index * 72 + 64)[0], 0)

    def test_slice_uses_fixed_positive_step_parameter_record(self) -> None:
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Slice", ["x"], ["y"], starts=[0], ends=[2], axes=[1], steps=[1]
                )
            ],
            "slice",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        model.ir_version = 10
        inferred = onnx.shape_inference.infer_shapes(model)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "slice.lwm"
            info = _write_model(model, output, inferred)
            encoded = output.read_bytes()
        self.assertEqual(info.node_count, 1)
        node_offset = struct.unpack_from("<4sHH6I13Q3Q", encoded)[12]
        self.assertEqual(struct.unpack_from("<H", encoded, node_offset)[0], OP_IDS["Slice"])
        self.assertEqual(struct.unpack_from("<I", encoded, node_offset + 64)[0], 136)

    def test_slice_control_initializers_are_materialized(self) -> None:
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Slice", ["x", "starts", "ends", "axes", "steps"], ["y"], name="slice"
                )
            ],
            "slice-inputs",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
            initializer=[
                helper.make_tensor("starts", TensorProto.INT64, [1], [0]),
                helper.make_tensor("ends", TensorProto.INT64, [1], [2]),
                helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
                helper.make_tensor("steps", TensorProto.INT64, [1], [1]),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        model.ir_version = 10
        converted = _materialize_slice_inputs(model)
        node = converted.graph.node[0]
        self.assertEqual(list(node.input), ["x", "starts", "ends", "axes", "steps"])
        attrs = {attribute.name: list(attribute.ints) for attribute in node.attribute}
        self.assertEqual(attrs["starts"], [0])
        self.assertEqual(attrs["ends"], [2])


if __name__ == "__main__":
    unittest.main()
