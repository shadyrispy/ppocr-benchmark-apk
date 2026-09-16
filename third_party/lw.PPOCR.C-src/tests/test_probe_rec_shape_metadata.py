from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from tools.probe_rec_shape_metadata import metadata_nodes, staticize
from tools.validate_small_rec_dynamic_rule import validate_report


class RecShapeMetadataProbeTests(unittest.TestCase):
    def test_collects_shape_and_slice_nodes_in_graph_order(self) -> None:
        graph = helper.make_graph(
            [
                helper.make_node("Shape", ["x"], ["shape"], name="shape-node"),
                helper.make_node("Slice", ["shape"], ["slice"], name="slice-node"),
                helper.make_node("Identity", ["slice"], ["y"], name="identity-node"),
            ],
            "probe",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 48, "W"])],
            [helper.make_tensor_value_info("y", TensorProto.INT64, [None])],
        )
        nodes = metadata_nodes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
        self.assertEqual([node["op"] for node in nodes], ["Shape", "Slice"])
        self.assertEqual([node["name"] for node in nodes], ["shape-node", "slice-node"])
        self.assertEqual(nodes[0]["outputs"], ["shape"])

    def test_staticize_replaces_shape_and_preserves_output(self) -> None:
        graph = helper.make_graph(
            [
                helper.make_node("Shape", ["x"], ["shape"], name="shape-node"),
                helper.make_node("Reshape", ["x", "shape"], ["y"], name="reshape-node"),
            ],
            "staticize",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 2])],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.onnx"
            output_path = root / "static.onnx"
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = 10
            onnx.save(model, model_path)
            report = staticize(model_path, 2, output_path)
            self.assertEqual(report["removed_nodes"][0]["name"], "shape-node")
            self.assertEqual(report["materialized_outputs"]["shape"], [1, 1, 1, 2])
            self.assertEqual(report["max_abs_error"], 0.0)
            self.assertTrue(output_path.is_file())

    def test_small_dynamic_rule_accepts_all_required_widths(self) -> None:
        widths = [192, 320, 480, 640, 960]
        report = {
            "schema_version": 1,
            "input": {"base_shape": [1, 3, 48, 1], "widths": widths},
            "metadata_nodes": [
                {"op": "Shape", "outputs": [name]}
                for name in ("Shape.1", "Shape.3", "Shape.7", "Shape.13")
            ]
            + [
                {"op": "Slice", "outputs": [name]}
                for name in ("Slice.1", "helper.slice.0")
            ],
            "probes": [],
        }
        for width in widths:
            report["probes"].append(
                {
                    "width": width,
                    "input_shape": [1, 3, 48, width],
                    "metadata": {
                        "Shape.1": [1, 120, 1, width // 8],
                        "Shape.3": [1, 120, 1, width // 8],
                        "Shape.7": [1, 8, width // 8, 15],
                        "Shape.13": [1, 8, width // 8, 15],
                        "Slice.1": [width // 8],
                        "helper.slice.0": [1, 120],
                    },
                }
            )
        summary = validate_report(report)
        self.assertEqual(summary["status"], "validated-analysis-only")
        self.assertEqual(summary["probed_widths"], widths)

    def test_small_dynamic_rule_rejects_wrong_ratio(self) -> None:
        widths = [192, 320, 480, 640, 960]
        report = {
            "schema_version": 1,
            "input": {"base_shape": [1, 3, 48, 1], "widths": widths},
            "metadata_nodes": [
                {"op": "Shape", "outputs": [name]}
                for name in ("Shape.1", "Shape.3", "Shape.7", "Shape.13")
            ]
            + [
                {"op": "Slice", "outputs": [name]}
                for name in ("Slice.1", "helper.slice.0")
            ],
            "probes": [],
        }
        for width in widths:
            report["probes"].append(
                {
                    "width": width,
                    "input_shape": [1, 3, 48, width],
                    "metadata": {
                        "Shape.1": [1, 120, 1, width // 8],
                        "Shape.3": [1, 120, 1, width // 8],
                        "Shape.7": [1, 8, width // 8 + (1 if width == 320 else 0), 15],
                        "Shape.13": [1, 8, width // 8, 15],
                        "Slice.1": [width // 8],
                        "helper.slice.0": [1, 120],
                    },
                }
            )
        with self.assertRaisesRegex(ValueError, "probe 320 output Shape.7"):
            validate_report(report)


if __name__ == "__main__":
    unittest.main()
