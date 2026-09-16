from __future__ import annotations

import unittest
from pathlib import Path

from converter.ppocr_contracts import (
    PP_OCRV6_REC_WIDTHS,
    PP_OCRV6_SMALL_REC_BASE_SHAPE,
    PP_OCRV6_SMALL_CLS_SHARED,
    PP_OCRV6_TINY_CLS_SHA256,
    SMALL_REC_DYNAMIC_OUTPUTS,
    small_rec_metadata,
)
from tools.run_ocr_scene_suite import parse_output
from tools.run_small_validation import recognized_text_sha256
from tools.compare_rec_accuracy import edit_distance, normalize
from tools.resolve_small_validation_contract import resolve_contract


ROOT = Path(__file__).resolve().parents[1]
from tools.validate_small_scene_baseline import compare_reports
from tools.validate_small_rec_dynamic_rule import validate_report


def valid_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "input": {"base_shape": list(PP_OCRV6_SMALL_REC_BASE_SHAPE), "widths": list(PP_OCRV6_REC_WIDTHS)},
        "metadata_nodes": [
            {"op": "Shape", "outputs": [name]}
            for name in SMALL_REC_DYNAMIC_OUTPUTS
        ],
        "probes": [
            {
                "width": width,
                "input_shape": [1, 3, 48, width],
                "metadata": small_rec_metadata(width),
            }
            for width in PP_OCRV6_REC_WIDTHS
        ],
    }


class SmallContractTests(unittest.TestCase):
    def test_small_profile_reuses_pinned_tiny_cls(self) -> None:
        self.assertTrue(PP_OCRV6_SMALL_CLS_SHARED)
        self.assertEqual(len(PP_OCRV6_TINY_CLS_SHA256), 64)

    def test_dynamic_metadata_contract_accepts_all_five_widths(self) -> None:
        summary = validate_report(valid_report())
        self.assertEqual(summary["required_widths"], list(PP_OCRV6_REC_WIDTHS))
        self.assertEqual(summary["status"], "validated-analysis-only")

    def test_dynamic_metadata_contract_rejects_wrong_width_rule(self) -> None:
        report = valid_report()
        probes = report["probes"]
        assert isinstance(probes, list)
        probes[0]["metadata"]["Shape.1"] = [1, 120, 1, 25]
        with self.assertRaisesRegex(ValueError, "output Shape.1"):
            validate_report(report)

    def test_scene_header_separates_image_and_detector_dimensions(self) -> None:
        header, lines = parse_output(
            "config rec_max_width=960 adaptive=yes\n"
            "lines=1 image=2200x900 detector_input=960x384\n"
            "0 text=示例 rec=0.9 det=0.8 cls=0/1.0 rotate=0 "
            "[(1,2),(3,2),(3,4),(1,4)]\n"
        )
        self.assertEqual(header["image_width"], 2200)
        self.assertEqual(header["image_height"], 900)
        self.assertEqual(header["detector_width"], 960)
        self.assertEqual(header["detector_height"], 384)
        self.assertEqual(lines[0]["text"], "示例")

    def test_full_ocr_text_checksum_is_newline_joined_utf8(self) -> None:
        output = (
            "lines=2 image=10x10 detector_input=10x10\n"
            "0 text=第一行 rec=1 det=1 cls=0/1 rotate=0 [(0,0),(1,0),(1,1),(0,1)]\n"
            "1 text=第二行 rec=1 det=1 cls=0/1 rotate=0 [(0,2),(1,2),(1,3),(0,3)]\n"
        )
        import hashlib

        expected = hashlib.sha256("第一行\n第二行".encode("utf-8")).hexdigest()
        self.assertEqual(recognized_text_sha256(output), expected)

    def test_scene_baseline_catches_text_regression(self) -> None:
        line = {
            "index": 0,
            "text": "示例",
            "box": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.0, 4.0],
            "rotation": 0,
            "det_score": 0.9,
            "rec_score": 0.8,
            "cls_score": 1.0,
        }
        report = {
            "schema_version": 1,
            "manifest_sha256": "a" * 64,
            "rec_max_width": 960,
            "models": {"detector": {"sha256": "b" * 64}},
            "scenes": [{
                "name": "scene",
                "status": "ok",
                "width": 10,
                "height": 10,
                "expected_source_lines": 1,
                "detected_lines": 1,
                "recognized_text": ["示例"],
                "rotations": [0],
                "lines": [line],
            }],
        }
        relocated = {**report, "models": {"detector": {"path": "/runner/work/build/det.lwm", "sha256": "b" * 64}}}
        self.assertEqual(compare_reports(report, relocated)["status"], "ok")
        changed = {**report, "scenes": [{**report["scenes"][0], "recognized_text": ["变更"]}]}
        with self.assertRaisesRegex(ValueError, "recognized_text changed"):
            compare_reports(report, changed)

    def test_rec_accuracy_metrics_use_unicode_codepoints(self) -> None:
        self.assertEqual(normalize("甲\r\n乙"), "甲\n乙")
        self.assertEqual(edit_distance("甲乙", "甲丙乙"), 1)
        self.assertEqual(edit_distance("甲乙", "甲"), 1)

    def test_pinned_small_validation_contract_is_resolvable(self) -> None:
        contract = resolve_contract(
            ROOT / "ci" / "ppocrv6-small-validation.json"
        )
        self.assertEqual(contract["rec_max_width"], 960)
        self.assertEqual(contract["expected_lines"], 16)
        self.assertEqual(
            contract["detector_path"], "models/ppocrv6-small/det.onnx"
        )
        self.assertEqual(
            contract["recognizer_path"], "models/ppocrv6-small/rec.onnx"
        )
        self.assertEqual(
            contract["classifier_path"], "models/ppocrv6-tiny/cls.onnx"
        )
        self.assertEqual(
            contract["dictionary_path"],
            "models/ppocrv6-shared/PP-OCRv6_small_rec_dict.txt",
        )
        self.assertEqual(len(contract["expected_full_text_sha256"]), 64)
        self.assertEqual(len(contract["model_catalog_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
