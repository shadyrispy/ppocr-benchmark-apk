from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from converter import analyze_onnx
from tools import compare_model_analysis


ROOT = pathlib.Path(__file__).resolve().parents[1]


class CompareModelAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = analyze_onnx.build_report([
            analyze_onnx.analyze_model(
                label,
                path,
                analyze_onnx.DEFAULT_REPRESENTATIVE_SHAPES[label],
            )
            for label, path in analyze_onnx.DEFAULT_MODELS.items()
        ])

    def test_diff_reports_new_and_unsupported_operator_types(self) -> None:
        candidate = json.loads(json.dumps(self.report))
        candidate["operator_union"]["HardSwish"] = ["rec"]
        candidate["operator_union_count"] += 1
        result = compare_model_analysis.compare(self.report, candidate)
        self.assertEqual(result["new_operator_types"], ["HardSwish"])
        self.assertEqual(result["candidate_unsupported_operator_types"], ["HardSwish"])

    def test_cli_writes_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            first = root / "tiny.json"
            second = root / "candidate.json"
            output = root / "diff.json"
            for path in (first, second):
                path.write_text(
                    json.dumps(self.report, ensure_ascii=False), encoding="utf-8"
                )
            self.assertEqual(
                compare_model_analysis.main([
                    str(first), str(second), "--json-output", str(output)
                ]),
                0,
            )
            parsed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(parsed["new_operator_types"], [])

    def test_cli_can_fail_on_unsupported_operator(self) -> None:
        candidate = json.loads(json.dumps(self.report))
        candidate["operator_union"]["HardSwish"] = ["rec"]
        candidate["operator_union_count"] += 1
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            base_path = root / "tiny.json"
            candidate_path = root / "candidate.json"
            base_path.write_text(json.dumps(self.report), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            self.assertEqual(
                compare_model_analysis.main([
                    str(base_path), str(candidate_path), "--fail-on-unsupported"
                ]),
                2,
            )


if __name__ == "__main__":
    unittest.main()
