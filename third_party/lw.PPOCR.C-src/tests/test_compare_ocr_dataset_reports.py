from __future__ import annotations

import unittest

from tools.compare_ocr_dataset_reports import compare_reports


def make_report(model: str, f1: float, cer: float) -> dict[str, object]:
    metrics = {
        "detection_precision": 0.9,
        "detection_recall": 0.8,
        "detection_f1": f1,
        "mean_matched_iou": 0.7,
        "exact_reference_line_rate": 0.4,
        "matched_exact_line_rate": 0.5,
        "cer_on_matched_lines": cer,
        "groups": {
            "category": {
                "project": {
                    "detection_recall": 0.8,
                    "mean_matched_iou": 0.7,
                    "exact_reference_line_rate": 0.4,
                    "matched_exact_line_rate": 0.5,
                    "cer_on_matched_lines": cer,
                }
            },
            "orientation_degrees": {},
            "canvas": {},
        },
    }
    return {
        "schema_version": 1,
        "status": "ok",
        "model": model,
        "rec_max_width": 960,
        "iou_matching_threshold": 0.3,
        "dataset": {"manifest_sha256": "abc", "version": 1, "seed": 7},
        "metrics": metrics,
    }


class CompareOcrDatasetReportsTests(unittest.TestCase):
    def test_reports_preserve_candidate_minus_baseline_delta(self) -> None:
        result = compare_reports(make_report("tiny", 0.8, 0.2), make_report("small", 0.9, 0.1))
        self.assertAlmostEqual(result["overall"]["detection_f1"]["delta"], 0.1)
        self.assertAlmostEqual(result["overall"]["cer_on_matched_lines"]["delta"], -0.1)
        self.assertAlmostEqual(result["groups"]["category"]["project"]["detection_recall"]["delta"], 0.0)

    def test_reports_must_use_same_manifest(self) -> None:
        baseline = make_report("tiny", 0.8, 0.2)
        candidate = make_report("small", 0.9, 0.1)
        candidate["dataset"]["manifest_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "different dataset"):
            compare_reports(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
