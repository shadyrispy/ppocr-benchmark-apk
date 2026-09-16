from __future__ import annotations

import unittest

from tools.check_ocr_dataset_regression import check_regression


def make_report(model: str, f1: float, exact: float, cer: float) -> dict[str, object]:
    metrics = {
        "detection_precision": 0.9,
        "detection_recall": 0.8,
        "detection_f1": f1,
        "mean_matched_iou": 0.7,
        "exact_reference_line_rate": exact,
        "matched_exact_line_rate": exact,
        "cer_on_matched_lines": cer,
        "groups": {"category": {}, "orientation_degrees": {}, "canvas": {}},
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


class CheckOcrDatasetRegressionTests(unittest.TestCase):
    def test_thresholds_pass_for_small_improvements(self) -> None:
        result = check_regression(
            make_report("tiny", 0.90, 0.60, 0.04),
            make_report("small", 0.91, 0.62, 0.03),
            max_f1_drop=0.01,
            max_exact_drop=0.01,
            max_cer_increase=0.01,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["failed_metrics"], [])

    def test_thresholds_report_each_regression(self) -> None:
        result = check_regression(
            make_report("tiny", 0.90, 0.60, 0.04),
            make_report("candidate", 0.88, 0.57, 0.06),
            max_f1_drop=0.01,
            max_exact_drop=0.01,
            max_cer_increase=0.01,
        )
        self.assertEqual(result["status"], "regression")
        self.assertEqual(
            result["failed_metrics"],
            ["detection_f1", "exact_reference_line_rate", "cer_on_matched_lines"],
        )
        self.assertFalse(result["checks"]["detection_f1"]["passed"])

    def test_compatibility_is_checked_before_thresholds(self) -> None:
        baseline = make_report("tiny", 0.90, 0.60, 0.04)
        candidate = make_report("candidate", 0.91, 0.62, 0.03)
        candidate["dataset"]["manifest_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "different dataset"):
            check_regression(
                baseline,
                candidate,
                max_f1_drop=0.01,
                max_exact_drop=0.01,
                max_cer_increase=0.01,
            )


if __name__ == "__main__":
    unittest.main()
