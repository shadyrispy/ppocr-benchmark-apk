from __future__ import annotations

import unittest

from tools.summarize_ocr_dataset_reports import summarize_reports


def make_report(model: str, f1: float, cer: float, manifest: str = "abc") -> dict[str, object]:
    metrics = {
        "detection_precision": 0.9,
        "detection_recall": 0.8,
        "detection_f1": f1,
        "mean_matched_iou": 0.7,
        "exact_reference_line_rate": 0.4,
        "matched_exact_line_rate": 0.5,
        "cer_on_matched_lines": cer,
    }
    return {
        "schema_version": 1,
        "status": "ok",
        "model": model,
        "rec_max_width": 960,
        "iou_matching_threshold": 0.3,
        "dataset": {"manifest_sha256": manifest, "version": 1, "seed": 7},
        "metrics": metrics,
    }


class SummarizeOcrDatasetReportsTests(unittest.TestCase):
    def test_summary_preserves_order_and_baseline_deltas(self) -> None:
        summary = summarize_reports(
            [
                ("tiny", make_report("ppocrv6-tiny", 0.8, 0.2)),
                ("small", make_report("ppocrv6-small", 0.9, 0.1)),
                ("medium", make_report("ppocrv6-medium", 0.95, 0.05)),
            ]
        )
        self.assertEqual(summary["baseline"], "tiny")
        self.assertEqual([entry["name"] for entry in summary["reports"]], ["tiny", "small", "medium"])
        self.assertAlmostEqual(summary["deltas_vs_baseline"]["medium"]["detection_f1"]["delta"], 0.15)

    def test_summary_rejects_different_manifests(self) -> None:
        with self.assertRaisesRegex(ValueError, "different dataset"):
            summarize_reports(
                [
                    ("tiny", make_report("tiny", 0.8, 0.2)),
                    ("medium", make_report("medium", 0.9, 0.1, manifest="different")),
                ]
            )

    def test_summary_requires_unique_names_and_two_reports(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            summarize_reports([("tiny", make_report("tiny", 0.8, 0.2))])
        with self.assertRaisesRegex(ValueError, "unique"):
            summarize_reports(
                [("same", make_report("tiny", 0.8, 0.2)), ("same", make_report("small", 0.9, 0.1))]
            )


if __name__ == "__main__":
    unittest.main()
