#!/usr/bin/env python3
"""Compare two project-owned full-OCR dataset reports.

The reports must come from the same generated manifest. The output preserves
the baseline and candidate values and records candidate-minus-baseline deltas
for the overall, category, orientation, and canvas groups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


FULL_KEYS = (
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "mean_matched_iou",
    "exact_reference_line_rate",
    "matched_exact_line_rate",
    "cer_on_matched_lines",
)
LINE_KEYS = (
    "detection_recall",
    "mean_matched_iou",
    "exact_reference_line_rate",
    "matched_exact_line_rate",
    "cer_on_matched_lines",
)
GROUP_TYPES = ("category", "orientation_degrees", "canvas")


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError(f"unsupported OCR report: {path}")
    if report.get("status") != "ok":
        raise ValueError(f"OCR report is not successful: {path}")
    if not isinstance(report.get("metrics"), dict):
        raise ValueError(f"OCR report lacks metrics: {path}")
    return report


def ensure_compatible(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    baseline_dataset = baseline.get("dataset", {})
    candidate_dataset = candidate.get("dataset", {})
    for key in ("manifest_sha256", "version", "seed"):
        if baseline_dataset.get(key) != candidate_dataset.get(key):
            raise ValueError(f"reports use different dataset {key}")
    for key in ("rec_max_width", "iou_matching_threshold"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"reports use different {key}")


def value_delta(
    baseline: dict[str, Any], candidate: dict[str, Any], keys: Iterable[str]
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for key in keys:
        old = baseline.get(key)
        new = candidate.get(key)
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            continue
        result[key] = {
            "baseline": float(old),
            "candidate": float(new),
            "delta": float(new) - float(old),
        }
    return result


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ensure_compatible(baseline, candidate)
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    groups: dict[str, dict[str, Any]] = {}
    for group_type in GROUP_TYPES:
        old_groups = baseline_metrics.get("groups", {}).get(group_type, {})
        new_groups = candidate_metrics.get("groups", {}).get(group_type, {})
        if not isinstance(old_groups, dict) or not isinstance(new_groups, dict):
            continue
        entries: dict[str, Any] = {}
        for name in sorted(set(old_groups) | set(new_groups)):
            old = old_groups.get(name, {})
            new = new_groups.get(name, {})
            entries[name] = value_delta(
                old,
                new,
                FULL_KEYS if group_type == "canvas" else LINE_KEYS,
            )
        groups[group_type] = entries
    return {
        "schema_version": 1,
        "baseline_model": baseline.get("model"),
        "candidate_model": candidate.get("model"),
        "rec_max_width": candidate.get("rec_max_width"),
        "dataset": candidate.get("dataset"),
        "overall": value_delta(baseline_metrics, candidate_metrics, FULL_KEYS),
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = compare_reports(load_report(args.baseline), load_report(args.candidate))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "baseline": report["baseline_model"],
                "candidate": report["candidate_model"],
                "overall": report["overall"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
