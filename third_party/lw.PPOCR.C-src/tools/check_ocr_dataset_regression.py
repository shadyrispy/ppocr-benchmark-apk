#!/usr/bin/env python3
"""Fail when a candidate OCR report regresses beyond explicit thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Support the documented ``python tools/<script>.py`` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_ocr_dataset_reports import compare_reports, load_report


CHECKS = (
    ("detection_f1", "max_f1_drop", "drop"),
    ("exact_reference_line_rate", "max_exact_drop", "drop"),
    ("cer_on_matched_lines", "max_cer_increase", "increase"),
)


def check_regression(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_f1_drop: float,
    max_exact_drop: float,
    max_cer_increase: float,
) -> dict[str, Any]:
    comparison = compare_reports(baseline, candidate)
    thresholds = {
        "max_f1_drop": max_f1_drop,
        "max_exact_drop": max_exact_drop,
        "max_cer_increase": max_cer_increase,
    }
    checks: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for metric, threshold_name, direction in CHECKS:
        delta = comparison["overall"][metric]["delta"]
        threshold = thresholds[threshold_name]
        if direction == "drop":
            passed = delta >= -threshold
            limit_text = f"delta >= {-threshold:g}"
        else:
            passed = delta <= threshold
            limit_text = f"delta <= {threshold:g}"
        checks[metric] = {
            "baseline": comparison["overall"][metric]["baseline"],
            "candidate": comparison["overall"][metric]["candidate"],
            "delta": delta,
            "threshold": threshold,
            "passed": passed,
            "rule": limit_text,
        }
        if not passed:
            failed.append(metric)
    return {
        "schema_version": 1,
        "status": "ok" if not failed else "regression",
        "baseline_model": comparison["baseline_model"],
        "candidate_model": comparison["candidate_model"],
        "rec_max_width": comparison["rec_max_width"],
        "dataset": comparison["dataset"],
        "thresholds": thresholds,
        "checks": checks,
        "failed_metrics": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-f1-drop", type=float, required=True)
    parser.add_argument("--max-exact-drop", type=float, required=True)
    parser.add_argument("--max-cer-increase", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.max_f1_drop, args.max_exact_drop, args.max_cer_increase) < 0:
        raise SystemExit("regression thresholds must be non-negative")
    try:
        result = check_regression(
            load_report(args.baseline),
            load_report(args.candidate),
            max_f1_drop=args.max_f1_drop,
            max_exact_drop=args.max_exact_drop,
            max_cer_increase=args.max_cer_increase,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
