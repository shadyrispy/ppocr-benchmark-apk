#!/usr/bin/env python3
"""Summarize several project-owned full-OCR reports on one generated corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Running ``python tools/<script>.py`` puts ``tools/`` rather than the
# repository root on sys.path. Add the root explicitly for that supported CLI
# form while keeping normal package imports unchanged.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_ocr_dataset_reports import (
    FULL_KEYS,
    ensure_compatible,
    load_report,
    value_delta,
)


def parse_report_spec(spec: str) -> tuple[str, Path]:
    name, separator, path = spec.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError("report must use NAME=PATH syntax")
    return name.strip(), Path(path.strip())


def summarize_reports(reports: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("at least two reports are required")
    names = [name for name, _ in reports]
    if len(set(names)) != len(names):
        raise ValueError("report names must be unique")
    baseline_name, baseline = reports[0]
    for _, report in reports[1:]:
        ensure_compatible(baseline, report)
    baseline_metrics = baseline["metrics"]
    entries: list[dict[str, Any]] = []
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for name, report in reports:
        metrics = report["metrics"]
        entries.append(
            {
                "name": name,
                "model": report.get("model"),
                "metrics": {key: metrics.get(key) for key in FULL_KEYS},
            }
        )
        if name != baseline_name:
            deltas[name] = value_delta(baseline_metrics, metrics, FULL_KEYS)
    return {
        "schema_version": 1,
        "baseline": baseline_name,
        "rec_max_width": baseline.get("rec_max_width"),
        "iou_matching_threshold": baseline.get("iou_matching_threshold"),
        "dataset": baseline.get("dataset"),
        "reports": entries,
        "deltas_vs_baseline": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="report label and JSON path; first report is the baseline",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        reports = [(name, load_report(path)) for name, path in map(parse_report_spec, args.report)]
        summary = summarize_reports(reports)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "baseline": summary["baseline"],
                "reports": [entry["name"] for entry in summary["reports"]],
                "deltas_vs_baseline": summary["deltas_vs_baseline"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
