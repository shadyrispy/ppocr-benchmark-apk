#!/usr/bin/env python3
"""Validate the narrow dynamic-shape rule observed in PP-OCRv6 Small REC.

This is a converter-development gate, not a dynamic LWM implementation. It
proves that an instrumented ONNX metadata report contains exactly the six
known Shape/Slice outputs and that every width-dependent value is derived from
the input width using the observed width / 8 rule. The validator is
deliberately model-specific and fail-closed: a changed graph must produce a
new review rather than silently being accepted by a generic heuristic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.ppocr_contracts import (
    PP_OCRV6_REC_WIDTHS,
    PP_OCRV6_SMALL_REC_BASE_SHAPE,
    SMALL_REC_DYNAMIC_OUTPUTS,
    small_rec_metadata,
)


REQUIRED_WIDTHS = PP_OCRV6_REC_WIDTHS
EXPECTED_OUTPUTS = set(SMALL_REC_DYNAMIC_OUTPUTS)


def load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict):
        raise ValueError(f"invalid metadata report: {path}")
    return report


def validate_report(
    report: dict[str, Any], required_widths: tuple[int, ...] = REQUIRED_WIDTHS
) -> dict[str, Any]:
    """Validate one probe report and return a stable summary."""
    if not required_widths or len(set(required_widths)) != len(required_widths) or any(
        width <= 0 or width % 8 for width in required_widths
    ):
        raise ValueError("required widths must be distinct positive multiples of eight")
    if report.get("schema_version") != 1:
        raise ValueError("metadata report schema_version must be 1")
    input_info = report.get("input")
    if not isinstance(input_info, dict):
        raise ValueError("metadata report is missing input metadata")
    if input_info.get("base_shape") != list(PP_OCRV6_SMALL_REC_BASE_SHAPE):
        raise ValueError(f"unexpected REC base shape: {input_info.get('base_shape')!r}")
    widths = input_info.get("widths")
    if not isinstance(widths, list) or any(not isinstance(width, int) for width in widths):
        raise ValueError("metadata report input.widths must be a list of integers")
    if len(set(widths)) != len(widths) or any(width <= 0 or width % 8 for width in widths):
        raise ValueError("probe widths must be distinct positive multiples of eight")
    missing_widths = sorted(set(required_widths) - set(widths))
    if missing_widths:
        raise ValueError(f"metadata report is missing required widths: {missing_widths}")

    nodes = report.get("metadata_nodes")
    if not isinstance(nodes, list):
        raise ValueError("metadata report is missing metadata_nodes")
    outputs = [output for node in nodes for output in node.get("outputs", [])]
    if set(outputs) != EXPECTED_OUTPUTS or len(outputs) != len(EXPECTED_OUTPUTS):
        raise ValueError(f"metadata node outputs do not match the contract: {outputs!r}")
    if any(node.get("op") not in {"Shape", "Slice"} for node in nodes):
        raise ValueError("metadata_nodes contains a non Shape/Slice operation")

    probes = report.get("probes")
    if not isinstance(probes, list):
        raise ValueError("metadata report is missing probes")
    by_width: dict[int, dict[str, Any]] = {}
    for probe in probes:
        if not isinstance(probe, dict):
            raise ValueError("metadata probe is not an object")
        width = probe.get("width")
        shape = probe.get("input_shape")
        metadata = probe.get("metadata")
        if not isinstance(width, int) or width in by_width:
            raise ValueError("metadata probes must have unique integer widths")
        if shape != [1, 3, 48, width] or not isinstance(metadata, dict):
            raise ValueError(f"probe {width} has an invalid input shape or metadata map")
        if set(metadata) != EXPECTED_OUTPUTS:
            raise ValueError(f"probe {width} metadata outputs do not match the contract")
        by_width[width] = metadata

    if set(by_width) != set(widths):
        raise ValueError("metadata probe widths do not match input.widths")
    for width in required_widths:
        metadata = by_width.get(width)
        if metadata is None:
            raise ValueError(f"metadata report has no probe for width {width}")
    for width, metadata in by_width.items():
        for name, expected in small_rec_metadata(width).items():
            actual = metadata[name]
            if actual != expected:
                raise ValueError(f"probe {width} output {name} is {actual!r}, expected {expected!r}")

    return {
        "schema_version": 1,
        "contract": "ppocrv6-small-rec-width-div-8",
        "required_widths": list(required_widths),
        "probed_widths": sorted(by_width),
        "metadata_outputs": sorted(EXPECTED_OUTPUTS),
        "status": "validated-analysis-only",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON emitted by probe_rec_shape_metadata.py")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--require-width",
        action="append",
        type=int,
        dest="required_widths",
        help="alternative required width; defaults to the five Small REC widths",
    )
    args = parser.parse_args(argv)
    required_widths = tuple(args.required_widths) if args.required_widths else REQUIRED_WIDTHS
    summary = validate_report(load_report(args.report), required_widths)
    encoded = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
