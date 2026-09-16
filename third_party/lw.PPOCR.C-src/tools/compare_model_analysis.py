#!/usr/bin/env python3
"""Compare two converter/analyze_onnx.py JSON reports.

This is a development-time decision aid for Tiny -> Small/Medium work.  It
does not claim runtime support; it reports graph and cost differences and
which operators are outside the current LWM operator table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from converter.lwm_v0 import OP_IDS


# These ONNX nodes are removed, folded, or lowered before the LWM node table
# is emitted. They are part of the analysis surface but are not missing LWM
# runtime operators (Tiny already relies on them).
CONVERTER_LOWERED_OPS = {
    "GlobalAveragePool",
    "Identity",
    "Shape",
    "Slice",
}


def load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict) or not isinstance(report.get("models"), list):
        raise ValueError(f"invalid model analysis report: {path}")
    return report


def _by_label(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = {}
    for model in report["models"]:
        label = model.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("analysis report contains a model without a label")
        if label in models:
            raise ValueError(f"duplicate model label: {label}")
        models[label] = model
    return models


def _operator_types(report: dict[str, Any]) -> set[str]:
    union = report.get("operator_union", {})
    if isinstance(union, dict):
        return {str(operator) for operator in union}
    if isinstance(union, list):
        return {str(operator) for operator in union}
    raise ValueError("analysis report contains an invalid operator_union")


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    """Return stable, compact metrics suitable for JSON or a table."""
    result = {}
    for label, model in sorted(_by_label(report).items()):
        operators = model.get("operator_counts", {})
        if not isinstance(operators, dict):
            raise ValueError(f"invalid operator_counts for {label}")
        result[label] = {
            "nodes": model.get("node_count", 0),
            "model_bytes": model.get("file_bytes", 0),
            "weight_bytes": model.get("initializer_bytes", 0),
            "flops": model.get("flops", {}).get("total", 0),
            "operators": len(operators),
            "dynamic_values": len(model.get("dynamic_values", [])),
            "operator_types": sorted(operators),
            "shape_inference_error": model.get("shape_inference_error"),
        }
    return result


def compare(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_models = _by_label(base)
    candidate_models = _by_label(candidate)
    base_ops = _operator_types(base)
    candidate_ops = _operator_types(candidate)
    return {
        "base": summarize(base),
        "candidate": summarize(candidate),
        "new_operator_types": sorted(candidate_ops - base_ops),
        "removed_operator_types": sorted(base_ops - candidate_ops),
        "candidate_unsupported_operator_types": sorted(
            candidate_ops - set(OP_IDS) - CONVERTER_LOWERED_OPS
        ),
        "missing_labels": sorted(set(base_models) - set(candidate_models)),
        "new_labels": sorted(set(candidate_models) - set(base_models)),
    }


def render_text(result: dict[str, Any], base_name: str, candidate_name: str) -> str:
    lines = [
        "Model Analysis Diff",
        f"base: {base_name}",
        f"candidate: {candidate_name}",
        "",
        "label       nodes       model bytes   weight bytes      FLOPs   ops  dynamic",
        "----------- ----------- ------------- --------------- --------- ----- --------",
    ]
    labels = sorted(set(result["base"]) | set(result["candidate"]))
    for label in labels:
        base = result["base"].get(label, {})
        candidate = result["candidate"].get(label, {})
        left = base or {key: "-" for key in ("nodes", "model_bytes", "weight_bytes", "flops", "operators", "dynamic_values")}
        right = candidate or left
        lines.append(
            f"{label:<11} {left['nodes']:>11} {left['model_bytes']:>13} "
            f"{left['weight_bytes']:>15} {left['flops']:>9} {left['operators']:>5} "
            f"{left['dynamic_values']:>8}"
        )
        if base and candidate:
            lines.append(
                f"{'':<11} {right['nodes']:>11} {right['model_bytes']:>13} "
                f"{right['weight_bytes']:>15} {right['flops']:>9} {right['operators']:>5} "
                f"{right['dynamic_values']:>8}  <- candidate"
            )
    lines.extend([
        "",
        "New operator types: " + (", ".join(result["new_operator_types"]) or "none"),
        "Candidate operators outside LWM: "
        + (", ".join(result["candidate_unsupported_operator_types"]) or "none"),
        "Removed operator types: "
        + (", ".join(result["removed_operator_types"]) or "none"),
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="baseline analysis JSON, usually Tiny")
    parser.add_argument("candidate", type=Path, help="candidate analysis JSON")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--fail-on-unsupported",
        action="store_true",
        help="return 2 when the candidate contains an operator outside the current LWM table",
    )
    args = parser.parse_args(argv)

    result = compare(load_report(args.base), load_report(args.candidate))
    text = render_text(result, args.base.name, args.candidate.name)
    print(text, end="")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if args.fail_on_unsupported and result["candidate_unsupported_operator_types"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
