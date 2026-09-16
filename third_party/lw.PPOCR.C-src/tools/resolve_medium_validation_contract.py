#!/usr/bin/env python3
"""Resolve the checked-in Medium analysis CI policy and model assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.validate_model_catalog import validate_catalog


SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
ALLOWED_WIDTHS = (192, 320, 480, 640, 960)
ALLOWED_DET_SHAPES = ((320, 320), (640, 640), (640, 960))
ROOT_KEYS = {
    "schema_version",
    "variant",
    "model_catalog",
    "det_shapes",
    "rec_widths",
    "numerical_gates",
    "full_ocr",
}
GATE_KEYS = {
    "error_threshold",
    "max_abs_error",
    "max_mean_abs_error",
    "max_fraction_over",
}


def require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} keys changed: missing={missing}, unknown={unknown}"
        )


def finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return number


def resolve_contract(
    contract_path: Path,
    expected_text_sha256_override: str | None = None,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    repository_root = contract_path.parent.parent
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("Medium validation contract must be an object")
    require_keys(contract, ROOT_KEYS, "Medium validation contract")
    if contract["schema_version"] != 2 or contract["variant"] != "ppocrv6-medium":
        raise ValueError("unsupported Medium validation contract")

    catalog_value = contract["model_catalog"]
    if not isinstance(catalog_value, str) or not catalog_value:
        raise ValueError("Medium validation model_catalog must be a relative path")
    catalog_relative = Path(catalog_value)
    if catalog_relative.is_absolute() or ".." in catalog_relative.parts:
        raise ValueError("Medium validation model_catalog must stay inside the repository")
    catalog_path = (repository_root / catalog_relative).resolve()
    try:
        catalog_path.relative_to(repository_root)
    except ValueError as exception:
        raise ValueError(
            "Medium validation model_catalog must stay inside the repository"
        ) from exception
    catalog_report = validate_catalog(catalog_path)
    assets = catalog_report["resolved"]["medium"]
    model_root = catalog_path.parent

    def repository_asset(role: str) -> str:
        path = (model_root / assets[role]).resolve()
        try:
            relative = path.relative_to(repository_root)
        except ValueError as exception:
            raise ValueError(
                f"Medium validation asset escapes the repository: {role}"
            ) from exception
        if not path.is_file():
            raise ValueError(f"Medium validation asset is missing: {relative.as_posix()}")
        return relative.as_posix()

    det_shapes = contract["det_shapes"]
    if det_shapes != [list(shape) for shape in ALLOWED_DET_SHAPES]:
        raise ValueError(
            f"Medium validation DET shapes must be "
            f"{[list(shape) for shape in ALLOWED_DET_SHAPES]}"
        )
    rec_widths = contract["rec_widths"]
    if rec_widths != list(ALLOWED_WIDTHS):
        raise ValueError(f"Medium validation REC widths must be {list(ALLOWED_WIDTHS)}")

    numerical_gates = contract["numerical_gates"]
    if not isinstance(numerical_gates, dict):
        raise ValueError("Medium validation numerical_gates must be an object")
    require_keys(numerical_gates, {"det", "rec"}, "Medium numerical_gates")
    resolved_gates: dict[str, dict[str, float]] = {}
    for graph in ("det", "rec"):
        gate = numerical_gates[graph]
        if not isinstance(gate, dict):
            raise ValueError(f"Medium {graph} numerical gate must be an object")
        require_keys(gate, GATE_KEYS, f"Medium {graph} numerical gate")
        resolved_gates[graph] = {
            key: finite_nonnegative(gate[key], f"Medium {graph}.{key}")
            for key in sorted(GATE_KEYS)
        }

    full_ocr = contract["full_ocr"]
    if not isinstance(full_ocr, dict):
        raise ValueError("Medium validation full_ocr must be an object")
    require_keys(
        full_ocr,
        {"rec_max_width", "expected_lines", "expected_text_sha256"},
        "Medium full_ocr",
    )
    if full_ocr["rec_max_width"] not in ALLOWED_WIDTHS:
        raise ValueError("Medium validation full_ocr.rec_max_width is invalid")
    if isinstance(full_ocr["expected_lines"], bool) or not isinstance(
        full_ocr["expected_lines"], int
    ) or full_ocr["expected_lines"] <= 0:
        raise ValueError("Medium validation full_ocr.expected_lines is invalid")
    expected_text_sha256 = expected_text_sha256_override or full_ocr["expected_text_sha256"]
    if not isinstance(expected_text_sha256, str) or not SHA256_RE.fullmatch(
        expected_text_sha256
    ):
        raise ValueError("expected Medium full OCR text SHA-256 is invalid")

    return {
        "model_catalog_path": catalog_path.relative_to(repository_root).as_posix(),
        "detector_path": repository_asset("det"),
        "classifier_path": repository_asset("cls"),
        "recognizer_path": repository_asset("rec"),
        "dictionary_path": repository_asset("dictionary"),
        "det_shapes": [list(shape) for shape in ALLOWED_DET_SHAPES],
        "rec_widths": list(ALLOWED_WIDTHS),
        "numerical_gates": resolved_gates,
        "expected_full_text_sha256": expected_text_sha256.lower(),
        "rec_max_width": int(full_ocr["rec_max_width"]),
        "expected_lines": int(full_ocr["expected_lines"]),
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "model_catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-full-text-sha256-override")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    resolved = resolve_contract(
        args.contract,
        args.expected_full_text_sha256_override,
    )
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8", newline="\n") as output:
            for key, value in resolved.items():
                rendered = (
                    json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (dict, list))
                    else value
                )
                output.write(f"{key}={rendered}\n")
    print(json.dumps(resolved, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
