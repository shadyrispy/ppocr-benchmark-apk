#!/usr/bin/env python3
"""Resolve the checked-in Small CI policy and model-catalog assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.validate_model_catalog import validate_catalog


SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
ALLOWED_WIDTHS = (192, 320, 480, 640, 960)


def resolve_contract(
    contract_path: Path,
    expected_text_sha256_override: str | None = None,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    repository_root = contract_path.parent.parent
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 2 or contract.get("variant") != "ppocrv6-small":
        raise ValueError("unsupported Small validation contract")
    catalog_value = contract.get("model_catalog")
    if not isinstance(catalog_value, str) or not catalog_value:
        raise ValueError("Small validation model_catalog must be a relative path")
    catalog_relative = Path(catalog_value)
    if catalog_relative.is_absolute() or ".." in catalog_relative.parts:
        raise ValueError("Small validation model_catalog must stay inside the repository")
    catalog_path = (repository_root / catalog_relative).resolve()
    try:
        catalog_path.relative_to(repository_root)
    except ValueError as exception:
        raise ValueError(
            "Small validation model_catalog must stay inside the repository"
        ) from exception
    catalog_report = validate_catalog(catalog_path)
    assets = catalog_report["resolved"]["small"]
    model_root = catalog_path.parent

    def repository_asset(role: str) -> str:
        path = (model_root / assets[role]).resolve()
        try:
            return path.relative_to(repository_root).as_posix()
        except ValueError as exception:
            raise ValueError(
                f"Small validation asset escapes the repository: {role}"
            ) from exception

    widths = contract.get("rec_widths")
    if widths != list(ALLOWED_WIDTHS):
        raise ValueError(f"Small validation REC widths must be {list(ALLOWED_WIDTHS)}")
    full_ocr = contract.get("full_ocr")
    if not isinstance(full_ocr, dict) or full_ocr.get("rec_max_width") not in ALLOWED_WIDTHS:
        raise ValueError("Small validation full_ocr.rec_max_width is invalid")
    expected_text_sha256 = expected_text_sha256_override
    if expected_text_sha256 is None or expected_text_sha256 == "":
        expected_text_sha256 = full_ocr.get("expected_text_sha256") or ""
    if expected_text_sha256 and not SHA256_RE.fullmatch(expected_text_sha256):
        raise ValueError("expected full OCR text SHA-256 is invalid")
    return {
        "model_catalog_path": catalog_path.relative_to(repository_root).as_posix(),
        "detector_path": repository_asset("det"),
        "classifier_path": repository_asset("cls"),
        "recognizer_path": repository_asset("rec"),
        "dictionary_path": repository_asset("dictionary"),
        "expected_full_text_sha256": expected_text_sha256.lower(),
        "rec_max_width": int(full_ocr["rec_max_width"]),
        "expected_lines": int(full_ocr.get("expected_lines", 0)),
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
                output.write(f"{key}={value}\n")
    print(json.dumps(resolved, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
