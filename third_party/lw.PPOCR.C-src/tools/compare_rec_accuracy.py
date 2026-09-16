#!/usr/bin/env python3
"""Compare Tiny, Small, and optional Medium REC output on the corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n"))


def edit_distance(expected: str, actual: str) -> int:
    previous = list(range(len(actual) + 1))
    for row, expected_char in enumerate(expected, 1):
        current = [row]
        for column, actual_char in enumerate(actual, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected_char != actual_char),
            ))
        previous = current
    return previous[-1]


def run_model(
    driver: Path,
    model: Path,
    dictionary: Path,
    sample: Path,
    cases: list[dict[str, Any]],
    target_width: int,
    temporary: Path,
) -> dict[str, Any]:
    rgb = np.asarray(Image.open(sample).convert("RGB"), dtype=np.uint8)
    temporary.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    total_distance = 0
    total_reference_chars = 0
    exact_lines = 0
    for index, case in enumerate(cases):
        x1, y1, x2, y2 = (int(value) for value in case["box"])
        bgr = np.ascontiguousarray(rgb[y1:y2, x1:x2, ::-1])
        source = temporary / f"{index:02d}-{case['name']}.bgr"
        output = temporary / f"{index:02d}-{case['name']}.txt"
        source.write_bytes(bgr.tobytes(order="C"))
        completed = subprocess.run(
            [
                str(driver), "pipeline", str(model), str(dictionary), str(source),
                str(bgr.shape[1]), str(bgr.shape[0]), str(bgr.shape[1] * 3),
                str(target_width), str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{case['name']} failed:\n{completed.stdout}\n{completed.stderr}")
        expected = normalize(str(case["text"]))
        actual = normalize(output.read_text(encoding="utf-8"))
        distance = edit_distance(expected, actual)
        total_distance += distance
        total_reference_chars += len(expected)
        exact = expected == actual
        exact_lines += int(exact)
        results.append({
            "name": case["name"],
            "expected": expected,
            "actual": actual,
            "edit_distance": distance,
            "reference_chars": len(expected),
            "exact": exact,
        })
    return {
        "cases": len(cases),
        "reference_chars": total_reference_chars,
        "edit_distance": total_distance,
        "cer": total_distance / total_reference_chars if total_reference_chars else 0.0,
        "exact_line_rate": exact_lines / len(cases) if cases else 0.0,
        "exact_lines": exact_lines,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tiny-rec", type=Path, required=True)
    parser.add_argument("--tiny-dictionary", type=Path, required=True)
    parser.add_argument("--small-rec", type=Path, required=True)
    parser.add_argument("--small-dictionary", type=Path, required=True)
    parser.add_argument(
        "--medium-rec",
        type=Path,
        help="optional Medium dynamic REC LWM; uses the Small dictionary by default",
    )
    parser.add_argument("--medium-dictionary", type=Path)
    parser.add_argument(
        "--target-width",
        type=int,
        choices=(192, 320, 480, 640, 960),
        help="override the corpus target width for this comparison",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    if corpus.get("schema_version") != 1:
        raise SystemExit("unsupported REC corpus schema")
    if hashlib.sha256(args.sample.read_bytes()).hexdigest() != corpus.get("source_sha256"):
        raise SystemExit("sample SHA-256 does not match REC corpus")
    cases = corpus.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("REC corpus must contain cases")
    target_width = args.target_width or int(corpus["target_width"])
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        metrics = {
            "tiny": run_model(args.driver, args.tiny_rec, args.tiny_dictionary, args.sample, cases, target_width, temporary / "tiny"),
            "small": run_model(args.driver, args.small_rec, args.small_dictionary, args.sample, cases, target_width, temporary / "small"),
        }
        if args.medium_rec is not None:
            metrics["medium"] = run_model(
                args.driver,
                args.medium_rec,
                args.medium_dictionary or args.small_dictionary,
                args.sample,
                cases,
                target_width,
                temporary / "medium",
            )
    report = {
        "schema_version": 1,
        "corpus": str(args.corpus),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "target_width": target_width,
        "metrics": metrics,
    }
    output = args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    summary = {
        "status": "ok",
        "target_width": target_width,
        "tiny": {key: metrics["tiny"][key] for key in ("cer", "exact_line_rate", "exact_lines")},
        "small": {key: metrics["small"][key] for key in ("cer", "exact_line_rate", "exact_lines")},
    }
    if "medium" in metrics:
        summary["medium"] = {
            key: metrics["medium"][key]
            for key in ("cer", "exact_line_rate", "exact_lines")
        }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
