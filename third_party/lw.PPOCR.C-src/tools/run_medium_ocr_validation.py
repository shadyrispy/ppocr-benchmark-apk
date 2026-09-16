#!/usr/bin/env python3
"""Run the analysis-only PP-OCRv6 Medium DET/REC full-OCR checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


LINE_RE = re.compile(r"^\d+ text=(?P<text>.*?) rec=")


def run(label: str, command: list[str], cwd: Path) -> str:
    print(f"[medium-ocr] {label}: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{completed.stderr}")
    return completed.stdout


def executable(build_dir: Path, name: str) -> Path:
    candidates = [build_dir / name]
    if sys.platform == "win32":
        candidates.insert(0, build_dir / "Release" / f"{name}.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"build executable not found: {name} under {build_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det-model", type=Path, required=True)
    parser.add_argument("--rec-model", type=Path, required=True)
    parser.add_argument("--cls-model", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-lines", type=int, default=16)
    parser.add_argument("--expected-first-text", default="纯臻营养护发素")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    det = output / "medium-det-dynamic.lwm"
    rec = output / "medium-rec-dynamic.lwm"
    run(
        "DET conversion",
        [
            sys.executable,
            "tools/convert_medium_det_experimental.py",
            "--model",
            str(args.det_model.resolve()),
            "--dynamic",
            "--output",
            str(det),
            "--report",
            str(output / "medium-det-dynamic.json"),
        ],
        root,
    )
    run(
        "REC conversion",
        [
            sys.executable,
            "tools/convert_medium_rec_experimental.py",
            "--model",
            str(args.rec_model.resolve()),
            "--dynamic",
            "--output",
            str(rec),
            "--report",
            str(output / "medium-rec-dynamic.json"),
        ],
        root,
    )
    stdout = run(
        "full OCR",
        [
            str(executable(args.build_dir.resolve(), "lw-ocr-ppm")),
            str(det),
            str(args.cls_model.resolve()),
            str(rec),
            str(args.dictionary.resolve()),
            str(args.sample.resolve()),
            "960",
        ],
        root,
    )
    match = re.search(r"(?m)^lines=(\d+)\s", stdout)
    if match is None or int(match.group(1)) != args.expected_lines:
        raise RuntimeError(f"expected {args.expected_lines} OCR lines:\n{stdout}")
    lines = [match.group("text") for line in stdout.splitlines() if (match := LINE_RE.match(line))]
    if len(lines) != args.expected_lines:
        raise RuntimeError(f"parsed {len(lines)} OCR lines, expected {args.expected_lines}")
    if lines[0] != args.expected_first_text:
        raise RuntimeError(f"first OCR line changed: {lines[0]!r}")
    text_sha256 = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    (output / "full-ocr.txt").write_text(stdout, encoding="utf-8", newline="\n")
    summary = {
        "status": "analysis-only",
        "lines": len(lines),
        "first_text": lines[0],
        "text_sha256": text_sha256,
        "det_model": str(args.det_model),
        "rec_model": str(args.rec_model),
        "shared_cls": str(args.cls_model),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
