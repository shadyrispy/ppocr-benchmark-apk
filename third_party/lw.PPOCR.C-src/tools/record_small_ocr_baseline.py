#!/usr/bin/env python3
"""Record an analysis-only full-OCR baseline from the public PPM demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


LINE_RE = re.compile(
    r"^(?P<index>\d+) text=(?P<text>.*?) rec=(?P<rec>[-+0-9.eE]+) "
    r"det=(?P<det>[-+0-9.eE]+) cls=(?P<label>\d+)/(?P<cls>[-+0-9.eE]+) "
    r"rotate=(?P<rotation>\d+) \[(?P<box>.*)\]$"
)
POINT_RE = re.compile(r"\(([-+0-9.eE]+),([-+0-9.eE]+)\)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rec-max-width",
        type=int,
        choices=(192, 320, 480, 640, 960),
        default=960,
        help="adaptive REC width limit passed to lw-ocr-ppm",
    )
    args = parser.parse_args()
    command = [
        str(args.ocr),
        str(args.detector),
        str(args.classifier) if args.classifier else "",
        str(args.recognizer),
        str(args.dictionary),
        str(args.sample),
        str(args.rec_max_width),
    ]
    if args.classifier is None:
        raise SystemExit("--classifier is required by the current lw-ocr-ppm CLI")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stdout + completed.stderr)
    header = next((line for line in completed.stdout.splitlines() if line.startswith("lines=")), None)
    if header is None:
        raise SystemExit("OCR output did not contain a result header")
    header_values = dict(item.split("=", 1) for item in header.split() if "=" in item)
    lines = []
    for raw_line in completed.stdout.splitlines():
        match = LINE_RE.match(raw_line)
        if match is None:
            continue
        points = POINT_RE.findall(match.group("box"))
        if len(points) != 4:
            raise SystemExit(f"invalid OCR box: {raw_line}")
        lines.append(
            {
                "index": int(match.group("index")),
                "text": match.group("text"),
                "box": [float(value) for point in points for value in point],
                "det_score": float(match.group("det")),
                "rec_score": float(match.group("rec")),
                "cls_label": int(match.group("label")),
                "cls_score": float(match.group("cls")),
                "rotation": int(match.group("rotation")),
            }
        )
    if int(header_values.get("lines", -1)) != len(lines):
        raise SystemExit("OCR header line count does not match parsed lines")
    report = {
        "schema_version": 1,
        "status": "analysis-only",
        "source": str(args.sample),
        "source_sha256": sha256(args.sample),
        "rec_max_width": args.rec_max_width,
        "models": {
            "detector": {"path": str(args.detector), "sha256": sha256(args.detector)},
            "classifier": {"path": str(args.classifier), "sha256": sha256(args.classifier)},
            "recognizer": {"path": str(args.recognizer), "sha256": sha256(args.recognizer)},
            "dictionary": {"path": str(args.dictionary), "sha256": sha256(args.dictionary)},
        },
        "header": header_values,
        "lines": lines,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
