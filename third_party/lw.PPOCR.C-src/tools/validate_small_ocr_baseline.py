#!/usr/bin/env python3
"""Replay and validate an analysis-only Small OCR baseline JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def verify_model_identities(baseline: dict[str, object], paths: dict[str, Path]) -> None:
    models = baseline.get("models")
    if not isinstance(models, dict):
        raise SystemExit("baseline is missing model identity metadata")
    for role, path in paths.items():
        entry = models.get(role)
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise SystemExit(f"baseline is missing SHA-256 for {role}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise SystemExit(f"{role} SHA-256 mismatch: {actual} != {entry['sha256']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--box-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--score-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--rec-max-width",
        type=int,
        choices=(192, 320, 480, 640, 960),
        default=None,
        help="adaptive REC width limit; defaults to the baseline value or 960",
    )
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_width = int(baseline.get("rec_max_width", 960))
    rec_max_width = args.rec_max_width if args.rec_max_width is not None else baseline_width
    if rec_max_width not in (192, 320, 480, 640, 960):
        raise SystemExit(f"invalid baseline rec_max_width: {rec_max_width}")
    source = baseline.get("source_sha256")
    import hashlib

    digest = hashlib.sha256(args.sample.read_bytes()).hexdigest()
    if digest != source:
        raise SystemExit(f"sample SHA-256 mismatch: {digest} != {source}")
    verify_model_identities(
        baseline,
        {
            "detector": args.detector,
            "classifier": args.classifier,
            "recognizer": args.recognizer,
            "dictionary": args.dictionary,
        },
    )
    completed = subprocess.run(
        [
            str(args.ocr),
            str(args.detector),
            str(args.classifier),
            str(args.recognizer),
            str(args.dictionary),
            str(args.sample),
            str(rec_max_width),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stdout + completed.stderr)
    actual = []
    for raw_line in completed.stdout.splitlines():
        match = LINE_RE.match(raw_line)
        if match is None:
            continue
        points = POINT_RE.findall(match.group("box"))
        if len(points) != 4:
            raise SystemExit(f"invalid OCR box: {raw_line}")
        actual.append(
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
    expected = baseline.get("lines", [])
    if len(actual) != len(expected):
        raise SystemExit(f"line count mismatch: {len(actual)} != {len(expected)}")
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got["index"] != index or want.get("index") != index:
            raise SystemExit(f"line index mismatch at {index}")
        if got["text"] != want.get("text"):
            raise SystemExit(f"line {index} text mismatch: {got['text']!r} != {want.get('text')!r}")
        if got["cls_label"] != want.get("cls_label") or got["rotation"] != want.get("rotation"):
            raise SystemExit(f"line {index} classification metadata changed")
        if max(abs(a - b) for a, b in zip(got["box"], want["box"])) > args.box_tolerance:
            raise SystemExit(f"line {index} box changed beyond tolerance")
        for score_name in ("det_score", "rec_score", "cls_score"):
            if not math.isfinite(got[score_name]) or abs(got[score_name] - want[score_name]) > args.score_tolerance:
                raise SystemExit(f"line {index} {score_name} changed beyond tolerance")
    print(json.dumps({"status": "ok", "lines": len(actual), "rec_max_width": rec_max_width}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
