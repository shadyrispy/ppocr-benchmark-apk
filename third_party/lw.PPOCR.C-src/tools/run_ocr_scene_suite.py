#!/usr/bin/env python3
"""Run the deterministic OCR scene set and write a compact JSON report.

The scene generator stores source strings in ``manifest.json``.  This runner
keeps those strings separate from OCR output: it records the expected logical
line count, detected regions, recognized text, rotations, and score minima,
but does not impose a model-specific exact-text gate.  That makes it useful
for comparing experimental models while still failing on process errors,
malformed output, or an empty detection result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


LINE_RE = re.compile(
    r"^(?P<index>\d+) text=(?P<text>.*?) rec=(?P<rec>[-+0-9.eE]+) "
    r"det=(?P<det>[-+0-9.eE]+) cls=(?P<label>\d+)/(?P<cls>[-+0-9.eE]+) "
    r"rotate=(?P<rotation>\d+) \[(?P<box>.*)\]$"
)
POINT_RE = re.compile(r"\(([-+0-9.eE]+),([-+0-9.eE]+)\)")
HEADER_RE = re.compile(
    r"^lines=(?P<detected>\d+) "
    r"image=(?P<image_width>\d+)x(?P<image_height>\d+) "
    r"detector_input=(?P<detector_width>\d+)x(?P<detector_height>\d+)$"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_output(stdout: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] = {}
    lines: list[dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        if raw_line.startswith("config "):
            header["config"] = raw_line.removeprefix("config ")
            continue
        header_match = HEADER_RE.match(raw_line)
        if header_match is not None:
            header.update(
                {
                    "detected_lines": int(header_match.group("detected")),
                    "image_width": int(header_match.group("image_width")),
                    "image_height": int(header_match.group("image_height")),
                    "detector_width": int(header_match.group("detector_width")),
                    "detector_height": int(header_match.group("detector_height")),
                }
            )
            continue
        match = LINE_RE.match(raw_line)
        if match is None:
            continue
        points = POINT_RE.findall(match.group("box"))
        if len(points) != 4:
            raise ValueError(f"invalid OCR box: {raw_line}")
        scores = {
            "det_score": float(match.group("det")),
            "rec_score": float(match.group("rec")),
            "cls_score": float(match.group("cls")),
        }
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError(f"non-finite OCR score: {raw_line}")
        lines.append(
            {
                "index": int(match.group("index")),
                "text": match.group("text"),
                "box": [float(value) for point in points for value in point],
                "rotation": int(match.group("rotation")),
                **scores,
            }
        )
    return header, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--rec-max-width", type=int, default=960, choices=(192, 320, 480, 640, 960))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise SystemExit("manifest must contain a non-empty scenes array")
    scene_root = args.manifest.parent
    report: dict[str, Any] = {
        "schema_version": 1,
        "manifest_sha256": sha256(args.manifest),
        "rec_max_width": args.rec_max_width,
        "models": {
            "detector": {"path": str(args.detector), "sha256": sha256(args.detector)},
            "classifier": {"path": str(args.classifier), "sha256": sha256(args.classifier)},
            "recognizer": {"path": str(args.recognizer), "sha256": sha256(args.recognizer)},
            "dictionary": {"path": str(args.dictionary), "sha256": sha256(args.dictionary)},
        },
        "scenes": [],
    }
    failures: list[str] = []
    for scene in scenes:
        name = scene.get("name")
        ppm_name = scene.get("ppm")
        if not isinstance(name, str) or not isinstance(ppm_name, str):
            raise SystemExit("manifest scene is missing name or ppm")
        ppm_path = scene_root / ppm_name
        completed = subprocess.run(
            [
                str(args.ocr),
                str(args.detector),
                str(args.classifier),
                str(args.recognizer),
                str(args.dictionary),
                str(ppm_path),
                str(args.rec_max_width),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        entry: dict[str, Any] = {
            "name": name,
            "png": scene.get("png"),
            "ppm": ppm_name,
            "width": scene.get("width"),
            "height": scene.get("height"),
            "expected_source_lines": len(scene.get("lines", [])),
            "status": "ok",
        }
        if completed.returncode != 0:
            entry["status"] = "failed"
            entry["error"] = (completed.stdout + completed.stderr).strip()[-4000:]
            failures.append(f"{name}: process exit {completed.returncode}")
        else:
            try:
                header, lines = parse_output(completed.stdout)
                if not lines:
                    raise ValueError("OCR returned no regions")
                entry.update(
                    {
                        "header": header,
                        "detected_lines": len(lines),
                        "recognized_text": [line["text"] for line in lines],
                        "rotations": [line["rotation"] for line in lines],
                        "lines": lines,
                        "min_det_score": min(line["det_score"] for line in lines),
                        "min_rec_score": min(line["rec_score"] for line in lines),
                    }
                )
            except (ValueError, KeyError) as error:
                entry["status"] = "failed"
                entry["error"] = str(error)
                failures.append(f"{name}: {error}")
        report["scenes"].append(entry)

    output = args.output or scene_root / "scene-suite-results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "failed" if failures else "ok", "scenes": len(scenes), "output": str(output)}, ensure_ascii=False))
    if failures:
        for failure in failures:
            print(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
