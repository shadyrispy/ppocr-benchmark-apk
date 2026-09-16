#!/usr/bin/env python3
"""Compare a deterministic OCR scene-suite report with a saved baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _compare_line(expected: dict[str, Any], actual: dict[str, Any], scene: str, index: int, box_tolerance: float, score_tolerance: float) -> None:
    for field in ("index", "text", "rotation"):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"{scene} line {index} {field} changed")
    expected_box = expected.get("box")
    actual_box = actual.get("box")
    if not isinstance(expected_box, list) or not isinstance(actual_box, list) or len(expected_box) != len(actual_box):
        raise ValueError(f"{scene} line {index} box shape changed")
    if any(abs(_number(got, "box") - _number(want, "box")) > box_tolerance for got, want in zip(actual_box, expected_box)):
        raise ValueError(f"{scene} line {index} box changed beyond tolerance")
    for field in ("det_score", "rec_score", "cls_score"):
        if abs(_number(actual.get(field), field) - _number(expected.get(field), field)) > score_tolerance:
            raise ValueError(f"{scene} line {index} {field} changed beyond tolerance")


def compare_reports(expected: dict[str, Any], actual: dict[str, Any], box_tolerance: float = 1.0e-3, score_tolerance: float = 1.0e-5) -> dict[str, Any]:
    if expected.get("schema_version") != actual.get("schema_version"):
        raise ValueError("scene baseline schema_version changed")
    if expected.get("manifest_sha256") != actual.get("manifest_sha256"):
        raise ValueError("scene manifest SHA-256 changed")
    if expected.get("rec_max_width") != actual.get("rec_max_width"):
        raise ValueError("scene REC max width changed")
    expected_models = expected.get("models")
    actual_models = actual.get("models")
    if not isinstance(expected_models, dict) or not isinstance(actual_models, dict):
        raise ValueError("scene model identities changed")
    if set(expected_models) != set(actual_models):
        raise ValueError("scene model roles changed")
    for role in expected_models:
        expected_identity = expected_models[role]
        actual_identity = actual_models[role]
        if not isinstance(expected_identity, dict) or not isinstance(actual_identity, dict):
            raise ValueError(f"scene model identity is malformed: {role}")
        if expected_identity.get("sha256") != actual_identity.get("sha256"):
            raise ValueError(f"scene model SHA-256 changed: {role}")
    expected_scenes = expected.get("scenes")
    actual_scenes = actual.get("scenes")
    if not isinstance(expected_scenes, list) or not isinstance(actual_scenes, list) or len(expected_scenes) != len(actual_scenes):
        raise ValueError("scene count changed")
    for expected_scene, actual_scene in zip(expected_scenes, actual_scenes):
        name = expected_scene.get("name")
        if name != actual_scene.get("name"):
            raise ValueError("scene order or name changed")
        if actual_scene.get("status") != "ok" or expected_scene.get("status") != "ok":
            raise ValueError(f"{name} is not an ok baseline scene")
        for field in ("width", "height", "expected_source_lines", "detected_lines", "recognized_text", "rotations"):
            if actual_scene.get(field) != expected_scene.get(field):
                raise ValueError(f"{name} {field} changed")
        expected_lines = expected_scene.get("lines")
        actual_lines = actual_scene.get("lines")
        if not isinstance(expected_lines, list) or not isinstance(actual_lines, list) or len(expected_lines) != len(actual_lines):
            raise ValueError(f"{name} line detail count changed")
        for index, (want, got) in enumerate(zip(expected_lines, actual_lines)):
            _compare_line(want, got, name, index, box_tolerance, score_tolerance)
    return {"status": "ok", "scenes": len(actual_scenes), "rec_max_width": actual.get("rec_max_width")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--box-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--score-tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()
    expected = json.loads(args.baseline.read_text(encoding="utf-8"))
    actual = json.loads(args.actual.read_text(encoding="utf-8"))
    print(json.dumps(compare_reports(expected, actual, args.box_tolerance, args.score_tolerance), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
