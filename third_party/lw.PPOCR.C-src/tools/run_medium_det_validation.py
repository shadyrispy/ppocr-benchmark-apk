#!/usr/bin/env python3
"""Run the analysis-only fixed-shape PP-OCRv6 Medium DET validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(label: str, command: list[str], cwd: Path) -> str:
    print(f"[medium-det] {label}: {' '.join(command)}")
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model = args.model.resolve()
    build = args.build_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    driver = executable(build, "det-graph-driver")
    shapes = ((320, 320), (640, 640), (640, 960))
    prefix = "medium-det-lwm-"

    for height, width in shapes:
        stem = f"{prefix}{height}x{width}"
        lwm = output / f"{stem}.lwm"
        run(
            f"convert shape {height}x{width}",
            [
                sys.executable,
                "tools/convert_medium_det_experimental.py",
                "--model",
                str(model),
                "--height",
                str(height),
                "--width",
                str(width),
                "--output",
                str(lwm),
                "--report",
                str(output / f"{stem}.json"),
            ],
            root,
        )
        run(
            f"graph shape {height}x{width}",
            [str(driver), str(lwm), str(height), str(width), str(output / f"{stem}.f32")],
            root,
        )

    run(
        "numerical gate",
        [
            sys.executable,
            "tools/compare_small_det_execution.py",
            "--model",
            str(model),
            "--output-dir",
            str(output),
            "--output-prefix",
            prefix,
            "--label",
            "Medium DET",
            "--height",
            "320",
            "--width",
            "320",
            "--height",
            "640",
            "--width",
            "640",
            "--height",
            "640",
            "--width",
            "960",
            "--error-threshold",
            "1e-4",
            "--max-abs-error",
            "1e-4",
            "--max-mean-abs-error",
            "1e-6",
            "--max-fraction-over",
            "1e-4",
        ],
        root,
    )
    summary = {
        "status": "analysis-only",
        "model": str(model),
        "shapes": [list(shape) for shape in shapes],
        "max_abs_error_gate": 1.0e-4,
        "max_mean_abs_error_gate": 1.0e-6,
        "max_fraction_over_gate": 1.0e-4,
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
