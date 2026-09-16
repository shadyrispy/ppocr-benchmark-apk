#!/usr/bin/env python3
"""Run the analysis-only fixed-width or dynamic PP-OCRv6 Medium REC validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.ppocr_contracts import PP_OCRV6_REC_WIDTHS


def run(label: str, command: list[str], cwd: Path) -> str:
    print(f"[medium] {label}: {' '.join(command)}")
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
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="validate one dynamic LWM across all supported REC widths",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model = args.model.resolve()
    build = args.build_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    driver = executable(build, "rec-graph-driver")

    widths = tuple(PP_OCRV6_REC_WIDTHS)
    output_prefix = "medium-rec-dynamic-w" if args.dynamic else "medium-lwm-w"
    reports: list[str] = []
    if args.dynamic:
        run(
            "convert dynamic graph",
            [
                sys.executable,
                "tools/convert_medium_rec_experimental.py",
                "--model",
                str(model),
                "--dynamic",
                "--output",
                str(output / "medium-rec-dynamic.lwm"),
                "--report",
                str(output / "medium-rec-dynamic.json"),
            ],
            root,
        )
        reports.append(str(output / "medium-rec-dynamic.json"))
        lwm = output / "medium-rec-dynamic.lwm"
        for width in widths:
            run(
                f"graph width {width}",
                [str(driver), str(lwm), str(width), str(output / f"{output_prefix}{width}.f32")],
                root,
            )
    else:
        for width in widths:
            lwm = output / f"{output_prefix}{width}.lwm"
            report = output / f"{output_prefix}{width}.json"
            run(
                f"convert width {width}",
                [
                    sys.executable,
                    "tools/convert_medium_rec_experimental.py",
                    "--model",
                    str(model),
                    "--width",
                    str(width),
                    "--output",
                    str(lwm),
                    "--report",
                    str(report),
                ],
                root,
            )
            reports.append(str(report))
            run(
                f"graph width {width}",
                [str(driver), str(lwm), str(width), str(output / f"{output_prefix}{width}.f32")],
                root,
            )

    run(
        "numerical gate",
        [
            sys.executable,
            "tools/compare_small_rec_execution.py",
            "--model",
            str(model),
            "--output-dir",
            str(output),
            "--output-prefix",
            output_prefix,
            "--label",
            "Medium REC",
            "--error-threshold",
            "1e-4",
            "--max-abs-error",
            "3e-4",
            "--max-mean-abs-error",
            "1e-6",
            "--max-fraction-over",
            "1e-4",
        ],
        root,
    )

    summary = {
        "status": "analysis-only",
        "dynamic": args.dynamic,
        "model": os.fspath(model),
        "widths": list(widths),
        "max_abs_error_gate": 3.0e-4,
        "max_mean_abs_error_gate": 1.0e-6,
        "max_fraction_over_gate": 1.0e-4,
        "reports": reports,
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
