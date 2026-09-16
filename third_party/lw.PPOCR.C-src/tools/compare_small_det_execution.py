#!/usr/bin/env python3
"""Compare fixed-shape DET LWM outputs with ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative number")
    return value


def check_gate(results: list[dict[str, object]], args: argparse.Namespace) -> list[str]:
    if (
        args.max_abs_error is None
        and args.max_mean_abs_error is None
        and args.max_fraction_over is None
    ):
        return []
    failures: list[str] = []
    for item in results:
        label = f"shape {item['height']}x{item['width']}"
        if not item["finite"]:
            failures.append(f"{label}: LWM output contains NaN or Inf")
        if args.max_abs_error is not None and item["max_abs_error"] > args.max_abs_error:
            failures.append(
                f"{label}: max_abs_error {item['max_abs_error']:.9g} "
                f"> {args.max_abs_error:.9g}"
            )
        if args.max_mean_abs_error is not None and item["mean_abs_error"] > args.max_mean_abs_error:
            failures.append(
                f"{label}: mean_abs_error {item['mean_abs_error']:.9g} "
                f"> {args.max_mean_abs_error:.9g}"
            )
        if args.max_fraction_over is not None and item["fraction_abs_error_over"] > args.max_fraction_over:
            failures.append(
                f"{label}: fraction_abs_error_over {item['fraction_abs_error_over']:.9g} "
                f"> {args.max_fraction_over:.9g}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        default="small-det-lwm-",
        help="prefix for per-shape .f32 files (default: small-det-lwm-)",
    )
    parser.add_argument(
        "--label",
        default="Small DET",
        help="label used when reporting a failed numerical gate",
    )
    parser.add_argument("--height", type=int, action="append", dest="heights")
    parser.add_argument("--width", type=int, action="append", dest="widths")
    parser.add_argument("--error-threshold", type=nonnegative_float, default=1.0e-4)
    parser.add_argument("--max-abs-error", type=nonnegative_float)
    parser.add_argument("--max-mean-abs-error", type=nonnegative_float)
    parser.add_argument("--max-fraction-over", type=nonnegative_float)
    args = parser.parse_args()
    heights = args.heights or [640]
    widths = args.widths or [640]
    if len(heights) != len(widths):
        raise SystemExit("--height and --width must be supplied the same number of times")
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    results = []
    for height, width in zip(heights, widths):
        input_values = (
            ((np.arange(3 * height * width, dtype=np.uint64) * 23) % 269)
            .astype(np.int32)
            - 134
        ).astype(np.float32) / 134.0
        onnx_output = session.run(
            None,
            {session.get_inputs()[0].name: input_values.reshape(1, 3, height, width)},
        )[0].reshape(-1)
        lwm_path = args.output_dir / f"{args.output_prefix}{height}x{width}.f32"
        lwm_output = np.fromfile(lwm_path, dtype=np.float32)
        if lwm_output.size != onnx_output.size:
            raise SystemExit(
                f"shape {height}x{width}: output size mismatch "
                f"{lwm_output.size} != {onnx_output.size}"
            )
        delta = np.abs(onnx_output - lwm_output)
        results.append(
            {
                "height": height,
                "width": width,
                "elements": int(onnx_output.size),
                "max_abs_error": float(np.max(delta)),
                "mean_abs_error": float(np.mean(delta)),
                "fraction_abs_error_gt_1e-4": float(np.mean(delta > 1.0e-4)),
                "error_threshold": args.error_threshold,
                "fraction_abs_error_over": float(np.mean(delta > args.error_threshold)),
                "finite": bool(np.isfinite(onnx_output).all() and np.isfinite(lwm_output).all()),
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    failures = check_gate(results, args)
    if failures:
        print(f"{args.label} numerical gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
