#!/usr/bin/env python3
"""Run the uninstrumented full-OCR benchmark across REC width buckets."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_once(
    benchmark: Path,
    detector: Path,
    classifier: Path,
    recognizer: Path,
    dictionary: Path,
    sample: Path,
    warmup: int,
    iterations: int,
    workers: int,
    width: int,
) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(benchmark),
            str(detector),
            str(classifier),
            str(recognizer),
            str(dictionary),
            str(sample),
            str(warmup),
            str(iterations),
            str(workers),
            str(width),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark failed for workers={workers}, width={width}:\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("benchmark returned no JSON output")
    try:
        report = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"benchmark returned invalid JSON: {lines[-1]!r}") from exc
    required = ("backend", "lines", "workers", "rec_target_width", "ocr_ms")
    missing = [key for key in required if key not in report]
    if missing:
        raise RuntimeError(f"benchmark JSON is missing fields: {', '.join(missing)}")
    return {
        "workers": workers,
        "rec_target_width": width,
        "backend": report["backend"],
        "lines": report["lines"],
        "mean_ms": report["ocr_ms"]["mean"],
        "p95_ms": report["ocr_ms"]["p95"],
        "detector_ms": report["detector_ms"]["mean"],
        "rss_after_warmup_bytes": report.get("rss_after_warmup_bytes"),
        "rss_final_bytes": report.get("rss_final_bytes"),
        "peak_rss_bytes": report.get("peak_rss_bytes"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--label", default="ocr")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--widths", type=int, nargs="+", default=[192, 320, 480, 640, 960])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.warmup <= 100:
        parser.error("--warmup must be between 0 and 100")
    if not 1 <= args.iterations <= 100:
        parser.error("--iterations must be between 1 and 100")
    if any(worker <= 0 or worker > 16 for worker in args.workers):
        parser.error("--workers values must be between 1 and 16")
    if any(width <= 0 for width in args.widths):
        parser.error("--widths values must be positive")

    results: list[dict[str, object]] = []
    for worker in args.workers:
        for width in args.widths:
            print(f"[benchmark] {args.label} workers={worker} width={width}")
            result = run_once(
                args.benchmark,
                args.detector,
                args.classifier,
                args.recognizer,
                args.dictionary,
                args.sample,
                args.warmup,
                args.iterations,
                worker,
                width,
            )
            result["label"] = args.label
            result["iterations"] = args.iterations
            results.append(result)

    report = {
        "schema_version": 1,
        "label": args.label,
        "sample": str(args.sample),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "workers": args.workers,
        "widths": args.widths,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": "ok", "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
