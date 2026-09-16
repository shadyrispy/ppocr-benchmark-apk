#!/usr/bin/env python3
"""Compare Tiny, Small, and Medium complete-OCR profile summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_profile(
    driver: Path,
    variant: str,
    detector: Path,
    classifier: Path,
    recognizer: Path,
    dictionary: Path,
    sample: Path,
    iterations: int,
    workers: int,
    target_width: int,
) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(driver),
            str(detector),
            str(classifier),
            str(recognizer),
            str(dictionary),
            str(sample),
            str(iterations),
            str(workers),
            str(target_width),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{variant} workers={workers} failed:\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
        )
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    wall = report["wall_nanoseconds"]
    line_work = report["line_work_nanoseconds"]
    return {
        "variant": variant,
        "workers": workers,
        "iterations": iterations,
        "rec_target_width": target_width,
        "lines": report["lines"],
        "output_checksum": report["output_checksum"],
        "total_ms": wall["total"] / 1.0e6,
        "det_graph_ms": wall["det_graph"] / 1.0e6,
        "line_workers_ms": wall["line_workers"] / 1.0e6,
        "line_worker_critical_ms": wall["line_worker_critical"] / 1.0e6,
        "rec_graph_ms": line_work["rec_graph"] / 1.0e6,
        "cls_graph_ms": line_work["cls_graph"] / 1.0e6,
        "mean_rec_width": report["rec_width"]["mean_resized_width"],
        "mean_padding_ratio": report["rec_width"]["mean_padding_ratio"],
        "conv_ms": report["conv_nanoseconds"] / 1.0e6,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--tiny-det", type=Path, required=True)
    parser.add_argument("--tiny-rec", type=Path, required=True)
    parser.add_argument("--tiny-dictionary", type=Path, required=True)
    parser.add_argument("--small-det", type=Path, required=True)
    parser.add_argument("--small-rec", type=Path, required=True)
    parser.add_argument("--small-dictionary", type=Path, required=True)
    parser.add_argument("--medium-det", type=Path, required=True)
    parser.add_argument("--medium-rec", type=Path, required=True)
    parser.add_argument("--medium-dictionary", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--workers", type=int, action="append", default=None)
    parser.add_argument("--rec-target-width", type=int, default=960)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.iterations > 100:
        parser.error("--iterations must be between 1 and 100")
    workers = tuple(args.workers or (1, 4))
    if any(value <= 0 or value > 16 for value in workers):
        parser.error("--workers values must be between 1 and 16")

    variants = (
        ("tiny", args.tiny_det, args.tiny_rec, args.tiny_dictionary),
        ("small", args.small_det, args.small_rec, args.small_dictionary),
        ("medium", args.medium_det, args.medium_rec, args.medium_dictionary),
    )
    results = []
    for worker_count in workers:
        for variant, detector, recognizer, dictionary in variants:
            print(f"[profile] {variant} workers={worker_count}")
            results.append(
                run_profile(
                    args.driver,
                    variant,
                    detector,
                    args.classifier,
                    recognizer,
                    dictionary,
                    args.sample,
                    args.iterations,
                    worker_count,
                    args.rec_target_width,
                )
            )
    report = {
        "schema_version": 1,
        "sample": str(args.sample),
        "iterations": args.iterations,
        "workers": list(workers),
        "rec_target_width": args.rec_target_width,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": "ok", "output": str(args.output), "results": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
