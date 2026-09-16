#!/usr/bin/env python3
"""Stage supplied Small assets plus the shared Tiny CLS for local validation.

The output is an analysis-only bundle in a caller-selected directory. It is
not a release packager and does not copy assets into the source tree unless
the caller explicitly points it there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.ppocr_contracts import (
    PP_OCRV6_SMALL_DET_SHA256,
    PP_OCRV6_SMALL_DICT_SHA256,
    PP_OCRV6_SMALL_REC_SHA256,
    PP_OCRV6_TINY_CLS_SHA256,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(source: Path, destination: Path, expected: str) -> str:
    if not source.is_file():
        raise ValueError(f"missing validation asset: {source}")
    actual = sha256(source)
    if actual != expected:
        raise ValueError(f"asset SHA-256 mismatch for {source}: {actual} != {expected}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return actual


def create_bundle(det: Path, cls: Path, rec: Path, dictionary: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checksums = {
        "det.onnx": stage(det, output_dir / "det.onnx", PP_OCRV6_SMALL_DET_SHA256),
        "cls.onnx": stage(cls, output_dir / "cls.onnx", PP_OCRV6_TINY_CLS_SHA256),
        "rec.onnx": stage(rec, output_dir / "rec.onnx", PP_OCRV6_SMALL_REC_SHA256),
        "ppocr_keys.txt": stage(dictionary, output_dir / "ppocr_keys.txt", PP_OCRV6_SMALL_DICT_SHA256),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "family": "PP-OCRv6",
        "variant": "small",
        "language": "zh",
        "runtime_status": "analysis-only",
        "models": {"det": "det.onnx", "cls": "cls.onnx", "rec": "rec.onnx"},
        "dictionary": "ppocr_keys.txt",
        "checksums": checksums,
        "classifier": {"variant": "tiny", "shared": True, "sha256": checksums["cls.onnx"]},
        "recommended": {
            "det": {"dynamic": True, "limit_side": 960},
            "rec": {"adaptive_width": True, "max_width": 960},
        },
    }
    (output_dir / "model.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det", type=Path, required=True)
    parser.add_argument("--cls", type=Path, required=True, help="shared Tiny CLS ONNX asset")
    parser.add_argument("--rec", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = create_bundle(args.det, args.cls, args.rec, args.dictionary, args.output_dir)
    print(json.dumps({"status": "ok", "output_dir": str(args.output_dir), "checksums": manifest["checksums"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
