#!/usr/bin/env python3
"""Validate the pinned PP-OCRv6 Small REC and dictionary contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.ppocr_contracts import (
    PP_OCRV6_SMALL_DICT_ENTRIES,
    PP_OCRV6_SMALL_DICT_SHA256,
    PP_OCRV6_SMALL_REC_CLASSES,
    PP_OCRV6_SMALL_REC_SHA256,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_classes(model_path: Path) -> int:
    model = onnx.load(str(model_path), load_external_data=True)
    if len(model.graph.output) != 1:
        raise ValueError("REC model must have exactly one output")
    dimensions = model.graph.output[0].type.tensor_type.shape.dim
    if not dimensions or not dimensions[-1].HasField("dim_value"):
        raise ValueError("REC output class dimension must be static")
    classes = int(dimensions[-1].dim_value)
    if classes <= 0:
        raise ValueError("REC output class dimension must be positive")
    return classes


def validate(rec_model: Path, dictionary: Path) -> dict[str, object]:
    rec_digest = sha256(rec_model)
    dict_digest = sha256(dictionary)
    if rec_digest != PP_OCRV6_SMALL_REC_SHA256:
        raise ValueError(f"Small REC SHA-256 mismatch: {rec_digest} != {PP_OCRV6_SMALL_REC_SHA256}")
    if dict_digest != PP_OCRV6_SMALL_DICT_SHA256:
        raise ValueError(f"Small dictionary SHA-256 mismatch: {dict_digest} != {PP_OCRV6_SMALL_DICT_SHA256}")
    entries = dictionary.read_text(encoding="utf-8").splitlines()
    classes = output_classes(rec_model)
    if len(entries) != PP_OCRV6_SMALL_DICT_ENTRIES:
        raise ValueError(f"dictionary entries {len(entries)} != {PP_OCRV6_SMALL_DICT_ENTRIES}")
    if classes != PP_OCRV6_SMALL_REC_CLASSES:
        raise ValueError(f"REC classes {classes} != {PP_OCRV6_SMALL_REC_CLASSES}")
    if classes != len(entries) + 2:
        raise ValueError(f"REC classes {classes} do not equal dictionary entries + 2")
    return {
        "status": "ok",
        "rec_sha256": rec_digest,
        "dictionary_sha256": dict_digest,
        "rec_classes": classes,
        "dictionary_entries": len(entries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rec-model", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.rec_model, args.dictionary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
