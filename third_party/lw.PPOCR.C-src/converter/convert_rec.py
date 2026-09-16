#!/usr/bin/env python3
"""Convert a PP-OCR recognition model through the shared LWM entry point.

The converter accepts an explicit ONNX path so model variants can share one
validation and lowering path. The current lowering policy still validates the
bundled Tiny asset until Small graph support is completed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from converter.lwm_v0 import convert_rec_model
except ModuleNotFoundError:  # Direct execution: python converter/convert_rec.py
    from lwm_v0 import convert_rec_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="input REC ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="output LWM file")
    parser.add_argument("--metadata-output", type=Path, help="optional conversion metadata JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    info = convert_rec_model(args.input, args.output)
    metadata = {
        "format": "LWM",
        "format_version": "0.1",
        "input": str(args.input),
        "output": str(args.output),
        **info.__dict__,
        "checksum": f"0x{info.checksum:016x}",
    }
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
