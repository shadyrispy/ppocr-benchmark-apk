#!/usr/bin/env python3
"""Prove that all four engines are asked to do numerically identical work.

The engines do NOT all consume the same file: lw.PPOCR.C's converter refuses any
model but its bundled *dynamic* ONNX, while ncnn/MNN cannot consume symbolic
dims at all. So:

  lw  -> source/det.onnx        (dynamic, LWM converted, shape propagated at run)
  ORT/MNN/ncnn -> static/det.*  (frozen at 960x960 / 80x160 / 48x320)

This script closes that loop by checking ORT's own output is identical when run
against both graphs, at the exact benchmark input tensor. If that holds, the
dynamic graph at a concrete shape and the frozen graph are the same function,
and every engine is solving the same problem.

Also emits models/ref/<name>.expected.npy, the reference every on-device backend
must reproduce within tolerance.
"""

import json
import os
import sys

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "models", "static")
SRC = os.path.join(ROOT, "models", "source")
REF = os.path.join(ROOT, "models", "ref")

SHAPES = {"det": (1, 3, 960, 960), "cls": (1, 3, 80, 160), "rec": (1, 3, 48, 320)}
IN_NAME = "x"


def run(path, x):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return s.run(None, {IN_NAME: x})[0]


def cmp(a, b, tag):
    if a.shape != b.shape:
        print(f"  [FAIL] {tag}: shape {a.shape} vs {b.shape}")
        return False
    absd = np.abs(a - b).astype(np.float64)
    max_abs = float(absd.max())
    denom = float(np.abs(a).astype(np.float64).max())
    rel = max_abs / denom if denom > 1e-12 else 0.0
    # cosine agreement
    fa, fb = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    cos = float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-30))
    ok = max_abs < 1e-4 or rel < 1e-4
    print(f"  [{'OK  ' if ok else 'FAIL'}] {tag}: max_abs={max_abs:.3e} rel={rel:.3e} cos={cos:.8f}")
    return ok


def main():
    report = {}
    all_ok = True
    for name, shape in SHAPES.items():
        print(f"\n=== {name} {shape}")
        x = np.load(os.path.join(REF, f"{name}.input.npy"))
        golden_static = np.load(os.path.join(REF, f"{name}.golden.npy"))
        ok = True
        ok &= cmp(golden_static, run(os.path.join(STATIC, f"{name}.static.onnx"), x), "ORT static vs saved golden")
        ok &= cmp(golden_static, run(os.path.join(SRC, f"{name}.onnx"), x), "ORT DYNAMIC vs ORT static")
        np.save(os.path.join(REF, f"{name}.expected.npy"), golden_static)
        report[name] = dict(shape=list(shape),
                            out_shape=list(golden_static.shape),
                            equivalent=bool(ok),
                            absmax=float(np.abs(golden_static).max()))
        all_ok &= bool(ok)
    with open(os.path.join(REF, "equivalence.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== summary ===")
    print(json.dumps(report, indent=2))
    print("\nALL EQUIVALENT" if all_ok else "\n*** EQUIVALENCE FAILED ***")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
