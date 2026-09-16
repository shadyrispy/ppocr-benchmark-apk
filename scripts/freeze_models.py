#!/usr/bin/env python3
"""Freeze PP-OCRv6 tiny det/cls/rec ONNX to static input shapes.

Why: the source graphs carry Paddle symbolic-shape dimensions such as
``floor(DynamicDimension.1/2 - 1/2) + 1``. lw.PPOCR.C has its own dynamic-shape
propagator, but ORT would pay runtime symbolic-derivation cost, and ncnn/MNN
cannot convert them at all. Freezing to one concrete shape makes the four
engines compare on identical input tensors.

Outputs:
  models/static/{det,cls,rec}.static.onnx
  models/ref/<name>.input.npy       deterministic synthetic input
  models/ref/<name>.golden.npy      ORT fp32 CPU reference output
  models/static/manifest.json       input/output shapes + sha256
"""

import hashlib
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "models", "source")
STATIC = os.path.join(ROOT, "models", "static")
REF = os.path.join(ROOT, "models", "ref")

# Input shapes for the benchmark. det limit_side=960 matches lw's recommended
# full-page profile; rec width 320 is one of lw's own width buckets
# (192/320/480/640/960); cls is already fixed at 80x160.
SPECS = {
    "det": dict(shape=(1, 3, 960, 960), seed=10001),
    "cls": dict(shape=(1, 3, 80, 160), seed=10002),
    "rec": dict(shape=(1, 3, 48, 320), seed=10003),
}

# Image-like input: PPOCR preprocess feeds roughly zero-mean /255 pixels.
# Synthetic keeps every engine identical and removes decode variance.
def make_input(shape, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape, dtype=np.float32)
    return np.clip(x * 0.25 + 0.5, 0.0, 1.0).astype(np.float32)


def has_dynamic(model):
    dims = set()
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_param:
                dims.add(d.dim_param)
    return dims


def freeze(name, shape):
    src = os.path.join(SRC, f"{name}.onnx")
    model = onnx.load(src)
    print(f"\n=== {name}: opset={[o.version for o in model.opset_import]} "
          f"dynamic_before={sorted(has_dynamic(model)) or 'none'}")

    shape_map = {model.graph.input[0].name: list(shape)}
    model_simp, check = onnxsim.simplify(model, input_shapes=shape_map, check_n=3)

    leftover = has_dynamic(model_simp)
    if not check:
        raise SystemExit(f"[FAIL] {name}: onnxsim output failed checker")
    if leftover:
        # keep static nodes only if inputs/outputs are clean
        io_dyn = set()
        for vi in list(model_simp.graph.input) + list(model_simp.graph.output):
            for d in vi.type.tensor_type.shape.dim:
                if d.dim_param:
                    io_dyn.add(d.dim_param)
        if io_dyn:
            print(f"[WARN] {name}: still dynamic after simplify: {sorted(io_dyn)}")

    out = os.path.join(STATIC, f"{name}.static.onnx")
    onnx.save(model_simp, out)

    ins = [(i.name, [getattr(d, 'dim_param', None) or d.dim_value
                     for d in i.type.tensor_type.shape.dim]) for i in model_simp.graph.input]
    outs = [(o.name, [getattr(d, 'dim_param', None) or d.dim_value
                      for d in o.type.tensor_type.shape.dim]) for o in model_simp.graph.output]
    print(f"  saved {out}  {os.path.getsize(out)/1024:.0f}KB")
    print(f"  inputs  {ins}")
    print(f"  outputs {outs}")
    print(f"  nodes {len(model.graph.node)} -> {len(model_simp.graph.node)}")
    return out, ins, outs


def golden(name, path, ins, tensor):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    fed = {ins[0][0]: tensor}
    res = sess.run(None, fed)
    y = res[0]
    np.save(os.path.join(REF, f"{name}.input.npy"), tensor)
    np.save(os.path.join(REF, f"{name}.golden.npy"), y)
    print(f"  golden {list(y.shape)} dtype={y.dtype} "
          f"min={float(y.min()):.5f} max={float(y.max()):.5f}")
    return y.shape


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(STATIC, exist_ok=True)
    os.makedirs(REF, exist_ok=True)
    manifest = {}
    for name, spec in SPECS.items():
        out, ins, outs = freeze(name, spec["shape"])
        tensor = make_input(spec["shape"], spec["seed"])
        oshape = golden(name, out, ins, tensor)
        manifest[name] = dict(
            source_sha256=sha256(os.path.join(SRC, f"{name}.onnx")),
            static_file=os.path.basename(out),
            static_sha256=sha256(out),
            input_name=ins[0][0],
            input_shape=list(spec["shape"]),
            output_name=outs[0][0],
            output_shape=[int(v) if isinstance(v, (int, np.integer)) else v for v in oshape],
        )
    with open(os.path.join(STATIC, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n=== manifest ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
