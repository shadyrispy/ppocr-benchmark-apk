#!/usr/bin/env python3
"""Package everything the Android app needs into app/src/main/assets.

Layout produced under app/src/main/assets/bench/:
  det.input.f32 / det.expected.f32   12345600 floats, NCHW little-endian
  cls.input.f32 / cls.expected.f32
  rec.input.f32 / rec.expected.f32
  sample.rgb                          raw RGB8 of the source image
  models/ort/{det,cls,rec}.static.onnx
  models/mnn/{det,cls,rec}.mnn
  models/ncnn/{det,cls,rec}.ncnn.{param,bin}
  models/lwm/{det,cls,rec}.lwm + ppocr_keys.txt
  model-manifest.json                 sha256 + shapes for provenance
"""

import hashlib
import json
import os
import shutil

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "app", "src", "main", "assets", "bench")

SHAPES = {"det": (1, 3, 960, 960), "cls": (1, 3, 80, 160), "rec": (1, 3, 48, 320)}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    if os.path.isdir(ASSETS):
        shutil.rmtree(ASSETS)
    os.makedirs(ASSETS)

    manifest = {"shapes": {k: list(v) for k, v in SHAPES.items()}, "files": {}}

    # --- tensors: the identical input every engine sees, plus ORT reference ---
    for name, shape in SHAPES.items():
        x = np.load(os.path.join(ROOT, "models", "ref", f"{name}.input.npy"))
        y = np.load(os.path.join(ROOT, "models", "ref", f"{name}.expected.npy"))
        assert tuple(x.shape) == shape, (name, x.shape)
        x = np.ascontiguousarray(x, dtype="<f4")
        y = np.ascontiguousarray(y, dtype="<f4")
        x.tofile(os.path.join(ASSETS, f"{name}.input.f32"))
        y.tofile(os.path.join(ASSETS, f"{name}.expected.f32"))
        manifest["files"][f"{name}.input.f32"] = sha256(os.path.join(ASSETS, f"{name}.input.f32"))
        manifest["files"][f"{name}.expected.f32"] = sha256(os.path.join(ASSETS, f"{name}.expected.f32"))
        print(f"{name}: {x.size} in / {y.size} out")

    # --- source image for the DET pre/post pipeline stage ---
    im = Image.open(os.path.join(ROOT, "models", "source", "sample.jpg")).convert("RGB")
    rgb = np.ascontiguousarray(np.array(im), dtype=np.uint8)
    rgb.tofile(os.path.join(ASSETS, "sample.rgb"))
    manifest["image"] = {"file": "sample.rgb", "w": im.width, "h": im.height, "c": 3}
    print(f"image: {im.width}x{im.height}")

    # --- models per engine ---
    plan = [
        ("ort", ROOT + "/models/static", ["det.static.onnx", "cls.static.onnx", "rec.static.onnx"]),
        ("mnn", ROOT + "/models/mnn", ["det.mnn", "cls.mnn", "rec.mnn"]),
        ("ncnn", ROOT + "/models/ncnn",
         ["det.ncnn.param", "det.ncnn.bin", "cls.ncnn.param", "cls.ncnn.bin",
          "rec.ncnn.param", "rec.ncnn.bin"]),
        ("ort-int8", ROOT + "/models/ort-int8", ["det.static.onnx", "cls.static.onnx", "rec.static.onnx"]),
        ("ncnn-int8", ROOT + "/models/ncnn-int8",
         ["det.ncnn.param", "det.ncnn.bin", "cls.ncnn.param", "cls.ncnn.bin",
          "rec.ncnn.param", "rec.ncnn.bin"]),
        ("lwm", ROOT + "/models/lwm", ["det.lwm", "cls.lwm", "rec.lwm"]),
    ]
    for engine, src, files in plan:
        dst = os.path.join(ASSETS, "models", engine)
        os.makedirs(dst, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            manifest["files"][f"models/{engine}/{f}"] = sha256(os.path.join(dst, f))
    shutil.copy2(os.path.join(ROOT, "models", "source", "ppocr_keys.txt"),
                 os.path.join(ASSETS, "models", "lwm", "ppocr_keys.txt"))

    with open(os.path.join(ASSETS, "model-manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(ASSETS) for f in fs)
    print(f"\nassets ready: {total/1e6:.1f} MB -> {ASSETS}")


if __name__ == "__main__":
    main()
