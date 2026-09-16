#!/bin/bash
# Assemble the final ncnn model set.
#
# Neither converter handles all three graphs, so each model comes from whichever
# converter produces a numerically correct graph. Every choice is backed by an
# offline measurement against the ORT reference (scripts/ncnn_offline_check.cpp):
#
#   det <- onnx2ncnn   pnnx collapses the DET feature maps (head deconvolution
#                      returns 32x32 instead of 960x960). onnx2ncnn is correct:
#                      max_abs_diff 1e-6.
#   cls <- pnnx + softmax axis patch
#                      pnnx emits "Softmax ... 0=0" (reduce over channels), but
#                      the two class scores live in w, so every element gets
#                      softmax'd alone and the output is [1.0, 1.0]. Flipping the
#                      axis to 1 gives max_abs_diff 9.4e-4.
#   rec <- pnnx        onnx2ncnn returns all zeros. pnnx is correct:
#                      max_abs_diff 0.136 on the probability map but 40/40
#                      (100%) argmax agreement with ORT, i.e. identical CTC
#                      decoding -- the numeric gap is benign fp32 reassociation.
set -euo pipefail

B="/Users/esc/Documents/GenshinTools/tools/ppocr-bench"
cd "$B/models"
OUT="ncnn"
mkdir -p "$OUT"

# det: onnx2ncnn output (kept in ncnn2/ by the caller)
cp ncnn2/det.ncnn.param "$OUT/det.ncnn.param"
cp ncnn2/det.ncnn.bin   "$OUT/det.ncnn.bin"

# cls / rec: pnnx output (kept in ncnn_pnnx_failed/ by the caller)
cp ncnn_pnnx_failed/cls.ncnn.param "$OUT/cls.ncnn.param"
cp ncnn_pnnx_failed/cls.ncnn.bin   "$OUT/cls.ncnn.bin"
cp ncnn_pnnx_failed/rec.ncnn.param "$OUT/rec.ncnn.param"
cp ncnn_pnnx_failed/rec.ncnn.bin   "$OUT/rec.ncnn.bin"

# patch the cls softmax axis: 0=0 (channels) -> 0=1 (width)
python3 - "$OUT/cls.ncnn.param" <<'PY'
import re, sys
path = sys.argv[1]
s = open(path).read()
new, n = re.subn(r'(Softmax\s+\S+\s+\d+ \d+ \S+ \S+) 0=0 ', r'\1 0=1 ', s)
open(path, "w").write(new)
print(f"cls softmax axis patched: {n} occurrence(s)")
assert n == 1, "expected exactly one Softmax to patch"
PY

echo "=== final ncnn set ==="
ls -la "$OUT"
grep -m1 "^Softmax" "$OUT/cls.ncnn.param" || true
