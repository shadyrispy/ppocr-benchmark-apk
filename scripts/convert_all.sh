#!/bin/bash
# Convert the three frozen ONNX graphs into every engine's native format.
# All four engines consume the SAME static input tensor (models/ref/*.input.npy),
# so any latency difference comes from the runtime, not from shape handling.
set -euo pipefail

B="/Users/esc/Documents/GenshinTools/tools/ppocr-bench"
cd "$B/models"
VENV="$B/.venv/bin/python"
PNNX="$B/third_party/pnnx/pnnx-20260704-macos/pnnx"
LWSRC="$B/third_party/lw.PPOCR.C-src"

declare -a SHAPES=("det:[1,3,960,960]" "cls:[1,3,80,160]" "rec:[1,3,48,320]")

echo "########## MNN ##########"
for s in "${SHAPES[@]}"; do
  n="${s%%:*}"; shape="${s##*:}"
  "$VENV" -m MNN.tools.mnnconvert -f ONNX \
      --modelFile "static/$n.static.onnx" \
      --MNNModel "mnn/$n.mnn" --bizCode ppocr 2>&1 \
      | grep -E "Converted|inputTensors|outputTensors|Error|error|Unsupported" || true
  echo "  -> mnn/$n.mnn $(du -h mnn/$n.mnn 2>/dev/null | cut -f1)"
done

echo
echo "########## ncnn (pnnx) ##########"
for s in "${SHAPES[@]}"; do
  n="${s%%:*}"; shape="${s##*:}"
  rm -rf "ncnn/$n.ncnn.param" "ncnn/$n.ncnn.bin"
  "$PNNX" "static/$n.static.onnx" "inputshape=$shape" 2>&1 | tail -25
  # pnnx writes "<input-stem>.ncnn.{param,bin}" next to the input file
  for ext in param bin; do
    [ -f "static/$n.static.ncnn.$ext" ] && mv "static/$n.static.ncnn.$ext" "ncnn/$n.ncnn.$ext"
  done
  [ -f "static/$n.static_pnnx.py" ] && mv "static/$n.static_pnnx.py" "ncnn/$n.pnnx.py"
  echo "  -> $(ls ncnn/$n.ncnn.* 2>/dev/null | tr '\n' ' ')"
done

echo
echo "########## lw.PPOCR.C (LWM) ##########"
# The converter hard-gates on the SHA-256 of the ORIGINAL bundled dynamic ONNX
# (converter/lwm_v0.py: SUPPORTED_{DET,CLS,REC}_SHA256). So lw runs the dynamic
# graph and propagates the shape at runtime -- that is its supported path.
# ORT/MNN/ncnn cannot consume symbolic dims, hence they get the frozen graph.
# scripts/verify_equivalence.py proves ORT(dynamic)==ORT(static), which closes
# the fairness gap without touching the upstream converter.
for n in det cls rec; do
  ( cd "$LWSRC" && "$VENV" -m converter.convert_"$n" \
      --input "$B/models/source/$n.onnx" \
      --output "$B/models/lwm/$n.lwm" \
      --metadata-output "$B/models/lwm/$n.metadata.json" 2>&1 | tail -8 )
  echo "  -> lwm/$n.lwm $(du -h lwm/$n.lwm 2>/dev/null | cut -f1)"
done

echo
echo "########## done ##########"
ls -la mnn ncnn lwm
