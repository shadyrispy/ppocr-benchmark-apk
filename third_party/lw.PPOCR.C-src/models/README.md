# Model assets

The repository includes the platform-independent PP-OCRv6 Tiny, Small, and
Medium ONNX conversion inputs. `ppocrv6-models.json` is the authoritative
catalog: it records every SHA-256, the upstream source, and the shared assets.
Small and Medium use the single
`ppocrv6-shared/PP-OCRv6_small_rec_dict.txt` dictionary. Tiny, Small, and
Medium use the single `ppocrv6-tiny/cls.onnx` direction classifier. The
release workflow also publishes these files as one checked model archive.

Deployment-specific TensorRT engines are intentionally excluded.

The Tiny directory also contains `model.json`, a schema-versioned model
package manifest with asset hashes and recommended preprocessing parameters.
Small and Medium remain analysis-only variants; adding the ONNX inputs to the
repository does not promote them to the default runtime, C ABI, Android, or
WASM model package. They can be analyzed without changing the runtime ABI:

```bash
python converter/analyze_onnx.py \
  --model-dir path/to/ppocrv6-small \
  --json-output ppocrv6-small-analysis.json \
  --markdown-output ppocrv6-small-analysis.md
```

If a source distribution does not ship a CLS file, pass explicit `--model`
arguments and record any shared CLS asset separately; do not silently assume
that Tiny and Small CLS weights are interchangeable.

The analysis-only path is intentional: passing ONNX validation does not claim
that the current LWM converter or full OCR pipeline supports that variant.

Medium REC fixed-width conversion can be reproduced for analysis with:

```bash
python tools/run_medium_rec_validation.py \
  --model models/ppocrv6-medium/rec.onnx \
  --build-dir build \
  --output-dir build-model-foundation/medium-analysis/validation
```

This generates temporary LWM files for widths 192, 320, 480, 640, and 960,
compares them with ONNX Runtime, and does not alter the released model catalog.
The corresponding Medium DET checkpoint uses
`tools/run_medium_det_validation.py` and tests `320x320`, `640x640`, and
`640x960`.

The preferred combined analysis gate reuses one dynamic DET and REC conversion,
checks every supported shape/width, and runs complete OCR with REC width 960:

```bash
python tools/run_medium_validation.py \
  --contract ci/ppocrv6-medium-validation.json \
  --build-dir build \
  --output-dir build-model-foundation/medium-validation-report
```

Generated LWM files and full graph tensors are temporary. Only compact reports
are retained by CI, and passing this gate does not promote Medium into a
production or platform release package.

To compare a candidate report with the checked-in Tiny baseline:

```bash
python tools/compare_model_analysis.py \
  docs/ppocrv6-tiny-analysis.json \
  ppocrv6-small-analysis.json
```

The diff highlights graph size, estimated FLOPs, dynamic values, new operator
types, and operators that are not represented by the current LWM table.

For a CI-style compatibility gate, add `--fail-on-unsupported`. It returns a
non-zero status when the candidate graph contains an operator outside the
current LWM table; this is still only a converter/runtime gate, not a release
approval.

To inspect the dynamic REC metadata before implementing a converter lowering,
use the ONNX Runtime probe:

```bash
python tools/probe_rec_shape_metadata.py \
  --model path/to/PP-OCRv6_small_rec.onnx \
  --json-output ppocrv6-small-rec-metadata.json
```

The probe records Shape/Slice values at representative widths. It does not
produce LWM and does not change runtime or release support.

The ONNX files are converter inputs. They are distributed in the dedicated
PP-OCRv6 model archive for reproducible analysis, not as dependencies of the
default pure-C runtime package.
