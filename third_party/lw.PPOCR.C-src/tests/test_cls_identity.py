from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import onnx

from tools.compare_cls_identity import compare


ROOT = Path(__file__).resolve().parents[1]
CLS = ROOT / "models" / "ppocrv6-tiny" / "cls.onnx"


class ClsIdentityTests(unittest.TestCase):
    def test_identical_cls_assets_are_marked_shared(self) -> None:
        result = compare(CLS, CLS)
        self.assertTrue(result["shared_asset"])
        self.assertTrue(result["structural_match"])
        self.assertEqual(result["recommendation"], "reuse-tiny-cls-contract")

    def test_changed_bytes_are_not_called_shared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.onnx"
            model = onnx.load(str(CLS), load_external_data=True)
            model.doc_string = "candidate-only metadata"
            onnx.save(model, str(candidate))
            result = compare(CLS, candidate)
        self.assertFalse(result["shared_asset"])
        self.assertTrue(result["structural_match"])
        self.assertEqual(
            result["recommendation"],
            "candidate-requires-independent-cls-conversion-and-gates",
        )


if __name__ == "__main__":
    unittest.main()
