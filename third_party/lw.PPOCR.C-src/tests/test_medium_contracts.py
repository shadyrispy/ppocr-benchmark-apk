from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.resolve_medium_validation_contract import (
    ALLOWED_DET_SHAPES,
    ALLOWED_WIDTHS,
    resolve_contract,
)
from tools.run_medium_validation import (
    command_gate_arguments,
    configure_utf8_output,
    recognized_text,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "ci" / "ppocrv6-medium-validation.json"


class MediumContractTests(unittest.TestCase):
    def test_validation_console_uses_utf8_with_safe_error_fallback(self) -> None:
        class ReconfigurableStream:
            def __init__(self) -> None:
                self.calls: list[dict[str, str]] = []

            def reconfigure(self, **kwargs: str) -> None:
                self.calls.append(kwargs)

        stdout = ReconfigurableStream()
        stderr = ReconfigurableStream()
        with mock.patch("tools.run_medium_validation.sys.stdout", stdout), mock.patch(
            "tools.run_medium_validation.sys.stderr", stderr
        ):
            configure_utf8_output()
        expected = [{"encoding": "utf-8", "errors": "backslashreplace"}]
        self.assertEqual(stdout.calls, expected)
        self.assertEqual(stderr.calls, expected)

    def test_pinned_contract_resolves_shared_assets_and_960_gate(self) -> None:
        resolved = resolve_contract(CONTRACT)
        self.assertEqual(
            resolved["detector_path"], "models/ppocrv6-medium/det.onnx"
        )
        self.assertEqual(
            resolved["recognizer_path"], "models/ppocrv6-medium/rec.onnx"
        )
        self.assertEqual(
            resolved["classifier_path"], "models/ppocrv6-tiny/cls.onnx"
        )
        self.assertEqual(
            resolved["dictionary_path"],
            "models/ppocrv6-shared/PP-OCRv6_small_rec_dict.txt",
        )
        self.assertEqual(resolved["det_shapes"], [list(x) for x in ALLOWED_DET_SHAPES])
        self.assertEqual(resolved["rec_widths"], list(ALLOWED_WIDTHS))
        self.assertEqual(resolved["rec_max_width"], 960)
        self.assertEqual(resolved["expected_lines"], 16)
        self.assertEqual(len(resolved["expected_full_text_sha256"]), 64)

    def test_checksum_override_is_explicit_and_normalized(self) -> None:
        override = "A" * 64
        resolved = resolve_contract(CONTRACT, override)
        self.assertEqual(resolved["expected_full_text_sha256"], override.lower())

    def test_contract_rejects_unknown_fields(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        contract["unexpected"] = True
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "medium.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown"):
                resolve_contract(path)

    def test_contract_hash_matches_exact_bytes(self) -> None:
        resolved = resolve_contract(CONTRACT)
        expected = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
        self.assertEqual(resolved["contract_sha256"], expected)

    def test_contract_rejects_relaxed_rec_gate(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        changed = copy.deepcopy(contract)
        changed["rec_widths"] = [320, 960]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "medium.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "REC widths"):
                resolve_contract(path)

    def test_full_ocr_parser_preserves_line_order(self) -> None:
        output = (
            "lines=2 image=10x10 detector_input=10x10\n"
            "0 text=第一行 rec=1 det=1 cls=0/1 rotate=0 [(0,0),(1,0),(1,1),(0,1)]\n"
            "1 text=第二行 rec=1 det=1 cls=0/1 rotate=0 [(0,2),(1,2),(1,3),(0,3)]\n"
        )
        self.assertEqual(recognized_text(output), ["第一行", "第二行"])

    def test_gate_arguments_include_every_limit(self) -> None:
        arguments = command_gate_arguments(
            {
                "error_threshold": 1.0e-4,
                "max_abs_error": 3.0e-4,
                "max_mean_abs_error": 1.0e-6,
                "max_fraction_over": 1.0e-4,
            }
        )
        self.assertEqual(
            arguments[::2],
            [
                "--error-threshold",
                "--max-abs-error",
                "--max-mean-abs-error",
                "--max-fraction-over",
            ],
        )


if __name__ == "__main__":
    unittest.main()
