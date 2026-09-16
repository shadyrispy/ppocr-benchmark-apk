from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.validate_small_ocr_baseline import verify_model_identities


class SmallOcrBaselineIdentityTests(unittest.TestCase):
    def test_model_hashes_must_match_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {role: root / f"{role}.bin" for role in ("detector", "classifier", "recognizer", "dictionary")}
            for index, path in enumerate(paths.values()):
                path.write_bytes(bytes([index + 1]) * (index + 3))
            baseline = {
                "models": {
                    role: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for role, path in paths.items()
                }
            }
            verify_model_identities(baseline, paths)
            paths["recognizer"].write_bytes(b"changed")
            with self.assertRaisesRegex(SystemExit, "recognizer SHA-256 mismatch"):
                verify_model_identities(baseline, paths)

    def test_missing_model_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            path.write_bytes(b"model")
            with self.assertRaisesRegex(SystemExit, "missing SHA-256"):
                verify_model_identities(
                    {"models": {"detector": {}}}, {"detector": path}
                )


if __name__ == "__main__":
    unittest.main()
