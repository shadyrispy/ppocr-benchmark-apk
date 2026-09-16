from __future__ import annotations

import hashlib
import pathlib
import unittest

from tools.validate_model_manifest import validate_manifest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "ppocrv6-tiny"


class TinyModelManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = validate_manifest(MODEL_DIR)

    def test_manifest_identity_and_schema(self) -> None:
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(self.manifest["family"], "PP-OCRv6")
        self.assertEqual(self.manifest["variant"], "tiny")
        self.assertEqual(self.manifest["runtime_status"], "primary")
        self.assertEqual(set(self.manifest["models"]), {"det", "cls", "rec"})

    def test_manifest_checksums_match_assets(self) -> None:
        for relative, expected in self.manifest["checksums"].items():
            digest = hashlib.sha256((MODEL_DIR / relative).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, relative)


if __name__ == "__main__":
    unittest.main()
