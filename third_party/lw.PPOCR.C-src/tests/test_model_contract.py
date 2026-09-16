from __future__ import annotations

import unittest
from pathlib import Path

from tools.validate_model_contract import validate_model_contract


ROOT = Path(__file__).resolve().parents[1]


class ModelContractTests(unittest.TestCase):
    def test_bundled_tiny_contract(self) -> None:
        contract = validate_model_contract(ROOT / "models" / "ppocrv6-tiny")
        self.assertEqual(contract["status"], "ok")
        self.assertEqual(contract["family"], "PP-OCRv6")
        self.assertEqual(contract["variant"], "tiny")
        self.assertEqual(contract["rec_classes"], 6906)
        self.assertEqual(contract["dictionary_entries"], 6904)


if __name__ == "__main__":
    unittest.main()
