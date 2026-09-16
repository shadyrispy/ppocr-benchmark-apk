#!/usr/bin/env python3
"""Create a release archive containing the catalogued PP-OCRv6 model assets."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.validate_model_catalog import validate_catalog


def package(catalog_path: Path, output_path: Path) -> dict[str, object]:
    report = validate_catalog(catalog_path)
    root = catalog_path.parent
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    members: dict[str, Path] = {"models/ppocrv6-models.json": catalog_path}
    for descriptor in catalog["shared_assets"].values():
        relative = descriptor["path"]
        members[f"models/{relative}"] = root / relative
    for variant in catalog["variants"].values():
        for role in ("det", "rec"):
            relative = variant[role]["path"]
            members[f"models/{relative}"] = root / relative
        dictionary = variant["dictionary"]
        if isinstance(dictionary, dict):
            relative = dictionary["path"]
            members[f"models/{relative}"] = root / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, source in sorted(members.items()):
            archive.write(source, name)
    return {"status": report["status"], "output": str(output_path), "members": sorted(members)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("models/ppocrv6-models.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(package(args.catalog, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
