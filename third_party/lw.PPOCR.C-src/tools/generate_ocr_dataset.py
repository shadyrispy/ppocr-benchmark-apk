#!/usr/bin/env python3
"""Generate a deterministic, metadata-backed local OCR stress dataset.

Generated images belong under build-local-data/ and are intentionally not
release assets. Real-font paths are supplied by the caller and are recorded by
basename plus SHA-256 so the manifest does not expose machine-specific paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


DEFAULT_TEXTS: tuple[tuple[str, str], ...] = (
    ("project", "lw.PPOCR.C 轻量级 C OCR 推理运行时"),
    ("project", "PP-OCRv6 Tiny Small Medium 模型验证"),
    ("long-line", "这是一段较长的项目文本用于测试 REC 自适应宽度和动态识别"),
    ("long-line", "单文件离线 HTML 支持图片 PDF 和 OCR 结果导出 JSON TXT"),
    ("mixed-cn-en", "DET CLS REC 三阶段流水线保持 UTF-8 文本和阅读顺序一致"),
    ("mixed-cn-en", "WASM Web SDK 与 C ABI 共用同一套 LWM v0.1 模型资产"),
    ("mixed-cn-en", "Android Java JNI Demo 支持 ARM64 图片 OCR 和本地模型缓存"),
    ("mixed-cn-en", "CSharp WinForms HTTP Server 与 Web Demo 使用统一 OCR 结果"),
    ("english", "lw.PPOCR.C full OCR regression and model contract validation"),
    ("english", "Standalone WASM HTML keeps OCR processing fully offline"),
    ("english", "Adaptive REC target width 192 320 480 640 960"),
    ("english", "AVX2 NEON LSX LASX and wasm128 SIMD backend"),
    ("identifier", "LWM-0.1 schema_version=1 wasmHostAbiVersion=1"),
    ("identifier", "det.lwm cls.lwm rec.lwm PP-OCRv6_small_rec_dict.txt"),
    ("identifier", "ARM64 LoongArch64 amd64 customer package"),
    ("identifier", "ocr-demo.html result.json result.txt 2026-09-07"),
    ("short-cn", "识别结果导出"),
    ("short-cn", "检测框与置信度"),
    ("short-cn", "阅读顺序测试"),
    ("short-cn", "方向分类 180 度"),
    ("short-cn", "PDF 页面 OCR"),
    ("short-cn", "模型缓存校验"),
)

ASCII_FALLBACK_TEXTS: tuple[tuple[str, str], ...] = (
    ("project", "lw.PPOCR.C OCR runtime"),
    ("project", "PP-OCRv6 Tiny Small Medium"),
    ("english", "Standalone WASM OCR demo"),
    ("english", "Adaptive REC width 960"),
    ("identifier", "det.lwm cls.lwm rec.lwm"),
    ("identifier", "LWM-0.1 ABI-1 JSON-1"),
    ("identifier", "ARM64 LoongArch64 amd64"),
)

CANVAS_SIZES: tuple[tuple[int, int], ...] = (
    (640, 480),
    (864, 976),
    (1024, 768),
    (1280, 960),
    (1536, 864),
    (1664, 480),
    (1792, 1392),
)
FONT_SIZES: tuple[int, ...] = (14, 16, 18, 22, 26, 30, 36, 44, 52)
CORE_ORIENTATIONS: tuple[int, ...] = (0, 0, 0, 0, 180, 180)
VERTICAL_ORIENTATIONS: tuple[int, ...] = (90, -90)


@dataclass(frozen=True)
class FontSpec:
    path: Path | None
    label: str
    sha256: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_pool_sha256(pool: Sequence[tuple[str, str]]) -> str:
    payload = json.dumps(
        list(pool), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def default_font_candidates() -> list[Path]:
    candidates = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "simsun.ttc",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    ]
    return list(dict.fromkeys(path for path in candidates if path.is_file()))


def resolve_fonts(font_paths: Sequence[Path]) -> list[FontSpec]:
    paths = list(font_paths) if font_paths else default_font_candidates()
    if not paths:
        return [FontSpec(None, "Pillow-default", None)]
    specs: list[FontSpec] = []
    for path in paths:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"font file not found: {path}")
        specs.append(FontSpec(path, path.name, sha256_file(path)))
    return specs


def load_font(spec: FontSpec, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if spec.path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(spec.path), size=size)


def render_text(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    orientation: int,
    angle: float,
    color: tuple[int, int, int],
) -> tuple[Image.Image, int]:
    left, top, right, bottom = font.getbbox(text)
    padding = 8
    width = max(1, right - left) + padding * 2
    height = max(1, bottom - top) + padding * 2
    sprite = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(sprite).text(
        (padding - left, padding - top),
        text,
        font=font,
        fill=(*color, 255),
    )
    if orientation:
        sprite = sprite.rotate(orientation, expand=True, resample=Image.Resampling.BICUBIC)
    if abs(angle) > 0.001:
        sprite = sprite.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    natural_width = max(1, right - left) * 48 // max(1, bottom - top)
    return sprite, natural_width


def fit_sprite(
    sprite: Image.Image,
    canvas: tuple[int, int],
    margin: int = 12,
    max_height: int | None = None,
) -> Image.Image:
    max_width = max(1, canvas[0] - margin * 2)
    height_limit = max(1, canvas[1] - margin * 2)
    if max_height is not None:
        height_limit = min(height_limit, max(1, max_height))
    scale = min(1.0, max_width / sprite.width, height_limit / sprite.height)
    if scale >= 1.0:
        return sprite
    size = (max(1, int(sprite.width * scale)), max(1, int(sprite.height * scale)))
    return sprite.resize(size, Image.Resampling.LANCZOS)


def rect_intersection(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> int:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def place_sprite(
    rng: random.Random,
    canvas: tuple[int, int],
    sprite: Image.Image,
    occupied: list[tuple[int, int, int, int]],
    gap: int = 4,
) -> tuple[int, int] | None:
    width, height = canvas
    max_x = max(0, width - sprite.width)
    max_y = max(0, height - sprite.height)
    for _ in range(256):
        x = rng.randint(0, max_x)
        y = rng.randint(0, max_y)
        guarded = (x - gap, y - gap, x + sprite.width + gap, y + sprite.height + gap)
        if all(rect_intersection(guarded, previous) == 0 for previous in occupied):
            return x, y

    # Random placement is quick for the common case. Scan a deterministic grid
    # before giving up so a crowded image never falls back to overlapping text.
    step = max(8, min(24, min(sprite.width, sprite.height) // 2))
    x_positions = list(range(0, max_x + 1, step))
    y_positions = list(range(0, max_y + 1, step))
    if not x_positions or x_positions[-1] != max_x:
        x_positions.append(max_x)
    if not y_positions or y_positions[-1] != max_y:
        y_positions.append(max_y)
    for y in y_positions:
        for x in x_positions:
            guarded = (x - gap, y - gap, x + sprite.width + gap, y + sprite.height + gap)
            if all(rect_intersection(guarded, previous) == 0 for previous in occupied):
                return x, y
    return None


def generate_dataset(
    output: Path,
    count: int,
    seed: int,
    font_paths: Sequence[Path] = (),
    image_format: str = "jpg",
    text_pool: Iterable[tuple[str, str]] = DEFAULT_TEXTS,
    force: bool = False,
    include_vertical: bool = False,
) -> dict[str, object]:
    if count <= 0:
        raise ValueError("count must be positive")
    if image_format not in ("jpg", "png"):
        raise ValueError("image_format must be 'jpg' or 'png'")
    output = output.resolve()
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty; use --force: {output}")
    output.mkdir(parents=True, exist_ok=True)
    specs = resolve_fonts(font_paths)
    pool = tuple(text_pool)
    if not pool:
        raise ValueError("text pool must not be empty")
    if all(spec.path is None for spec in specs) and pool == DEFAULT_TEXTS:
        pool = ASCII_FALLBACK_TEXTS
    rng = random.Random(seed)
    orientations = (
        CORE_ORIENTATIONS + VERTICAL_ORIENTATIONS
        if include_vertical
        else CORE_ORIENTATIONS
    )
    corpus_id = (
        "lw-ppocr-c-project-v1-vertical"
        if include_vertical
        else "lw-ppocr-c-project-v1"
    )
    image_records: list[dict[str, object]] = []
    for image_index in range(1, count + 1):
        width, height = rng.choice(CANVAS_SIZES)
        background = tuple(rng.randint(12, 248) for _ in range(3))
        base = Image.new("RGB", (width, height), background)
        occupied: list[tuple[int, int, int, int]] = []
        line_records: list[dict[str, object]] = []
        line_count = rng.randint(4, 8)
        placed_line_count = 0
        pending = []
        for _ in range(line_count):
            category, text = rng.choice(pool)
            font_spec = rng.choice(specs)
            font_size = rng.choice(FONT_SIZES)
            orientation = rng.choice(orientations)
            angle = round(rng.uniform(-15.0, 15.0), 3)
            brightness = sum(background) / 3.0
            if brightness < 128:
                color = tuple(rng.randint(180, 255) for _ in range(3))
            else:
                color = tuple(rng.randint(0, 80) for _ in range(3))
            font = load_font(font_spec, font_size)
            sprite, natural_width = render_text(text, font, orientation, angle, color)
            sprite = fit_sprite(
                sprite,
                (width, height),
                max_height=height // max(3, line_count),
            )
            pending.append(
                (
                    sprite,
                    natural_width,
                    category,
                    text,
                    font_spec,
                    font_size,
                    angle,
                    orientation,
                    color,
                )
            )

        # Packing large blocks first leaves more usable space for the shorter
        # lines and avoids the sparse images caused by greedy random order.
        pending.sort(key=lambda item: item[0].width * item[0].height, reverse=True)
        for (
            sprite,
            natural_width,
            category,
            text,
            font_spec,
            font_size,
            angle,
            orientation,
            color,
        ) in pending:
            position = place_sprite(rng, (width, height), sprite, occupied)
            if position is None:
                # Keep the image useful and truthful rather than painting a
                # line over an existing one when the requested density is too
                # high for this canvas/font combination.
                continue
            x, y = position
            occupied.append((x, y, x + sprite.width, y + sprite.height))
            base.paste(sprite, (x, y), sprite)
            placed_line_count += 1
            line_records.append(
                {
                    "text": text,
                    "bbox": [x, y, x + sprite.width, y + sprite.height],
                    "natural_width_at_height_48": natural_width,
                    "angle_degrees": angle,
                    "orientation_degrees": orientation,
                    "font": font_spec.label,
                    "font_size": font_size,
                    "color_rgb": list(color),
                    "category": category,
                }
            )
        suffix = "jpg" if image_format == "jpg" else "png"
        image_path = output / f"img-{image_index:03d}.{suffix}"
        if image_format == "jpg":
            base.save(image_path, format="JPEG", quality=95, subsampling=0, optimize=False, progressive=False)
        else:
            base.save(image_path, format="PNG", optimize=False)
        image_records.append(
            {
                "file": image_path.name,
                "width": width,
                "height": height,
                "background_rgb": list(background),
                "requested_line_count": line_count,
                "placed_line_count": placed_line_count,
                "sha256": sha256_file(image_path),
                "lines": line_records,
            }
        )
    manifest = {
        "version": 1,
        "seed": seed,
        "generator": {
            "name": "lw.PPOCR.C",
            "tool": "tools/generate_ocr_dataset.py",
            "corpus_id": corpus_id,
            "text_pool_sha256": text_pool_sha256(pool),
            "orientation_policy": "0/180/90/-90" if include_vertical else "0/180",
            "renderer": "Pillow",
            "renderer_version": Image.__version__,
            "image_format": image_format,
            "fonts": [
                {"file": spec.label, "sha256": spec.sha256} for spec in specs
            ],
        },
        "images": image_records,
    }
    (output / "metadata.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build-local-data/lw-generated-ocr"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--font", type=Path, action="append", default=[])
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg")
    parser.add_argument(
        "--include-vertical",
        action="store_true",
        help="include 90/-90 degree vertical text as a separate stress corpus",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = generate_dataset(
        args.output,
        args.count,
        args.seed,
        args.font,
        args.format,
        force=args.force,
        include_vertical=args.include_vertical,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "images": len(manifest["images"]),
                "seed": args.seed,
                "format": args.format,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
