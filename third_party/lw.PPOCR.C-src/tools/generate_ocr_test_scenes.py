#!/usr/bin/env python3
"""Generate deterministic OCR scenes for local Small-model experiments.

The scenes are deliberately code-generated rather than AI-rendered: every
string is known exactly, while layout, contrast, rotation, and density vary.
PNG and PPM copies plus a JSON manifest are written to an ignored output
directory by default.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont


FONT_REGULAR: Path | None = None
FONT_BOLD: Path | None = None
REGULAR_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
)
BOLD_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Bold.otf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"),
)


def resolve_font(explicit: Path | None, candidates: tuple[Path, ...]) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"font file does not exist: {explicit}")
        return explicit
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise SystemExit(f"no CJK font found; pass --font explicitly (searched: {searched})")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    if FONT_REGULAR is None:
        raise RuntimeError("font paths have not been initialized")
    path = FONT_BOLD if bold and FONT_BOLD is not None else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size, index=0)


def text_size(draw: ImageDraw.ImageDraw, value: str, face: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), value, font=face)
    return box[2] - box[0], box[3] - box[1]


def paste_rotated_text(
    image: Image.Image,
    text: str,
    center: tuple[int, int],
    size: int,
    angle: float,
    fill: tuple[int, int, int],
    bold: bool = False,
) -> None:
    face = font(size, bold)
    probe = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    probe_draw = ImageDraw.Draw(probe)
    width, height = text_size(probe_draw, text, face)
    tile = Image.new("RGBA", (width + size * 2, height + size * 2), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((size, size // 2), text, font=face, fill=fill + (255,))
    tile = tile.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(tile, (int(center[0] - tile.width / 2), int(center[1] - tile.height / 2)))


def clean_receipt() -> tuple[Image.Image, list[dict[str, Any]]]:
    image = Image.new("RGB", (1200, 900), (250, 249, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((35, 30, 1165, 870), outline=(205, 205, 198), width=3)
    draw.text((78, 58), "华东仓库出货单", font=font(42, True), fill=(30, 30, 30))
    draw.text((82, 120), "订单号：SH-2026-0906-0188    日期：2026-09-06", font=font(24), fill=(60, 60, 60))
    draw.line((75, 170, 1125, 170), fill=(160, 160, 155), width=2)
    lines = [
        "客户名称：苏州明远电子有限公司",
        "收货地址：江苏省苏州市工业园区星湖街 88 号",
        "商品名称：工业视觉检测灯板",
        "规格型号：LV-24W-6500K / 批次 A17",
        "数量：128 件    单价：¥86.50    金额：¥11,072.00",
        "包装方式：防静电袋 + 五层瓦楞纸箱",
        "质检结果：合格（抽检 16 件，全部点亮）",
        "物流单号：YT2026090600188    联系电话：0512-6688-1200",
        "备注：请在收货后 24 小时内完成外观确认。",
    ]
    truth = []
    y = 215
    for index, value in enumerate(lines):
        face = font(30 if index in (0, 2) else 26, index in (0, 2))
        draw.text((92, y), value, font=face, fill=(28, 28, 28))
        truth.append({"text": value, "angle": 0, "x": 92, "y": y})
        y += 60
    draw.line((75, 790, 1125, 790), fill=(160, 160, 155), width=2)
    draw.text((92, 812), "经办人：王敏        复核：陈浩        第 1 / 1 页", font=font(24), fill=(70, 70, 70))
    return image, truth


def dense_mixed() -> tuple[Image.Image, list[dict[str, Any]]]:
    image = Image.new("RGB", (1600, 1000), (246, 241, 228))
    draw = ImageDraw.Draw(image)
    draw.text((70, 48), "设备巡检记录 / EQUIPMENT INSPECTION", font=font(38, True), fill=(38, 58, 70))
    draw.text((74, 106), "Line B · Shift 02 · 2026-09-06 14:35:18", font=font(24), fill=(70, 82, 88))
    rows = [
        "01 温度传感器 T-001    24.8 °C    NORMAL",
        "02 输送带速度 V-017    1.25 m/s   NORMAL",
        "03 电机电流 M-204       3.82 A     CHECK",
        "04 压力阀 P-012         0.64 MPa   NORMAL",
        "05 光电开关 S-088       ON         NORMAL",
        "06 轴承振动 B-033       1.8 mm/s   NORMAL",
        "07 冷却液液位           78 %       NORMAL",
        "08 安全门状态           CLOSED     NORMAL",
        "09 生产计数             001284     TOTAL",
        "10 维护工单             WO-2609067 OPEN",
    ]
    truth = []
    y = 190
    for index, value in enumerate(rows):
        x = 92 + (index % 2) * 760
        row_y = y + (index // 2) * 92
        draw.rounded_rectangle((x - 22, row_y - 18, x + 685, row_y + 50), radius=12, outline=(202, 192, 170), width=2)
        draw.text((x, row_y), value, font=font(26, index in (2, 9)), fill=(35, 45, 48))
        truth.append({"text": value, "angle": 0, "x": x, "y": row_y})
    draw.text((90, 720), "结论：设备运行稳定，建议对 M-204 安排下一班次复检。", font=font(30, True), fill=(50, 64, 70))
    truth.append({"text": "结论：设备运行稳定，建议对 M-204 安排下一班次复检。", "angle": 0, "x": 90, "y": 720})
    return image, truth


def low_contrast() -> tuple[Image.Image, list[dict[str, Any]]]:
    image = Image.new("RGB", (1400, 860), (222, 224, 220))
    draw = ImageDraw.Draw(image)
    for y in range(0, image.height, 24):
        shade = 218 + (y // 24) % 5
        draw.line((0, y, image.width, y), fill=(shade, shade + 1, shade - 1), width=1)
    lines = [
        "浅色背景文本测试：识别边界和低对比度字符",
        "仓位 A-07 / 货架 03 / 数量 056",
        "有效期：2027-12-31    批号：LC-0906",
        "请保持页面平整，避免阴影覆盖文字区域",
        "服务热线 400-820-7788    service@example.test",
        "最终确认：已完成入库登记。",
    ]
    truth = []
    y = 130
    for index, value in enumerate(lines):
        x = 95 + (index % 2) * 24
        shade = 92 + index * 5
        draw.text((x, y), value, font=font(32 if index == 0 else 28), fill=(shade, shade, shade))
        truth.append({"text": value, "angle": 0, "x": x, "y": y})
        y += 105
    return image, truth


def rotated_blocks() -> tuple[Image.Image, list[dict[str, Any]]]:
    image = Image.new("RGBA", (1500, 1000), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    truth = []
    horizontal = [
        "方向分类测试：主标题保持水平",
        "横排文本与旋转文本同时出现",
        "CLS rotation 180 degree sample",
    ]
    for index, value in enumerate(horizontal):
        x, y = 120, 100 + index * 90
        draw.text((x, y), value, font=font(34, index == 0), fill=(30, 30, 30))
        truth.append({"text": value, "angle": 0, "x": x, "y": y})
    paste_rotated_text(image, "竖排标签 A-17", (1250, 350), 34, 90, (35, 35, 35), True)
    truth.append({"text": "竖排标签 A-17", "angle": 90, "x": 1250, "y": 350})
    paste_rotated_text(image, "旋转 180 度 OCR", (760, 780), 36, 180, (35, 35, 35), False)
    truth.append({"text": "旋转 180 度 OCR", "angle": 180, "x": 760, "y": 780})
    return image.convert("RGB"), truth


def sparse_layout() -> tuple[Image.Image, list[dict[str, Any]]]:
    image = Image.new("RGB", (1800, 1100), (244, 247, 250))
    draw = ImageDraw.Draw(image)
    cards = [
        (120, 150, "项目名称：北斗数据采集终端", 34),
        (940, 170, "状态：已验收", 42),
        (160, 520, "版本 v2.6.14", 38),
        (1030, 540, "负责人：李晨", 34),
        (530, 850, "签收编号：SZ-AB-260906-09", 32),
    ]
    truth = []
    for x, y, value, size in cards:
        width = text_size(draw, value, font(size, True))[0] + 58
        height = size + 56
        draw.rounded_rectangle((x - 28, y - 20, x + width, y + height), radius=18, fill=(255, 255, 255), outline=(184, 199, 214), width=3)
        draw.text((x, y), value, font=font(size, True), fill=(35, 61, 88))
        truth.append({"text": value, "angle": 0, "x": x, "y": y})
    return image, truth


def long_lines() -> tuple[Image.Image, list[dict[str, Any]]]:
    image = Image.new("RGB", (2200, 900), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    lines = [
        "长文本压力测试：这是一段用于验证识别宽度自适应和文本顺序的中文句子，包含多个标点符号。",
        "English mixed line: adaptive REC width should preserve 2026-09-06 / OCR-REC-960 / 98.6%.",
        "第二行继续测试较长的文本内容，观察检测框是否完整覆盖整行以及识别结果是否发生截断。",
        "联系方式：400-820-7788；邮箱：support@example.test；地址：上海市浦东新区张江路 88 号。",
        "结尾行：Small dynamic REC width 192 320 480 640 960。",
    ]
    truth = []
    y = 100
    for index, value in enumerate(lines):
        size = 34 if index != 1 else 30
        draw.text((80, y), value, font=font(size, index == 0), fill=(25, 25, 25))
        truth.append({"text": value, "angle": 0, "x": 80, "y": y})
        y += 135
    return image, truth


SCENES = {
    "01_clean_receipt": clean_receipt,
    "02_dense_mixed": dense_mixed,
    "03_low_contrast": low_contrast,
    "04_rotated_blocks": rotated_blocks,
    "05_sparse_layout": sparse_layout,
    "06_long_lines": long_lines,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("build-model-foundation/ocr-scenes"))
    parser.add_argument("--font", type=Path, help="regular CJK font; auto-detected when omitted")
    parser.add_argument("--bold-font", type=Path, help="bold CJK font; regular font is reused when omitted")
    args = parser.parse_args()
    global FONT_REGULAR, FONT_BOLD
    FONT_REGULAR = resolve_font(args.font, REGULAR_FONT_CANDIDATES)
    FONT_BOLD = resolve_font(args.bold_font, BOLD_FONT_CANDIDATES) if args.bold_font else FONT_REGULAR
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(20260906)
    manifest: dict[str, Any] = {"schema_version": 1, "generator": "tools/generate_ocr_test_scenes.py", "scenes": []}
    for name, builder in SCENES.items():
        image, truth = builder()
        png_path = args.output_dir / f"{name}.png"
        ppm_path = args.output_dir / f"{name}.ppm"
        image.save(png_path, format="PNG", optimize=False)
        image.save(ppm_path, format="PPM")
        manifest["scenes"].append({"name": name, "png": png_path.name, "ppm": ppm_path.name, "width": image.width, "height": image.height, "lines": truth})
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output_dir": str(args.output_dir), "scenes": len(manifest["scenes"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
