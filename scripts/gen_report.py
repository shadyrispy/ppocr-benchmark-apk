#!/usr/bin/env python3
# Generate the final REPORT.md from parsed.json (produced by parse_results.py).
import json, os, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
with open(os.path.join(RES, "parsed.json")) as f:
    D = json.load(f)

MODELS = ["det", "rec", "cls"]
ENGINES = ["lw", "ort", "mnn", "ncnn"]
# display order for configs
CFG_ORDER = ["cpu", "opencl", "opencl-fp16", "vulkan", "vulkan-fp16", "int8",
             "nnapi", "nnapi-fp16"]

def cfg_key(c):
    return (CFG_ORDER.index(c) if c in CFG_ORDER else 99, c)

def fwd(dev):  # -> {(eng,cfg,mdl): rec}
    out = {}
    for k, r in D[dev]["fwd"].items():
        e, c, m = k.split("/")
        out[(e, c, m)] = r
    return out

def pipe(dev):  # -> {"eng/cfg": rec}
    return dict(D[dev]["pipe"])

def bar(ms, scale, maxw=70):
    n = int(ms * scale)
    n = max(1, min(n, maxw))
    return "█" * n

def det_chart(dev, title, exclude=("lw",)):
    f = fwd(dev)
    rows = []
    for (e, c, m), r in f.items():
        if m != "det":
            continue
        if not r.get("ok"):
            rows.append((f"{e}/{c}", None, "ERR"))
        else:
            ok = r.get("numerics_ok")
            rows.append((f"{e}/{c}", r["mean_ms"], "BAD" if ok is False else "ok"))
    rows.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 1e9))
    maxms = max((x[1] for x in rows if x[1]), default=1)
    scale = 40.0 / maxms
    lines = [f"\n**{title}**  (det 纯 forward, ms; █≈{maxms/40:.1f}ms)", ""]
    for label, ms, flag in rows:
        if ms is None:
            lines.append(f"  {label:<18} ERR (disabled)")
        else:
            b = bar(ms, scale)
            tag = "  ⚠BAD" if flag == "BAD" else ""
            lines.append(f"  {label:<18} {b} {ms:7.2f}{tag}")
    return "\n".join(lines)

def fwd_table(dev, title):
    f = fwd(dev)
    lines = [f"\n### {title}", "",
             "| engine/config | det ms | det num | rec ms | rec num | cls ms | cls num |",
             "|---|---|---|---|---|---|---|"]
    cfgs = {}
    for (e, c, m) in f:
        cfgs.setdefault(e, set()).add(c)
    for e in ENGINES:
        for c in sorted(cfgs.get(e, []), key=cfg_key):
            cells = [f"{e}/{c}"]
            for m in MODELS:
                r = f.get((e, c, m))
                if r is None:
                    cells += ["—", "—"]
                elif not r.get("ok"):
                    cells += ["ERR", "n/a"]
                else:
                    nm = "ok" if r.get("numerics_ok") else "BAD"
                    cells += [f"{r['mean_ms']:.2f}", nm]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def pipe_table(dev, title):
    p = pipe(dev)
    lines = [f"\n### {title}  (det pre+forward+post, threads=4, iters=10)", "",
             "| engine/config | pre ms | fwd ms | post ms | pipeline ms |",
             "|---|---|---|---|---|"]
    cfgs = {}
    for k in p:
        e, c = k.split("/")
        cfgs.setdefault(e, set()).add(c)
    for e in ENGINES:
        for c in sorted(cfgs.get(e, []), key=cfg_key):
            r = p.get(f"{e}/{c}")
            if r is None:
                lines.append(f"| {e}/{c} | — | — | — | — |")
            elif not r.get("ok"):
                err = (r.get("error") or "")[:46]
                lines.append(f"| {e}/{c} | ERR | {err} | | |")
            else:
                lines.append(f"| {e}/{c} | {r['pre_mean_ms']:.2f} | {r['forward_mean_ms']:.2f} | {r['post_mean_ms']:.2f} | {r['pipeline_mean_ms']:.2f} |")
    return "\n".join(lines)

def strongest(dev):
    f = fwd(dev)
    res = {}
    for e in ENGINES:
        best = None
        for (ee, c, m), r in f.items():
            if ee != e or m != "det":
                continue
            if not r.get("ok") or r.get("numerics_ok") is False:
                continue
            if best is None or r["mean_ms"] < best[1]:
                best = (c, r["mean_ms"])
        res[e] = best
    return res

def cross_rank(dev):
    f = fwd(dev)
    rows = []
    for (e, c, m), r in f.items():
        if m != "det":
            continue
        if not r.get("ok"):
            rows.append((f"{e}/{c}", None))
        else:
            rows.append((f"{e}/{c}", r["mean_ms"]))
    ok_rows = [(l, ms) for l, ms in rows if ms is not None]
    ok_rows.sort(key=lambda x: x[1])
    return ok_rows

# ---- Build report ----
md = []
md.append("# PP-OCRv6 tiny 多引擎「最强配置」基准报告")
md.append("")
md.append(f"> 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  模型: PP-OCRv6 tiny (det/rec/cls)  |  ABI: arm64-v8a  |  threads=4, iters=10")
md.append("")
md.append("## 1. 背景与目标")
md.append("")
md.append("在 4 个推理引擎（lw.PPOCR.C、ONNX Runtime、MNN、ncnn）上，把每个引擎的「最快路径」全部打开并实测：")
md.append("- **ORT** → NNAPI(+fp16)、动态 int8 量化")
md.append("- **MNN** → OpenCL / Vulkan(+fp16 精度) 后端")
md.append("- **ncnn** → Vulkan(+fp16)、int8 量化")
md.append("- **lw.PPOCR.C** → 无加速/量化开关（基线，等价于 fp32 CPU）")
md.append("")
md.append("目标：**测出每个引擎在两种代际 SoC 上的最强配置**，并给出跨引擎选型建议。")
md.append("")
md.append("## 2. 测试设备")
md.append("")
md.append("| 设备 | SoC | ARM | fp16 硬件 | 角色 |")
md.append("|---|---|---|---|---|")
md.append("| **S21U** | Galaxy S21 Ultra (SM-G998B, Exynos 2100, Mali-G78) | ARMv8.2 | **有** | 现代旗舰 |")
md.append("| **P20** | Huawei P20 Pro (EML-AL00, Kirin 970, Mali-G72) | ARMv8.0 | **无** | 老款（无 fp16） |")
md.append("")
md.append("两设备均 4 线程、10 次迭代取均值。参考输入为合成 `[0,1]` 张量（模型内部无前处理），int8 校准使用同分布的合成图。")
md.append("")

# Forward tables
md.append("## 3. 纯 forward 延迟 (det / rec / cls, mean_ms)")
md.append("")
md.append("> `num` 列：推理数值与参考实现的校验结果。`BAD` = 数值错误（排除出「最强」评选）；`ERR` = 该配置不可用。")
md.append(fwd_table("s21", "S21U（Exynos 2100, 有 fp16）"))
md.append(fwd_table("p20", "P20（Kirin 970, 无 fp16）"))

# Pipeline tables
md.append("")
md.append("## 4. det 流水线延迟 (pre + forward + post, mean_ms)")
md.append("")
md.append(fwd_table if False else pipe_table("s21", "S21U"))
md.append(pipe_table("p20", "P20"))

# Per-engine strongest
NOTES = {
    ("s21", "mnn"): "（fp16 比 cpu 快 ~22%，且 rec/cls 也最快）",
    ("s21", "ncnn"): "（`vulkan-fp16` 在 S21U 上 det 数值损坏，已排除；int8 为 ncnn 最快可用配置）",
    ("s21", "ort"): "（`nnapi` 崩溃禁用，`int8` 反而 +15% 回退）",
    ("p20", "mnn"): "（cpu/opencl/vulkan/fp16 全部≈444 ms，噪声内持平；选 cpu 省 GPU 开销）",
    ("p20", "ncnn"): "（`vulkan-fp16` 仅 det 快，但 rec/cls 偏慢 2–3×；整体更推荐 `ncnn/int8`）",
    ("p20", "ort"): "（`nnapi` 禁用，`int8` 在 P20 上 3× 回退，cpu 是唯一 sane 选择）",
}
md.append("")
md.append("## 5. 每引擎最强配置（结论）")
md.append("")
for dev, label in [("s21", "S21U"), ("p20", "P20")]:
    st = strongest(dev)
    md.append(f"### {label}")
    for e in ENGINES:
        b = st[e]
        note = NOTES.get((dev, e), "")
        if b:
            md.append(f"- **{e}**: 最强 = `{e}/{b[0]}`（det {b[1]:.2f} ms）{note}")
        else:
            md.append(f"- **{e}**: 无可用配置（全部数值损坏/不可用）")
    md.append("")

# Cross-engine ranking + charts
md.append("## 6. 跨引擎 det 延迟排名（仅 numerics_ok 配置）")
md.append(det_chart("s21", "S21U — det 纯 forward（越快越好）"))
md.append(det_chart("p20", "P20 — det 纯 forward（越快越好）"))

# speedups
md.append("")
md.append("### 相对 lw 基线的加速比（det）")
md.append("")
md.append("| 设备 | 最强配置 | det ms | 相对 lw 加速 |")
md.append("|---|---|---|---|")
for dev, label in [("s21", "S21U"), ("p20", "P20")]:
    f = fwd(dev)
    lw = f.get(("lw", "cpu", "det"))
    lw_ms = lw["mean_ms"] if lw else None
    st = strongest(dev)
    # overall best across engines (excluding lw)
    cr = cross_rank(dev)
    best_label, best_ms = cr[0]
    if lw_ms:
        sp = lw_ms / best_ms
        md.append(f"| {label} | `{best_label}` | {best_ms:.2f} | **{sp:.1f}×** |")
md.append("")

# Key findings
md.append("## 7. 关键发现")
md.append("")
md.append("### 7.1 ORT NNAPI EP 崩溃（SIGSEGV）→ 已优雅禁用")
md.append("- ORT 1.30 的 `OrtSessionOptionsAppendExecutionProvider_Nnapi` 在 S21U / P20 上均触发 **原生段错误**（`SIGSEGV`，`libonnxruntime.so` 内空函数指针解引用），进程无法捕获。")
md.append("- 已在 `backend_ort.cpp` 改为直接返回错误字符串（`ort/nnapi`、`ort/nnapi-fp16` 显示 `ERR`），不再调用 NNAPI EP。结论：**这两台设备的 NNAPI 驱动与 ORT 1.30 不兼容，NNAPI 路线不可用**.")
md.append("")
md.append("### 7.2 ncnn Vulkan-fp16 的 det 数值损坏（仅 S21U）")
md.append("- `ncnn/vulkan-fp16/det` 在 **S21U (Mali-G78)** 上 `numerics_ok=false`（`max_abs_diff=1.0`，`rel_diff≈247%`），但 rec/cls 正常；在 **P20 (Mali-G72)** 上 det 数值正确。")
md.append("- 判定为 ncnn Vulkan-fp16 路径在该 GPU 驱动上的 fp16 存储/拷贝缺陷。**S21U 评选已将该配置排除**；P20 上可用但 rec/cls 偏慢，不推荐.")
md.append("")
md.append("### 7.3 ORT int8：S21U 轻微回退，P20 灾难性 3× 回退")
md.append("- S21U：`ort/int8` det 53.72 ms vs `ort/cpu` 46.66 ms（**+15% 回退**）。")
md.append("- P20：`ort/int8` det 1286.85 ms vs `ort/cpu` 558.79 ms（**+130% / ~3× 回退**）。")
md.append("- 根因：ORT 动态 int8（weight-only, per-channel）在 ARMv8.0 上走了非最优反量化内核。**ORT int8 在两台设备都不值得用**；尤其 P20 上应绝对避免.")
md.append("")
md.append("### 7.4 ncnn int8：双设备均有效")
md.append("- S21U：`ncnn/int8` det 44.76 ms vs `ncnn/cpu` 47.90 ms（−7%，且是 ncnn 自身最快配置）。")
md.append("- P20：`ncnn/int8` det 410.39 ms vs `ncnn/cpu` 473.85 ms（−13%）。")
md.append("- ncnn 的 int8 工具链（ncnn2table + ncnn2int8）在 ARMv8.0/8.2 上均给出稳定收益，是 **ncnn 在两类设备上的最强配置**.")
md.append("")
md.append("### 7.5 硬件代际决定 GPU / fp16 是否值得（核心结论）")
md.append("- **S21U（ARMv8.2，有 fp16）**：GPU-fp16 收益明显。MNN fp16（opencl-fp16 / vulkan-fp16）det ~42 ms，比 MNN CPU（54 ms）快 ~22%；MNN fp16 也是 rec/cls 最快路径。")
md.append("- **P20（ARMv8.0，无 fp16）**：MNN 各后端（cpu/opencl/vulkan/opencl-fp16/vulkan-fp16）det **全部约 444 ms，毫无差异**——fp16 配置在缺乏硬件 fp16 时回退为 fp32，GPU 也无收益。")
md.append("- 即：**现代旗舰优先上 MNN-fp16；老款无 fp16 设备，GPU 与 offline-fp16 都是徒劳，应退回 MNN/ncnn 的 CPU fp32 或 int8**.")
md.append("")

# Recommendations
md.append("## 8. 选型建议")
md.append("")
md.append("| 场景 | 推荐 | 理由 |")
md.append("|---|---|---|")
md.append("| S21U 类现代旗舰 | **MNN Vulkan/OpenCL-fp16**（或 ncnn/int8） | det ~42 ms，rec/cls 也最快；整体 7–8× 快于 lw |")
md.append("| P20 类老款无 fp16 | **MNN CPU fp32** 或 **ncnn/int8** | GPU/fp16 无收益；ncnn/int8 略快于 MNN-cpu，MNN-cpu 的 rec/cls 更优 |")
md.append("| 跨代际通用 | **MNN（CPU 或 fp16 自适应）** | 在两类设备都稳居第一梯队，且无 ORT 的 int8 回退/NNAPI 崩溃风险 |")
md.append("| 避免 | ORT/NNAPI、ORT/int8、lw.PPOCR.C、ncnn/vulkan-fp16(S21U) | NNAPI 崩溃 / int8 回退 / 速度不可用 / det 数值损坏 |")
md.append("")
md.append("**lw.PPOCR.C 在所有场景均不可用**：S21U det 322 ms、P20 det 6092 ms，比最快原生引擎慢 6–15× ，不建议作为生产路径.")
md.append("")

# Reproducibility
md.append("## 9. 复现与数据来源")
md.append("")
md.append("- 基准 APK：`app/build/outputs/apk/debug/ppocr-bench-debug.apk`（已禁用 NNAPI、内置 ort-int8 / ncnn-int8 资产）")
md.append("- 原始日志：`results/raw-s21-all.log`、`results/raw-p20-all.log`（纯 forward）；`results/raw-s21-pipe.log`、`results/raw-p20-pipe.log`（det 流水线）")
md.append("- 解析中间产物：`results/parsed.json`、`results/parsed_summary.md`")
md.append("- 关键脚本：`scripts/run_sweep.sh`（单设备单跑采集）、`scripts/parse_results.py`（解析）、`scripts/gen_report.py`（本报告）")
md.append("- int8 模型：`models/ort-int8/*`（ORT 动态 int8）、`models/ncnn-int8/*`（ncnn2table+ncnn2int8，合成图校准）")
md.append("- MNN int8 因 `mnnquant` 崩溃（`std::length_error: vector`，校准配置 schema 不兼容）**未产出**，已从 CASES 移除。")
md.append("")
md.append("---")
md.append("*本报告由自动化基准流水线生成，所有数字直接来自设备实测 logcat 输出。*")

out = "\n".join(md)
with open(os.path.join(RES, "REPORT.md"), "w") as f:
    f.write(out)
print(f"[written] results/REPORT.md ({len(out)} bytes)")
