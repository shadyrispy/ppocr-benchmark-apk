#!/usr/bin/env python3
# Parse ppocr-bench raw logs into structured forward-latency + pipeline tables.
# Reads only the authoritative `forward X/Y/Z -> {...}` and `pipeline X/Y -> {...}` lines.
import json, re, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

def load(path):
    fwd = {}   # (engine, config, model) -> dict
    pipe = {}  # (engine, config) -> dict
    fwd_re = re.compile(r'forward (\w+)/([\w-]+)/(\w+) -> (\{.*\})$')
    pipe_re = re.compile(r'pipeline (\w+)/([\w-]+) -> (\{.*\})$')
    import re as _re
    # Benchmark may emit bare nan/inf (broken GPU output) -> sanitize to null.
    san = _re.compile(r'\b(nan|NaN|NAN|inf|Inf|Infinity|-inf|-Inf|-Infinity)\b')
    def clean(s):
        return san.sub('null', s)
    with open(path) as f:
        for line in f:
            m = fwd_re.search(line)
            if m:
                eng, cfg, model, js = m.groups()
                fwd[(eng, cfg, model)] = json.loads(clean(js))
                continue
            m = pipe_re.search(line)
            if m:
                eng, cfg, js = m.groups()
                pipe[(eng, cfg)] = json.loads(clean(js))
    return fwd, pipe

def fmt(v, unit="ms"):
    return f"{v:.2f}{('' if unit is None else '')}"

def main():
    s21_fwd, s21_pipe = load(os.path.join(RES, "raw-s21-all.log"))
    p20_fwd, p20_pipe = load(os.path.join(RES, "raw-p20-all.log"))
    s21_pfwd, s21_ppipe = load(os.path.join(RES, "raw-s21-pipe.log"))
    p20_pfwd, p20_ppipe = load(os.path.join(RES, "raw-p20-pipe.log"))

    models = ["det", "rec", "cls"]
    # union of configs per engine, ordered
    def configs(fwd):
        out = {}
        for (e, c, m) in fwd:
            out.setdefault(e, set()).add(c)
        return out

    s21_cfg = configs(s21_fwd)
    p20_cfg = configs(p20_fwd)

    # config display order
    order = ["cpu", "opencl", "opencl-fp16", "vulkan", "vulkan-fp16", "int8", "nnapi", "nnapi-fp16"]
    def sortea(cfgs):
        return sorted(cfgs, key=lambda c: (order.index(c) if c in order else 99, c))

    # Build markdown tables
    def fwd_table(fwd, title):
        lines = [f"\n### {title}", "",
                 "| engine/config | det (ms) | rec (ms) | cls (ms) | det num_ok | rec num_ok | cls num_ok |",
                 "|---|---|---|---|---|---|---|"]
        for eng in ["lw", "ort", "mnn", "ncnn"]:
            if eng not in fwd and eng not in [k[0] for k in fwd]:
                continue
            cfgs = sortea(s21_cfg.get(eng, [])) if False else None
            # use union of both devices' configs for that engine
            allcfg = set()
            for src in (s21_fwd, p20_fwd):
                for (e, c, m) in src:
                    if e == eng: allcfg.add(c)
            for c in sortea(allcfg):
                row = [f"{eng}/{c}"]
                ok = True
                for mdl in models:
                    rec = fwd.get((eng, c, mdl))
                    if rec is None:
                        row.append("—")
                    else:
                        row.append(f"{rec['mean_ms']:.2f}")
                        if rec.get("numerics_ok") is False:
                            ok = False
                # numerics flags
                nflags = []
                for mdl in models:
                    rec = fwd.get((eng, c, mdl))
                    nflags.append("ok" if (rec and rec.get("numerics_ok")) else ("BAD" if rec else "—"))
                row.append("/".join(nflags[:1]))
                # actually put num flags per model at end
                lines.append("| " + " | ".join(row) + " | " + " | ".join(nflags) + " |")
        return "\n".join(lines)

    # Simpler: dedicated per-device forward table with numerics inline
    def fwd_table2(fwd, cfg_src, title):
        lines = [f"\n### {title}", "",
                 "| engine/config | det ms | det num | rec ms | rec num | cls ms | cls num |",
                 "|---|---|---|---|---|---|---|"]
        engines = ["lw", "ort", "mnn", "ncnn"]
        for eng in engines:
            cfgs = set()
            for (e, c, m) in fwd:
                if e == eng: cfgs.add(c)
            for c in sortea(cfgs):
                cells = [f"{eng}/{c}"]
                for mdl in models:
                    rec = fwd.get((eng, c, mdl))
                    if rec is None:
                        cells += ["—", "—"]
                    elif not rec.get("ok"):
                        cells += ["ERR", "n/a"]
                    else:
                        nm = "ok" if rec.get("numerics_ok") else "BAD"
                        cells += [f"{rec['mean_ms']:.2f}", nm]
                lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    # Pipeline table
    def pipe_table(pipe, title):
        lines = [f"\n### {title}  (det pre+forward+post, threads=4, iters=10)", "",
                 "| engine/config | pre ms | fwd ms | post ms | pipeline ms |",
                 "|---|---|---|---|---|"]
        order2 = ["lw", "ort", "mnn", "ncnn"]
        allcfg = {}
        for (e, c) in pipe:
            allcfg.setdefault(e, set()).add(c)
        for eng in order2:
            for c in sortea(allcfg.get(eng, [])):
                r = pipe.get((eng, c))
                if r is None:
                    lines.append(f"| {eng}/{c} | — | — | — | — |")
                    continue
                if not r.get("ok"):
                    lines.append(f"| {eng}/{c} | ERR | {r.get('error','')[:40]} | | |")
                    continue
                lines.append(f"| {eng}/{c} | {r['pre_mean_ms']:.2f} | {r['forward_mean_ms']:.2f} | {r['post_mean_ms']:.2f} | {r['pipeline_mean_ms']:.2f} |")
        return "\n".join(lines)

    md = []
    md.append("# PP-OCRv6 tiny — 多引擎最强配置基准 (forward + pipeline)")
    md.append("\nDevices: **S21U** = Galaxy S21 Ultra (SM-G998B, Exynos 2100, ARMv8.2, 有 fp16); **P20** = Huawei P20 Pro (EML-AL00, Kirin 970, ARMv8.0, 无 fp16).")
    md.append("Threads=4, iters=10, ABI=arm64-v8a. forward = 纯推理 (det/rec/cls 分离); pipeline = det 流水线 pre+forward+post.")
    md.append(fwd_table2(s21_fwd, s21_cfg, "S21U 纯 forward 延迟 (mean_ms) + 数值校验"))
    md.append(fwd_table2(p20_fwd, p20_cfg, "P20 纯 forward 延迟 (mean_ms) + 数值校验"))
    md.append(pipe_table(s21_ppipe, "S21U det 流水线延迟"))
    md.append(pipe_table(p20_ppipe, "P20 det 流水线延迟"))

    # Helpers for ranking
    def best_det(fwd, engine):
        best = None
        for (e, c, m), r in fwd.items():
            if e == engine and m == "det" and r.get("numerics_ok"):
                if best is None or r["mean_ms"] < best[1]:
                    best = (c, r["mean_ms"])
        return best

    md.append("\n## 每引擎最强配置 (按 det 纯 forward mean_ms 排名，仅计入 numerics_ok)\n")
    for dev, fwd in [("S21U", s21_fwd), ("P20", p20_fwd)]:
        md.append(f"### {dev}")
        for eng in ["lw", "ort", "mnn", "ncnn"]:
            b = best_det(fwd, eng)
            if b:
                md.append(f"- **{eng}**: 最强 = `{eng}/{b[0]}` (det {b[1]:.2f} ms)")
            else:
                md.append(f"- **{eng}**: 无 numerics_ok 配置 (全部数值损坏或不可用)")
        md.append("")

    out = "\n".join(md)
    with open(os.path.join(RES, "parsed_summary.md"), "w") as f:
        f.write(out)
    print(out)
    # also dump structured json
    struct = {"s21": {"fwd": {f"{e}/{c}/{m}": r for (e,c,m),r in s21_fwd.items()},
                       "pipe": {f"{e}/{c}": r for (e,c),r in s21_ppipe.items()}},
              "p20": {"fwd": {f"{e}/{c}/{m}": r for (e,c,m),r in p20_fwd.items()},
                       "pipe": {f"{e}/{c}": r for (e,c),r in p20_ppipe.items()}}}
    with open(os.path.join(RES, "parsed.json"), "w") as f:
        json.dump(struct, f, indent=1)
    print("\n[written] results/parsed_summary.md and results/parsed.json", file=sys.stderr)

if __name__ == "__main__":
    main()
