#!/usr/bin/env python3
"""Summarise bench-*.json / logcat dumps into a comparison table."""
import json
import os
import re
import sys

ENGINE_LABEL = {
    "lw": "lw.PPOCR.C",
    "ort": "ONNX Runtime",
    "mnn": "MNN",
    "ncnn": "ncnn",
}


def parse(path):
    rows = []
    text = open(path, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r'\{"ok".*?\}', text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not obj.get("ok"):
            rows.append(dict(engine=obj.get("engine", "?"), model="?", error=obj.get("error")))
            continue
        rows.append(obj)
    return rows


def main():
    files = sys.argv[1:]
    for f in files:
        name = os.path.basename(f)
        rows = parse(f)
        print(f"\n############ {name} ############")
        if not rows:
            print("  no results parsed")
            continue
        # dedupe: the app logs each case twice
        seen = set()
        uniq = []
        for r in rows:
            key = (r.get("engine"), r.get("model"), r.get("threads"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        hdr = f"{'engine':<13}{'model':<6}{'thr':>3}{'load_ms':>10}{'mean_ms':>11}{'median':>10}{'p95':>10}{'min':>10}{'max':>11}{'fps':>9}{'maxdiff':>10}  ok"
        # Pipeline rows carry pre/forward/post instead of the single-model
        # timing block, so they get their own table.
        if any("pipeline_mean_ms" in r for r in uniq):
            ph = (f"{'engine':<13}{'thr':>4}{'pre_ms':>10}{'fwd_ms':>10}"
                  f"{'post_ms':>10}{'pipeline_ms':>13}{'fps':>9}")
            print(ph)
            print("-" * len(ph))
            for r in uniq:
                if "error" in r:
                    print(f"{r['engine']:<13}  ERROR: {r['error']}")
                    continue
                tot = r["pipeline_mean_ms"]
                print("{:<13}{:>4}{:>10.2f}{:>10.2f}{:>10.2f}{:>13.2f}{:>9.2f}".format(
                    ENGINE_LABEL.get(r["engine"], r["engine"]), r["threads"],
                    r["pre_mean_ms"], r["forward_mean_ms"], r["post_mean_ms"],
                    tot, 1000.0 / tot if tot > 0 else 0.0))
            continue
        print(hdr)
        print("-" * len(hdr))
        for r in uniq:
            if "error" in r:
                print(f"{r['engine']:<13}{r['model']:<6}  ERROR: {r['error']}")
                continue
            print("{:<13}{:<6}{:>3}{:>10.1f}{:>11.2f}{:>10.2f}{:>10.2f}{:>10.2f}{:>11.2f}{:>9.2f}{:>10.4f}  {}".format(
                ENGINE_LABEL.get(r["engine"], r["engine"]), r["model"], r["threads"],
                r["load_ms"], r["mean_ms"], r["median_ms"], r["p95_ms"], r["min_ms"],
                r["max_ms"], r["throughput_fps"], r["max_abs_diff"], r["numerics_ok"]))


if __name__ == "__main__":
    main()
