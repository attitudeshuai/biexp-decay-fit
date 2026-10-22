#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闪烁体衰减曲线：指数分解。

用法：
    python3 src/decay.py --data data/decay.json --out out/report.json
"""
import argparse
import datetime as dt
import json
import math
import os


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyse(raw):
    samples = raw["samples"]
    ts = [float(s["t"]) for s in samples]
    ys = [float(s["y"]) for s in samples]

    # 单指数：先取对数再线性回归
    pts = [(t, y) for t, y in zip(ts, ys) if y > 1.0]
    n = len(pts)
    sx = sum(t for t, _ in pts)
    sy = sum(math.log(y) for _, y in pts)
    sxx = sum(t * t for t, _ in pts)
    sxy = sum(t * math.log(y) for t, y in pts)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    tau = -1.0 / slope if slope else 1.0
    a = math.exp((sy - slope * sx) / n)

    resid = 0.0
    for t, y in zip(ts, ys):
        resid += (y - a * math.exp(-t / tau)) ** 2
    rmse = math.sqrt(resid / len(ts))

    return {
        "n_total": len(ts),
        "tau1_ns": round(tau, 3),
        "a1_counts": round(a, 2),
        "baseline_counts": 0.0,
        "rmse_counts": round(rmse, 3),
        "generated_at": dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "meta": {"model": "monoexp"},
    }


def main():
    ap = argparse.ArgumentParser(description="衰减曲线分解")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = load(args.data)
    report = analyse(raw)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
