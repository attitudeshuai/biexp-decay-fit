#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闪烁体衰减曲线复核：双指数分解（两个时间常数 + 基线）。

命令行：
    python3 src/decay.py --data data/decay.json --out out/report.json

口径：
  · 模型 y(t) = A1·exp(-t/τ1) + A2·exp(-t/τ2) + c，约定 τ1 < τ2；
  · 探测器上限写在同一份记录的 instrument.saturation_counts 里，读数落在上限附近
    （差不超过 saturation_margin）的点是被限幅的，必须整点剔除、不参与拟合；
  · τ1、τ2 先在二维网格上找残差平方和最小，再分别用黄金分割细化；
    给定两个时间常数时 A1、A2、c 是线性最小二乘（三阶方程闭式解）；
  · rmse_counts = sqrt(残差平方和 /(n - 4))，用来判断这次拟合是否到噪声量级；
  · 输出先写临时文件再原子替换；非法输入以退出码 2 结束且不落文件。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

EXIT_INVALID = 2


def _fail(msg):
    sys.stderr.write("非法输入：%s\n" % msg)
    raise SystemExit(EXIT_INVALID)


def load_data(path):
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        _fail("读不到记录 %s" % path)
    except ValueError as e:
        _fail("记录不是合法 JSON：%s" % e)
    if not isinstance(raw, dict):
        _fail("记录顶层要是对象")
    inst = raw.get("instrument")
    if not isinstance(inst, dict):
        _fail("缺 instrument")
    sat = inst.get("saturation_counts")
    if not isinstance(sat, (int, float)) or isinstance(sat, bool) or float(sat) <= 0:
        _fail("instrument.saturation_counts 要是正数")
    cfg = raw.get("fit") or {}
    margin = cfg.get("saturation_margin", 0.5)
    min_used = cfg.get("min_used", 40)
    if not isinstance(margin, (int, float)) or isinstance(margin, bool) or float(margin) < 0:
        _fail("fit.saturation_margin 不能为负")
    if not isinstance(min_used, int) or isinstance(min_used, bool) or min_used < 20:
        _fail("fit.min_used 要是不小于 20 的整数")

    samples = raw.get("samples")
    if not isinstance(samples, list) or len(samples) < 100:
        _fail("samples 至少要 100 个采样点")
    times, ys = [], []
    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            _fail("samples[%d] 不是对象" % i)
        try:
            t = float(s["t"])
            y = float(s["y"])
        except (KeyError, TypeError, ValueError):
            _fail("samples[%d] 的 t/y 必须是数值" % i)
        if not (math.isfinite(t) and math.isfinite(y)):
            _fail("samples[%d] 有非有限数值" % i)
        if t < 0 or (times and t <= times[-1]):
            _fail("samples[%d].t 必须单调递增且非负" % i)
        times.append(t)
        ys.append(y)
    return {"times": times, "ys": ys, "sat": float(sat), "margin": float(margin),
            "min_used": int(min_used), "unit": (raw.get("unit") or {})}


def keep_unsaturated(ys, sat, margin):
    limit = sat - margin
    return [i for i, y in enumerate(ys) if y < limit]


# ---------------------------------------------------------------- 线性代数
def solve3(a, b):
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-14:
            return None
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for k in range(col, 4):
            m[col][k] /= pv
        for r in range(3):
            if r != col:
                f = m[r][col]
                for k in range(col, 4):
                    m[r][k] -= f * m[col][k]
    return [m[0][3], m[1][3], m[2][3]]


def _linear_fit(ts, ys, tau1, tau2):
    """给定两个时间常数，解 A1、A2、c 的最小二乘。"""
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for t, y in zip(ts, ys):
        x = [math.exp(-t / tau1), math.exp(-t / tau2), 1.0]
        for i in range(3):
            xty[i] += x[i] * y
            for j in range(3):
                xtx[i][j] += x[i] * x[j]
    return solve3(xtx, xty)


def _sse(ts, ys, tau1, tau2):
    coef = _linear_fit(ts, ys, tau1, tau2)
    if coef is None:
        return None, None
    acc = 0.0
    for t, y in zip(ts, ys):
        m = coef[0] * math.exp(-t / tau1) + coef[1] * math.exp(-t / tau2) + coef[2]
        acc += (y - m) ** 2
    return acc, coef


def _golden(fn, lo, hi, iters=60):
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iters):
        if fc is None or fd is None:
            break
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = fn(d)
    return (a + b) / 2.0


def fit_biexp(ts, ys):
    t_min = min(ts)
    t_max = max(ts)
    span = max(t_max - t_min, 1e-9)
    t1_lo, t1_hi = span / 500.0, span / 5.0
    best = None
    for i in range(41):
        tau1 = t1_lo * (t1_hi / t1_lo) ** (i / 40.0)
        for j in range(41):
            tau2 = tau1 * 1.2 * (200.0 ** (j / 40.0))
            if tau2 <= tau1:
                continue
            sse, coef = _sse(ts, ys, tau1, tau2)
            if sse is not None and (best is None or sse < best[0]):
                best = (sse, tau1, tau2)
    tau1, tau2 = best[1], best[2]
    for _ in range(8):
        tau1 = _golden(lambda x: _sse(ts, ys, x, tau2)[0],
                       max(t1_lo, tau1 / 1.6), min(tau2 / 1.05, tau1 * 1.6))
        tau2 = _golden(lambda x: _sse(ts, ys, tau1, x)[0],
                       max(tau1 * 1.05, tau2 / 1.6), min(tau1 * 200.0, tau2 * 1.6))
    sse, coef = _sse(ts, ys, tau1, tau2)
    return {"tau1": tau1, "tau2": tau2, "a1": coef[0], "a2": coef[1],
            "base": coef[2], "sse": sse}


def analyse(scan):
    idx = keep_unsaturated(scan["ys"], scan["sat"], scan["margin"])
    if len(idx) < scan["min_used"]:
        _fail("剔除饱和点后只剩 %d 个有效点，少于要求的 %d" % (len(idx), scan["min_used"]))
    ts = [scan["times"][i] for i in idx]
    ys = [scan["ys"][i] for i in idx]
    fit = fit_biexp(ts, ys)
    dof = len(ts) - 4
    if dof <= 0:
        _fail("有效点太少，自由度不够")
    fit["rmse"] = math.sqrt(fit["sse"] / dof)
    fit["n_used"] = len(ts)
    fit["n_total"] = len(scan["times"])
    fit["dof"] = dof
    return fit


def build_report(scan, res):
    ratio = res["a1"] / res["a2"] if abs(res["a2"]) > 1e-12 else 0.0
    return {
        "n_total": res["n_total"],
        "n_used": res["n_used"],
        "dof": res["dof"],
        "tau1_ns": round(res["tau1"], 3),
        "tau2_ns": round(res["tau2"], 3),
        "a1_counts": round(res["a1"], 2),
        "a2_counts": round(res["a2"], 2),
        "baseline_counts": round(res["base"], 2),
        "amplitude_ratio": round(ratio, 3),
        "rmse_counts": round(res["rmse"], 3),
        "meta": {"model": "biexp", "unit": scan["unit"]},
    }


def atomic_write(path, obj):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    body = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".dec-", suffix=".tmp", dir=out_dir or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description="双指数衰减分解")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    scan = load_data(args.data)
    res = analyse(scan)
    atomic_write(args.out, build_report(scan, res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
