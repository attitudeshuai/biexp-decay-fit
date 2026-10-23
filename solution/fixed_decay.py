#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闪烁体衰减曲线复核：双指数分解（两个时间常数 + 基线 + 堆积系数）。

命令行：
    python3 src/decay.py --data data/decay.json --out out/report.json

口径：
  · 真实波形 u(t) = A1·exp(-t/τ1) + A2·exp(-t/τ2) + c，约定 τ1 < τ2；
  · 读出电路有堆积，模型写在 instrument.pileup_model 里：非瘫痪型 y = u/(1 + k·u)，
    瘫痪型 y = u·exp(-k·u)（反解要牛顿迭代）；k 是堆积系数（每计数），
    记录里不给，要和曲线一起估出来；
  · 记录若带 calibration（若干「真计数率 → 读数」标定点），堆积系数必须先在标定点上
    定出来（一维黄金分割），再反解曲线拟合两个时间常数；没有标定点时才连 k 一起联合拟合；
  · 探测器上限写在 instrument.saturation_counts 里，读数落在上限附近（差不超过
    saturation_margin）的点是被限幅的，必须整点剔除、不参与拟合；
  · 给定 k 时把读数反解成 u = y/(1 - k·y)，此时 A1、A2、c 是线性最小二乘
    （三阶方程闭式解）；k 先在网格上定位，再与 τ1、τ2 轮流用黄金分割细化；
  · rmse_counts = sqrt(Σ(实测 - 模型)²/(n - 6))，残差在实测域上算；
  · 复核缓存按记录字节的 sha256 命中复用，未命中追加；缓存最多 32 条，超出先丢最早的；
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
CACHE_LIMIT = 32


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
    if not isinstance(sat, (int, float)) or isinstance(sat, bool) or not (float(sat) > 0):
        _fail("instrument.saturation_counts 要是正数")
    cfg = raw.get("fit") or {}
    margin = cfg.get("saturation_margin", 0.5)
    min_used = cfg.get("min_used", 40)
    if not isinstance(margin, (int, float)) or isinstance(margin, bool) or float(margin) < 0:
        _fail("fit.saturation_margin 不能为负")
    if not isinstance(min_used, int) or isinstance(min_used, bool) or min_used < 20:
        _fail("fit.min_used 要是不小于 20 的整数")

    model = (inst.get("pileup_model") or "nonparalyzable")
    if model not in ("nonparalyzable", "paralyzable"):
        _fail("instrument.pileup_model 只认 nonparalyzable / paralyzable")

    calib = []
    raw_cal = raw.get("calibration")
    if raw_cal is not None:
        if not isinstance(raw_cal, list) or len(raw_cal) < 3:
            _fail("calibration 要是至少 3 个标定点的列表")
        for i, c in enumerate(raw_cal):
            if not isinstance(c, dict):
                _fail("calibration[%d] 不是对象" % i)
            try:
                u_true = float(c["true_counts"])
                y_obs = float(c["observed"])
            except (KeyError, TypeError, ValueError):
                _fail("calibration[%d] 的 true_counts/observed 必须是数值" % i)
            if not (math.isfinite(u_true) and math.isfinite(y_obs)) or u_true <= 0 or y_obs <= 0:
                _fail("calibration[%d] 的取值必须为正的有限数" % i)
            calib.append((u_true, y_obs))

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
            "min_used": int(min_used), "unit": (raw.get("unit") or {}), "model": model,
            "calibration": calib}


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


def _is_paralyzable(model):
    """记录声明的堆积模型是不是瘫痪型。"""
    return model == "paralyzable"


def _invert_one(y, k, model):
    """把读数反解回真实计数。非瘫痪型闭式；瘫痪型要用不动点/牛顿迭代。"""
    if not _is_paralyzable(model):
        d = 1.0 - k * y
        if d < 0.02:
            return None
        return y / d
    # 瘫痪型：解 u·exp(-k·u) = y，牛顿法，初值取 y
    u = y
    for _ in range(200):
        e = math.exp(-k * u)
        f = u * e - y
        df = e * (1.0 - k * u)
        if df <= 0:
            return None
        step = f / df
        u -= step
        if u < 0:
            u = y * 0.5
        if abs(step) <= 1e-13 * max(1.0, abs(u)):
            break
    if not math.isfinite(u) or u <= 0:
        return None
    return u


def _invert(ys, k, model="nonparalyzable"):
    out = []
    for y in ys:
        v = _invert_one(y, k, model)
        if v is None:
            return None
        out.append(v)
    return out


def _forward(u, k, model):
    """真实计数过一遍读出堆积后的读数。"""
    if _is_paralyzable(model):
        return u * math.exp(-k * u)
    return u / (1.0 + k * u)


def _model_obs(t, tau1, tau2, a1, a2, base, k, model="nonparalyzable"):
    """模型在实测域上的取值：u 先算出来，再走一遍堆积。"""
    u = a1 * math.exp(-t / tau1) + a2 * math.exp(-t / tau2) + base
    return _forward(u, k, model)


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


def _solve(a, b):
    """通用线性方程组：高斯消元带主元。"""
    n = len(a)
    m = [a[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-18:
            return None
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for k in range(col, n + 1):
            m[col][k] /= pv
        for r in range(n):
            if r != col:
                f = m[r][col]
                for k in range(col, n + 1):
                    m[r][k] -= f * m[col][k]
    return [m[i][n] for i in range(n)]


def _sse_params(ts, ys, p, model="nonparalyzable"):
    t1, t2, a1, a2, c, k = p
    if t1 <= 0 or t2 <= t1 or k < 0:
        return None
    acc = 0.0
    for t, y in zip(ts, ys):
        u = a1 * math.exp(-t / t1) + a2 * math.exp(-t / t2) + c
        if u <= 0:
            return None
        acc += (y - _forward(u, k, model)) ** 2
    return acc


def _refine(ts, ys, p, model="nonparalyzable", iters=120):
    """实测域残差上的 Levenberg-Marquardt 精修，六个参数一起动。"""
    p = list(p)
    sse = _sse_params(ts, ys, p, model)
    if sse is None:
        return None
    lam = 1e-3
    for _ in range(iters):
        t1, t2, a1, a2, c, k = p
        jtj = [[0.0] * 6 for _ in range(6)]
        jtr = [0.0] * 6
        for t, y in zip(ts, ys):
            e1 = math.exp(-t / t1)
            e2 = math.exp(-t / t2)
            u = a1 * e1 + a2 * e2 + c
            if model == "nonparalyzable":
                w = 1.0 / (1.0 + k * u)
                g = w * w
                m = u * w
            else:
                e = math.exp(-k * u)
                g = e * (1.0 - k * u)
                m = u * e
            # dy/d参数；对 k 的偏导两种模型都等于 -u²·g
            jac = [g * a1 * e1 * t / (t1 * t1),
                   g * a2 * e2 * t / (t2 * t2),
                   g * e1, g * e2, g, -u * u * g]
            r = y - m
            for i in range(6):
                jtr[i] += jac[i] * r
                for j in range(6):
                    jtj[i][j] += jac[i] * jac[j]
        ok = False
        while lam < 1e8:
            aug = [row[:] for row in jtj]
            for i in range(6):
                aug[i][i] *= (1.0 + lam)
                aug[i][i] += 1e-12
            step = _solve(aug, jtr)
            if step is None:
                lam *= 10.0
                continue
            trial = [p[i] + step[i] for i in range(6)]
            trial[5] = max(0.0, trial[5])
            new = _sse_params(ts, ys, trial, model)
            if new is not None and new < sse:
                p = trial
                sse = new
                lam = max(lam / 3.0, 1e-10)
                ok = True
                break
            lam *= 4.0
        if not ok:
            break
    return p, sse


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


K_GRID = (0.0, 1.5e-5, 3.0e-5, 5.0e-5, 8.0e-5, 1.3e-4, 2.0e-4)



def _calib_sse(calib, k, model):
    acc = 0.0
    for u_true, y_obs in calib:
        m = _forward(u_true, k, model)
        acc += (y_obs - m) ** 2
    return acc


def fit_k_from_calibration(calib, model):
    """标定点上「真计数 → 读数」是一维关系，k 用黄金分割在 [0, 0.9/max(u)] 里搜。"""
    if not calib:
        return None
    hi = 50.0 / max(u for u, _ in calib)
    return _golden(lambda x: _calib_sse(calib, x, model), 0.0, hi)


def fit_rates(ts, ys, model, k):
    """k 已知时把读数反解回真计数，再按双指数拟合两个时间常数。"""
    uy = _invert(ys, k, model)
    if uy is None:
        return None
    span = max(ts) - min(ts)
    t1_lo, t1_hi = span / 500.0, span / 5.0
    best = None
    for i in range(41):
        tau1 = t1_lo * (t1_hi / t1_lo) ** (i / 40.0)
        for j in range(41):
            tau2 = tau1 * 1.2 * (200.0 ** (j / 40.0))
            if not (tau1 < tau2):
                continue
            sse, coef = _sse(ts, uy, tau1, tau2)
            if sse is not None and (best is None or sse < best[0]):
                best = (sse, tau1, tau2, coef)
    if best is None:
        return None
    _, tau1, tau2, coef = best
    t1_lo2 = max(t1_lo, span / 500.0)
    for _ in range(8):
        tau1 = _golden(lambda x: _sse(ts, uy, x, tau2)[0],
                       max(t1_lo2, tau1 / 1.6), min(tau2 / 1.05, tau1 * 1.6))
        tau2 = _golden(lambda x: _sse(ts, uy, tau1, x)[0],
                       max(tau1 * 1.05, tau2 / 1.6), min(tau1 * 200.0, tau2 * 1.6))
    sse, coef = _sse(ts, uy, tau1, tau2)
    if coef is None:
        return None
    if not (tau1 < tau2):
        tau1, tau2, coef[0], coef[1] = tau2, tau1, coef[1], coef[0]
    return {"tau1": tau1, "tau2": tau2, "a1": coef[0], "a2": coef[1],
            "base": coef[2], "pileup": k, "sse": sse}

def fit_biexp(ts, ys, model="nonparalyzable"):
    t_min = min(ts)
    t_max = max(ts)
    span = max(t_max - t_min, 1e-9)
    t1_lo, t1_hi = span / 500.0, span / 5.0
    # 第一步：在（τ1、τ2、k）三维网格上粗定位，系数用反解后的线性最小二乘
    cands = []
    for k in K_GRID:
        yy = _invert(ys, k, model)
        if yy is None:
            continue
        for i in range(17):
            tau1 = t1_lo * (t1_hi / t1_lo) ** (i / 16.0)
            for j in range(17):
                tau2 = tau1 * 1.2 * (200.0 ** (j / 16.0))
                if not (tau1 < tau2):
                    continue
                _, coef = _sse(ts, yy, tau1, tau2)
                if coef is None:
                    continue
                cand = [tau1, tau2, coef[0], coef[1], coef[2], k]
                val = _sse_params(ts, ys, cand, model)
                if val is not None:
                    cands.append((val, cand))
    if not cands:
        return None
    # 第二步：取最好的几个起点分别做实测域 LM，选最优（k 的盆地浅，单起点会卡在 k=0）
    cands.sort(key=lambda t: t[0])
    seeds = []
    for val, cand in cands:
        if all(abs(cand[5] - c[5]) > 1e-9 for c in seeds):
            seeds.append(cand)
        if not (len(seeds) < 4):
            break
    best_p, best_sse = None, None
    for seed in seeds:
        out = _refine(ts, ys, seed, model)
        if out is None:
            continue
        p_try, sse_try = out
        if best_sse is None or sse_try < best_sse:
            best_p, best_sse = p_try, sse_try
    if best_p is None:
        best_p, best_sse = cands[0][1], cands[0][0]
    p, sse = best_p, best_sse
    tau1, tau2, a1, a2, c, k = p
    if not (tau1 < tau2):
        tau1, tau2, a1, a2 = tau2, tau1, a2, a1
    return {"tau1": tau1, "tau2": tau2, "a1": a1, "a2": a2,
            "base": c, "pileup": k, "sse": sse}



def analyse(scan):
    idx = keep_unsaturated(scan["ys"], scan["sat"], scan["margin"])
    if len(idx) < scan["min_used"]:
        _fail("剔除饱和点后只剩 %d 个有效点，少于要求的 %d" % (len(idx), scan["min_used"]))
    ts = [scan["times"][i] for i in idx]
    ys = [scan["ys"][i] for i in idx]
    fit = None
    if scan["calibration"]:
        k_cal = fit_k_from_calibration(scan["calibration"], scan["model"])
        if k_cal is not None:
            fit = fit_rates(ts, ys, scan["model"], k_cal)
    if fit is None:
        fit = fit_biexp(ts, ys, scan["model"])
    if fit is None:
        _fail("拟合失败，数据不足以定出参数")
    dof = len(ts) - 6
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
        "samples_total": res["n_total"],
        "samples_used": res["n_used"],
        "dof": res["dof"],
        "fast_tau_ns": round(res["tau1"], 3),
        "slow_tau_ns": round(res["tau2"], 3),
        "fast_amplitude": round(res["a1"], 2),
        "slow_amplitude": round(res["a2"], 2),
        "baseline": round(res["base"], 2),
        "pileup_scale": round(res["pileup"] * 1e6, 2),
        "amplitude_ratio": round(ratio, 3),
        "residual_rms": round(res["rmse"], 3),
        "method_note": {"estimator": "joint-two-exponential", "unit": scan["unit"]},
    }


def load_cache(path):
    """读复核缓存；读不出来或结构不对就当空缓存。"""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return []
    if not isinstance(raw, dict) or raw.get("version") != 1:
        return []
    entries = raw.get("entries")
    if not isinstance(entries, list):
        return []
    out = []
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("key"), str) and isinstance(e.get("report"), dict):
            out.append({"key": e["key"], "report": e["report"]})
    return out


def record_key(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def save_cache(path, entries):
    """缓存与报告都先写临时文件再原子替换。"""
    atomic_write(path, {"version": 1, "entries": entries})


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
    ap.add_argument("--cache", default=None)
    args = ap.parse_args(argv)

    try:
        key = record_key(args.data)
    except OSError:
        _fail("读不到记录 %s" % args.data)

    entries = load_cache(args.cache) if args.cache else []
    report = None
    for e in entries:
        if e["key"] == key:
            report = e["report"]
            break

    if report is None:
        scan = load_data(args.data)
        res = analyse(scan)
        report = build_report(scan, res)

    if args.cache:
        hit = None
        for i, e in enumerate(entries):
            if e["key"] == key:
                hit = i
                break
        entry = {"key": key, "report": report}
        if hit is None:
            entries = entries + [entry]
            if len(entries) > CACHE_LIMIT:
                entries = entries[-CACHE_LIMIT:]
        else:
            entries = entries[:]
            entries[hit] = entry
        try:
            save_cache(args.cache, entries)
        except OSError:
            _fail("缓存写不进去：%s" % args.cache)

    try:
        atomic_write(args.out, report)
    except OSError:
        _fail("报告写不进去：%s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
