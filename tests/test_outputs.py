#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""判分输出：10 条行为测试，按重要程度加权，权重和 = 1。

每条测试自己造一段双指数衰减曲线，按两个时间常数的二维搜索独立复算
（自己的网格与细化路径），再驱动被测程序的命令行入口比对报告。
期望值不读被测代码，也不读参考解的输出。
"""
import json
import math
import os
import subprocess
import sys
import tempfile

APP = os.environ.get("CT_APP_DIR", "/app")
ENTRY = os.path.join(APP, "src", "decay.py")

WEIGHTS = {
    "test_tau_pair_matches_independent_fit": 0.15,
    "test_near_degenerate_components_resolved": 0.14,
    "test_saturated_points_removed": 0.11,
    "test_second_curve_with_other_rates": 0.11,
    "test_rmse_near_noise_level": 0.07,
    "test_baseline_and_ratio_consistent": 0.07,
    "test_invalid_data_rejected_without_output": 0.11,
    "test_report_schema_and_rounding": 0.08,
    "test_rerun_produces_identical_bytes": 0.08,
    "test_failed_run_keeps_previous_report": 0.08,
}


class _Lcg(object):
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_u(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return (self.state % 1000003) / 1000003.0


def make_curve(a1=8200.0, tau1=3.5, a2=1500.0, tau2=18.0, base=12.0,
               dt_ns=0.5, duration_ns=120.0, sigma=3.0, saturation=6000.0, seed=101):
    rng = _Lcg(seed)
    n = int(round(duration_ns / dt_ns)) + 1
    samples = []
    for i in range(n):
        t = round(i * dt_ns, 2)
        y = a1 * math.exp(-t / tau1) + a2 * math.exp(-t / tau2) + base
        y += (rng.next_u() + rng.next_u() - 1.0) * math.sqrt(6.0) * sigma
        if y > saturation:
            y = saturation
        samples.append({"t": t, "y": round(y, 3)})
    return {"unit": {"time": "ns", "counts": "counts"},
            "instrument": {"saturation_counts": saturation, "time_step_ns": dt_ns},
            "samples": samples,
            "fit": {"saturation_margin": 0.5, "min_used": 40}}


# ---------------------------------------------------------------- 独立复算
def _solve3(a, b):
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


def _fit_lin(ts, ys, t1, t2):
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for t, y in zip(ts, ys):
        x = [math.exp(-t / t1), math.exp(-t / t2), 1.0]
        for i in range(3):
            xty[i] += x[i] * y
            for j in range(3):
                xtx[i][j] += x[i] * x[j]
    return _solve3(xtx, xty)


def _sse(ts, ys, t1, t2):
    c = _fit_lin(ts, ys, t1, t2)
    if c is None:
        return None, None
    acc = 0.0
    for t, y in zip(ts, ys):
        acc += (y - (c[0] * math.exp(-t / t1) + c[1] * math.exp(-t / t2) + c[2])) ** 2
    return acc, c


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


def expect(data):
    sat = float(data["instrument"]["saturation_counts"])
    margin = float(data["fit"]["saturation_margin"])
    ts, ys = [], []
    for s in data["samples"]:
        if float(s["y"]) >= sat - margin:
            continue
        ts.append(float(s["t"]))
        ys.append(float(s["y"]))
    span = max(ts) - min(ts)
    lo, hi = span / 500.0, span / 5.0
    best = None
    for i in range(31):
        t1 = lo * (hi / lo) ** (i / 30.0)
        for j in range(31):
            t2 = t1 * 1.2 * (200.0 ** (j / 30.0))
            sse, coef = _sse(ts, ys, t1, t2)
            if sse is not None and (best is None or sse < best[0]):
                best = (sse, t1, t2)
    t1, t2 = best[1], best[2]
    for _ in range(6):
        t1 = _golden(lambda x: _sse(ts, ys, x, t2)[0], max(lo, t1 / 1.5), min(t2 / 1.05, t1 * 1.5))
        t2 = _golden(lambda x: _sse(ts, ys, t1, x)[0], max(t1 * 1.05, t2 / 1.5), min(t1 * 200.0, t2 * 1.5))
    sse, coef = _sse(ts, ys, t1, t2)
    dof = len(ts) - 4
    return {"tau1": t1, "tau2": t2, "a1": coef[0], "a2": coef[1], "base": coef[2],
            "rmse": math.sqrt(sse / dof), "n_used": len(ts), "dof": dof, "sse": sse}


# ---------------------------------------------------------------- 驱动被测程序
def _run(path, out_path, timeout=240):
    cmd = [sys.executable, ENTRY, "--data", path, "--out", out_path]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def run_case(data):
    tmp = tempfile.mkdtemp(prefix="dec-case-")
    dpath = os.path.join(tmp, "decay.json")
    opath = os.path.join(tmp, "report.json")
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    proc = _run(dpath, opath)
    report = None
    if os.path.exists(opath):
        try:
            with open(opath, encoding="utf-8") as f:
                report = json.load(f)
        except ValueError:
            report = None
    return report, {"data": dpath, "out": opath, "proc": proc, "dir": tmp}


def _complete(report):
    need = {"n_total", "n_used", "dof", "tau1_ns", "tau2_ns", "a1_counts",
            "a2_counts", "baseline_counts", "amplitude_ratio", "rmse_counts"}
    if not isinstance(report, dict) or not need.issubset(set(report)):
        return False
    for key in ("tau1_ns", "tau2_ns", "a1_counts", "a2_counts", "baseline_counts", "rmse_counts"):
        try:
            if not math.isfinite(float(report[key])):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _close(got, want, rel, floor=0.0):
    try:
        v = float(got)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    return abs(v - want) <= max(rel * abs(want), floor)


# ---------------------------------------------------------------- 测试
def test_tau_pair_matches_independent_fit():
    """题面原文：「按双指数模型拟合两个时间常数、两个幅度与基线，约定 τ1 < τ2」。

    出厂曲线（τ1≈3.5 ns、τ2≈18 ns）：两个时间常数与独立复算相差不超过 1.5%，幅度不超过 2%。
    """
    data = make_curve(seed=101)
    report, _ = run_case(data)
    if not _complete(report):
        return False
    exp = expect(data)
    return (_close(report["tau1_ns"], exp["tau1"], 0.015)
            and _close(report["tau2_ns"], exp["tau2"], 0.015)
            and _close(report["a1_counts"], exp["a1"], 0.02)
            and _close(report["a2_counts"], exp["a2"], 0.02))


def test_near_degenerate_components_resolved():
    """题面原文：「两个时间常数拉不开时也要分开，不能退化成一个指数」。

    τ1=4 ns、τ2=11 ns 的曲线（比值约 2.7，比出厂曲线近得多）：两个时间常数相差不超过 4%，
    幅度相差不超过 8%。单指数拟合在这里会整体跑偏。
    """
    data = make_curve(tau1=4.0, tau2=11.0, a1=5000.0, a2=2200.0, sigma=4.0, seed=211)
    report, _ = run_case(data)
    if not _complete(report):
        return False
    exp = expect(data)
    return (_close(report["tau1_ns"], exp["tau1"], 0.04)
            and _close(report["tau2_ns"], exp["tau2"], 0.04)
            and _close(report["a1_counts"], exp["a1"], 0.08)
            and _close(report["a2_counts"], exp["a2"], 0.08))


def test_saturated_points_removed():
    """题面原文：「读数落在探测器上限附近的点是被限幅的，必须整点剔除」。

    出厂曲线前段有 4 个点顶在上限：n_used 必须只算未饱和点，参数不受影响。
    """
    data = make_curve(seed=307)
    report, _ = run_case(data)
    if not _complete(report):
        return False
    exp = expect(data)
    return (report.get("n_used") == exp["n_used"]
            and report.get("n_total") == len(data["samples"])
            and _close(report["tau1_ns"], exp["tau1"], 0.015))


def test_second_curve_with_other_rates():
    """题面原文：「复检会喂别的样品曲线，时间常数与幅度都不同」。

    换 τ1=1.2 ns、τ2=45 ns、幅度 12000/900 的曲线后仍要吻合。
    """
    data = make_curve(a1=12000.0, tau1=1.2, a2=900.0, tau2=45.0, base=8.0,
                      saturation=9000.0, sigma=5.0, duration_ns=200.0, seed=409)
    report, _ = run_case(data)
    if not _complete(report):
        return False
    exp = expect(data)
    return (_close(report["tau1_ns"], exp["tau1"], 0.02)
            and _close(report["tau2_ns"], exp["tau2"], 0.02))


def test_rmse_near_noise_level():
    """题面原文：「报告给出拟合残差，用来判断这次拟合是否到噪声量级」。

    噪声 σ=3 counts：rmse 落在 2 到 5 之间，且与独立复算相差不超过 20%。
    """
    data = make_curve(seed=503)
    report, _ = run_case(data)
    if not _complete(report):
        return False
    exp = expect(data)
    return 2.0 <= float(report["rmse_counts"]) <= 5.0 and _close(report["rmse_counts"], exp["rmse"], 0.20)


def test_baseline_and_ratio_consistent():
    """题面原文：「给出基线计数与两个幅度的比值，比值要等于 a1/a2」。

    基线在 12 counts 附近（±3），amplitude_ratio 与 a1/a2 一致（0.5%）。
    """
    data = make_curve(seed=601)
    report, _ = run_case(data)
    if not _complete(report):
        return False
    try:
        a1 = float(report["a1_counts"]); a2 = float(report["a2_counts"])
        base = float(report["baseline_counts"]); ratio = float(report["amplitude_ratio"])
    except (KeyError, TypeError, ValueError):
        return False
    if not (8.0 <= base <= 16.0):
        return False
    return abs(ratio - a1 / a2) <= 0.005 * abs(a1 / a2)


def test_invalid_data_rejected_without_output():
    """题面原文：「记录不合法时以退出码 2 结束，不产出报告」。

    四种坏记录：采样点不足 100、时间不单调、饱和上限非正、有效点不足。
    """
    good = make_curve(seed=701)
    few = json.loads(json.dumps(good))
    few["samples"] = few["samples"][:60]
    nonmono = json.loads(json.dumps(good))
    nonmono["samples"][10]["t"] = nonmono["samples"][9]["t"]
    badsat = json.loads(json.dumps(good))
    badsat["instrument"]["saturation_counts"] = 0
    short = json.loads(json.dumps(good))
    short["fit"]["min_used"] = 300
    for data in (few, nonmono, badsat, short):
        tmp = tempfile.mkdtemp(prefix="dec-bad-")
        dpath = os.path.join(tmp, "decay.json")
        opath = os.path.join(tmp, "report.json")
        with open(dpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        proc = _run(dpath, opath)
        if proc.returncode != 2 or os.path.exists(opath) or not (proc.stderr or "").strip():
            return False
        if [n for n in os.listdir(tmp) if n.endswith(".tmp")]:
            return False
    return True


def test_report_schema_and_rounding():
    """题面原文：「报告结构固定，时间常数三位小数、幅度与基线两位、比值与残差三位」。

    校验顶层字段、位数与有限性，文件里不出现 NaN 或 Infinity。
    """
    data = make_curve(seed=809)
    report, ctx = run_case(data)
    if not _complete(report):
        return False
    for key in ("tau1_ns", "tau2_ns", "amplitude_ratio", "rmse_counts"):
        v = float(report[key])
        if abs(v - round(v, 3)) > 1e-12:
            return False
    for key in ("a1_counts", "a2_counts", "baseline_counts"):
        v = float(report[key])
        if abs(v - round(v, 2)) > 1e-12:
            return False
    if report.get("dof") != report.get("n_used") - 4:
        return False
    if float(report["tau1_ns"]) >= float(report["tau2_ns"]):
        return False
    raw = open(ctx["out"], encoding="utf-8").read()
    return "NaN" not in raw and "Infinity" not in raw


def test_rerun_produces_identical_bytes():
    """题面原文：「同一份记录重复运行，报告逐字节一致」。

    连跑两次，两个报告文件字节完全相同（报告里不能带运行时间戳）。
    """
    data = make_curve(seed=907)
    first, ctx1 = run_case(data)
    if not _complete(first):
        return False
    body1 = open(ctx1["out"], "rb").read()
    second, ctx2 = run_case(data)
    if second is None:
        return False
    return body1 == open(ctx2["out"], "rb").read()


def test_failed_run_keeps_previous_report():
    """题面原文：「运行失败时不能覆盖上一次已经写好的报告」。

    先跑一次合格记录留下报告，再用时间不单调的坏记录跑：退出码 2，旧报告字节不变。
    """
    good = make_curve(seed=1009)
    tmp = tempfile.mkdtemp(prefix="dec-keep-")
    dpath = os.path.join(tmp, "decay.json")
    opath = os.path.join(tmp, "report.json")
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(good, f, ensure_ascii=False)
    first = _run(dpath, opath)
    if first.returncode != 0 or not os.path.exists(opath):
        return False
    before = open(opath, "rb").read()
    broken = json.loads(json.dumps(good))
    broken["samples"][5]["t"] = broken["samples"][4]["t"]
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(broken, f, ensure_ascii=False)
    second = _run(dpath, opath)
    if second.returncode != 2:
        return False
    return open(opath, "rb").read() == before


def main():
    total = 0.0
    for name, weight in WEIGHTS.items():
        fn = globals().get(name)
        try:
            passed = bool(fn and fn())
        except Exception as e:  # noqa: BLE001 —— 任何异常都按不通过计
            print("%s: 异常 %r" % (name, e), file=sys.stderr)
            passed = False
        total += weight * passed
        print("%s: %s (权重 %.2f)" % (name, "PASS" if passed else "FAIL", weight), file=sys.stderr)
    print("%.6f" % total)


if __name__ == "__main__":
    main()
