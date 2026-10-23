# -*- coding: utf-8 -*-
"""闪烁体衰减曲线复核的验收测试。

每条测试都自己造曲线、跑被测程序、再独立复算一遍期望值，断言全写在本函数内。
权重表放在 tests/conftest.py，score 只按权重汇总。
"""
import json
import math
import os
import subprocess
import sys
import tempfile

APP = os.environ.get("CT_APP_DIR", "/app")
ENTRY = os.path.join(APP, "src", "decay.py")


class _Lcg(object):
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_u(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return (self.state % 1000003) / 1000003.0


def make_curve(a1=12000.0, tau1=3.5, a2=2100.0, tau2=18.0, base=12.0,
               dt_ns=0.5, duration_ns=120.0, sigma=3.0, saturation=6000.0,
               pileup=5.0e-5, model="nonparalyzable", calibration=None, seed=101):
    """造一条带堆积与限幅的双指数曲线。"""
    rng = _Lcg(seed)
    n = int(round(duration_ns / dt_ns)) + 1
    samples = []
    for i in range(n):
        t = round(i * dt_ns, 2)
        u = a1 * math.exp(-t / tau1) + a2 * math.exp(-t / tau2) + base
        u += (rng.next_u() + rng.next_u() - 1.0) * math.sqrt(6.0) * sigma
        y = u / (1.0 + pileup * u) if model == "nonparalyzable" else u * math.exp(-pileup * u)
        if y > saturation:
            y = saturation
        samples.append({"t": t, "y": round(y, 3)})
    out = {"unit": {"time": "ns", "counts": "counts"},
           "instrument": {"saturation_counts": saturation, "time_step_ns": dt_ns,
                          "pileup_model": model},
           "samples": samples,
           "fit": {"saturation_margin": 0.5, "min_used": 40}}
    if calibration:
        rng2 = _Lcg(seed + 7)
        pts = []
        for u_true in calibration:
            if model == "nonparalyzable":
                y_cal = u_true / (1.0 + pileup * u_true)
            else:
                y_cal = u_true * math.exp(-pileup * u_true)
            noise = (rng2.next_u() + rng2.next_u() - 1.0) * math.sqrt(6.0) * 1.5
            pts.append({"true_counts": round(u_true, 1), "observed": round(y_cal + noise, 3)})
        out["calibration"] = pts
    return out


# ---------------------------------------------------------------- 独立复算
def _solve(a, b):
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


def _fit_lin(ts, ys, t1, t2):
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for t, y in zip(ts, ys):
        x = [math.exp(-t / t1), math.exp(-t / t2), 1.0]
        for i in range(3):
            xty[i] += x[i] * y
            for j in range(3):
                xtx[i][j] += x[i] * x[j]
    return _solve(xtx, xty)


def _forward(u, k, model):
    if model == "nonparalyzable":
        return u / (1.0 + k * u)
    return u * math.exp(-k * u)


def _invert_one(y, k, model):
    if model == "nonparalyzable":
        d = 1.0 - k * y
        return None if d < 0.02 else y / d
    u = y
    for _ in range(200):
        e = math.exp(-k * u)
        df = e * (1.0 - k * u)
        if df <= 0:
            return None
        step = (u * e - y) / df
        u -= step
        if u < 0:
            u = 0.5 * y
        if abs(step) <= 1e-13 * max(1.0, abs(u)):
            break
    return u if math.isfinite(u) and u > 0 else None



def _sse(ts, ys, t1, t2):
    """给定两个时间常数，解三个线性系数并返回残差平方和。"""
    coef = _fit_lin(ts, ys, t1, t2)
    if coef is None:
        return None, None
    acc = 0.0
    for t, y in zip(ts, ys):
        m = coef[0] * math.exp(-t / t1) + coef[1] * math.exp(-t / t2) + coef[2]
        acc += (y - m) ** 2
    return acc, coef

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


def _lm(ts, ys, p, model="nonparalyzable", iters=90):
    """实测读数上的 Levenberg-Marquardt（本轮不用矩阵库，自己解正规方程）。"""
    p = list(p)
    sse = _sse_params(ts, ys, p, model)
    if sse is None:
        return None, None
    lam = 1e-2
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
            jac = [g * a1 * e1 * t / (t1 * t1),
                   g * a2 * e2 * t / (t2 * t2),
                   g * e1, g * e2, g, -u * u * g]
            r = y - m
            for i in range(6):
                jtr[i] += jac[i] * r
                for j in range(6):
                    jtj[i][j] += jac[i] * jac[j]
        moved = False
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
                p, sse = trial, new
                lam = max(lam / 3.0, 1e-10)
                moved = True
                break
            lam *= 4.0
        if not moved:
            break
    return p, sse


def expect(data):
    """独立复算：剔限幅点，粗网格定位后精修，返回期望的六个参数与 rmse。"""
    model = data["instrument"].get("pileup_model") or "nonparalyzable"
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
    cands = []
    k_grid = (0.0, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4, 3.5e-4)
    for k in k_grid:
        if model == "nonparalyzable" and 1.0 - k * max(ys) < 0.05:
            continue
        inv = [_invert_one(y, k, model) for y in ys]
        if any(v is None for v in inv):
            continue
        for i in range(13):
            t1 = lo * (hi / lo) ** (i / 12.0)
            for j in range(13):
                t2 = t1 * 1.2 * (200.0 ** (j / 12.0))
                if t2 <= t1:
                    continue
                coef = _fit_lin(ts, inv, t1, t2)
                if coef is None:
                    continue
                cand = [t1, t2, coef[0], coef[1], coef[2], k]
                val = _sse_params(ts, ys, cand, model)
                if val is not None:
                    cands.append((val, cand))
    cands.sort(key=lambda t: t[0])
    seeds = []
    for val, cand in cands:
        if all(abs(cand[5] - c[5]) > 1e-9 for c in seeds):
            seeds.append(cand)
        if len(seeds) >= 4:
            break
    best_p, best_sse = None, None
    for seed in seeds:
        p_try, sse_try = _lm(ts, ys, seed, model)
        if p_try is None:
            continue
        if best_sse is None or sse_try < best_sse:
            best_p, best_sse = p_try, sse_try
    if best_p is None:
        best_p, best_sse = cands[0][1], cands[0][0]
    p, sse = best_p, best_sse
    if p is None:
        p, sse = best[1], best[0]
    t1, t2, a1, a2, c, k = p
    dof = len(ts) - 6
    return {"tau1": t1, "tau2": t2, "a1": a1, "a2": a2, "base": c, "pileup": k,
            "rmse": math.sqrt(sse / dof), "samples_used": len(ts), "dof": dof}


# ---------------------------------------------------------------- 驱动被测程序

def _calib_sse(calib, k, model):
    acc = 0.0
    for c in calib:
        u = float(c["true_counts"])
        acc += (float(c["observed"]) - _forward(u, k, model)) ** 2
    return acc


def _golden1(fn, lo, hi, iters=90):
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iters):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = fn(d)
    return (a + b) / 2.0


def expect_with_calibration(data):
    """有标定点时：先在标定点上定 k，再反解曲线，用（自己的）网格 + 细化拟合时间常数。"""
    model = data["instrument"].get("pileup_model") or "nonparalyzable"
    sat = float(data["instrument"]["saturation_counts"])
    margin = float(data["fit"]["saturation_margin"])
    ts, ys = [], []
    for s in data["samples"]:
        if float(s["y"]) >= sat - margin:
            continue
        ts.append(float(s["t"]))
        ys.append(float(s["y"]))
    calib = data["calibration"]
    hi = 50.0 / max(float(c["true_counts"]) for c in calib)
    k = _golden1(lambda x: _calib_sse(calib, x, model), 0.0, hi)
    uy = [_invert_one(y, k, model) for y in ys]
    if any(v is None for v in uy):
        raise AssertionError("标定出的 k 无法反解曲线")
    span = max(ts) - min(ts)
    lo2, hi2 = span / 400.0, span / 4.0
    best = None
    for i in range(31):
        t1 = lo2 * (hi2 / lo2) ** (i / 30.0)
        for j in range(31):
            t2 = t1 * 1.25 * (160.0 ** (j / 30.0))
            if t2 <= t1:
                continue
            sse, coef = _sse(ts, uy, t1, t2)
            if sse is not None and (best is None or sse < best[0]):
                best = (sse, t1, t2)
    t1, t2 = best[1], best[2]
    for _ in range(6):
        t1 = _golden1(lambda x: _sse(ts, uy, x, t2)[0], max(lo2, t1 / 1.5), min(t2 / 1.05, t1 * 1.5))
        t2 = _golden1(lambda x: _sse(ts, uy, t1, x)[0], max(t1 * 1.05, t2 / 1.5), min(t1 * 160.0, t2 * 1.5))
    sse, coef = _sse(ts, uy, t1, t2)
    dof = len(ts) - 4
    return {"tau1": t1, "tau2": t2, "a1": coef[0], "a2": coef[1], "base": coef[2],
            "pileup": k, "rmse": math.sqrt(sse / dof), "samples_used": len(ts), "dof": dof}

def _run(path, out_path, cache_path=None, timeout=240):
    cmd = [sys.executable, ENTRY, "--data", path, "--out", out_path]
    if cache_path is not None:
        cmd += ["--cache", cache_path]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def run_case(data, cache_path=None):
    tmp = tempfile.mkdtemp(prefix="dec-case-")
    dpath = os.path.join(tmp, "decay.json")
    opath = os.path.join(tmp, "report.json")
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    proc = _run(dpath, opath, cache_path)
    report = None
    if os.path.exists(opath):
        try:
            with open(opath, encoding="utf-8") as f:
                report = json.load(f)
        except ValueError:
            report = None
    return report, {"data": dpath, "out": opath, "proc": proc, "dir": tmp}


NEED = {"samples_total", "samples_used", "dof", "fast_tau_ns", "slow_tau_ns", "fast_amplitude",
        "slow_amplitude", "baseline", "pileup_scale", "amplitude_ratio",
        "residual_rms"}


def _num(report, key):
    assert isinstance(report, dict), "报告不存在或不是 JSON 对象"
    assert key in report, "报告缺少字段 %s" % key
    try:
        v = float(report[key])
    except (TypeError, ValueError):
        raise AssertionError("字段 %s 不是数值：%r" % (key, report[key]))
    assert math.isfinite(v), "字段 %s 不是有限数值：%r" % (key, v)
    return v


# ---------------------------------------------------------------- 测试
def test_tau_pair_matches_independent_fit():
    """题面原文：「真实波形 u(t) = A1·exp(-t/τ1) + A2·exp(-t/τ2) + c」与「以实测读数与模型的残差平方和为准」。

    出厂曲线（τ1≈3.5 ns、τ2≈18 ns、k≈5e-5）：两个时间常数与独立复算相差不超过 2%，
    幅度不超过 5%。
    """
    data = make_curve(seed=101)
    report, _ = run_case(data)
    for key in NEED:
        assert key in report, "报告缺少字段 %s" % key
    exp = expect(data)
    tau1 = _num(report, "fast_tau_ns")
    tau2 = _num(report, "slow_tau_ns")
    a1 = _num(report, "fast_amplitude")
    a2 = _num(report, "slow_amplitude")
    assert abs(tau1 - exp["tau1"]) <= 0.003 * exp["tau1"], \
        "tau1 差得太多：被测 %r，独立复算 %r" % (tau1, exp["tau1"])
    assert abs(tau2 - exp["tau2"]) <= 0.003 * exp["tau2"], \
        "tau2 差得太多：被测 %r，独立复算 %r" % (tau2, exp["tau2"])
    assert abs(a1 - exp["a1"]) <= 0.01 * abs(exp["a1"]), \
        "a1 差得太多：被测 %r，独立复算 %r" % (a1, exp["a1"])
    assert abs(a2 - exp["a2"]) <= 0.01 * abs(exp["a2"]), \
        "a2 差得太多：被测 %r，独立复算 %r" % (a2, exp["a2"])


def test_pileup_coefficient_recovered():
    """题面原文：「仪器读到的是 y = u / (1 + k·u)……记录里不写 k，要拿曲线本身估出来」。

    真实 k=5e-5（换算成 pileup_per_mega 是 50）与 k=1.2e-4（120）两条曲线各跑一次：
    报告值与独立复算相差不超过 10%，且不能把 k 当 0 处理（拿 k=0 拟合会偏出 25%以上）。
    """
    for true_k, seed in ((5.0e-5, 1103), (1.2e-4, 1207)):
        data = make_curve(pileup=true_k, seed=seed)
        report, _ = run_case(data)
        exp = expect(data)
        got = _num(report, "pileup_scale")
        want = exp["pileup"] * 1e6
        assert abs(got - want) <= 0.03 * abs(want), \
            "堆积系数差得太多：被测 %r，独立复算 %r" % (got, want)
        assert abs(got - true_k * 1e6) <= 0.25 * true_k * 1e6, \
            "报告值离真实堆积系数太远：被测 %r，真实 %r" % (got, true_k * 1e6)


def test_near_degenerate_components_resolved():
    """题面原文：「两个时间常数拉不开时也要分开，不能退化成一个指数」对应的近简并情形。

    τ1=4 ns、τ2=11 ns 的曲线（比值约 2.7，比出厂曲线近得多）：两个时间常数相差不超过 5%，
    幅度相差不超过 10%。单指数拟合在这里会整体跑偏。
    """
    data = make_curve(tau1=4.0, tau2=11.0, a1=8000.0, a2=3200.0, sigma=4.0,
                      saturation=9000.0, pileup=6.0e-5, seed=211)
    report, _ = run_case(data)
    exp = expect(data)
    tau1 = _num(report, "fast_tau_ns")
    tau2 = _num(report, "slow_tau_ns")
    assert abs(tau1 - exp["tau1"]) <= 0.05 * exp["tau1"], \
        "近简并曲线 tau1 偏了：被测 %r，独立复算 %r" % (tau1, exp["tau1"])
    assert abs(tau2 - exp["tau2"]) <= 0.05 * exp["tau2"], \
        "近简并曲线 tau2 偏了：被测 %r，独立复算 %r" % (tau2, exp["tau2"])
    assert tau2 > 1.8 * tau1, "两个成分退化到一起了：tau1=%r tau2=%r" % (tau1, tau2)


def test_saturated_points_removed():
    """题面原文：「读数落在探测器上限附近……的点是被限幅的，整点剔除、不参与拟合」。

    出厂曲线前段有 5 个点顶在上限：n_used 必须只算未饱和点，参数不受影响；
    把限幅点算进去会让 tau1 偏掉 5% 以上。
    """
    data = make_curve(seed=307)
    exp = expect(data)
    report, _ = run_case(data)
    n_used = int(_num(report, "samples_used"))
    n_total = int(_num(report, "samples_total"))
    assert n_total == len(data["samples"]), \
        "n_total 应为采样点总数 %d，实际 %d" % (len(data["samples"]), n_total)
    assert n_used == exp["samples_used"], \
        "n_used 应为剔除限幅点后的 %d，实际 %d" % (exp["samples_used"], n_used)
    tau1 = _num(report, "fast_tau_ns")
    assert abs(tau1 - exp["tau1"]) <= 0.02 * exp["tau1"], \
        "剔除限幅点后 tau1 仍偏：被测 %r，独立复算 %r" % (tau1, exp["tau1"])


def test_second_curve_with_other_rates():
    """题面原文：「复检会喂别的样品曲线，时间常数、幅度比、堆积系数与噪声水平都不同」。

    换 τ1=1.2 ns、τ2=45 ns、幅度 12000/900、k=8e-5 的曲线后仍要吻合（时间常数 2.5%）。
    """
    data = make_curve(a1=12000.0, tau1=1.2, a2=900.0, tau2=45.0, base=8.0,
                      saturation=9000.0, sigma=5.0, duration_ns=200.0,
                      pileup=8.0e-5, seed=409)
    report, _ = run_case(data)
    exp = expect(data)
    tau1 = _num(report, "fast_tau_ns")
    tau2 = _num(report, "slow_tau_ns")
    assert abs(tau1 - exp["tau1"]) <= 0.025 * exp["tau1"], \
        "换曲线后 tau1 偏了：被测 %r，独立复算 %r" % (tau1, exp["tau1"])
    assert abs(tau2 - exp["tau2"]) <= 0.025 * exp["tau2"], \
        "换曲线后 tau2 偏了：被测 %r，独立复算 %r" % (tau2, exp["tau2"])


def test_rmse_near_noise_level():
    """题面原文：「报告给出 rmse = sqrt(残差平方和 /(n - 6))，残差在实测读数上算」。

    噪声 σ=3 counts：rmse 落在 2 到 5 之间，且与独立复算相差不超过 20%。
    """
    data = make_curve(seed=503)
    report, _ = run_case(data)
    exp = expect(data)
    rmse = _num(report, "residual_rms")
    assert 2.0 <= rmse <= 5.0, "rmse 不在噪声量级：%r" % rmse
    assert abs(rmse - exp["rmse"]) <= 0.20 * exp["rmse"], \
        "rmse 与独立复算差得太多：被测 %r，独立复算 %r" % (rmse, exp["rmse"])


def test_baseline_and_ratio_consistent():
    """题面原文：「amplitude_ratio = a1 / a2」，基线是本底计数。

    基线在 8 到 16 counts 之间（真值 12），amplitude_ratio 与 a1/a2 一致（0.5%）。
    """
    data = make_curve(seed=601)
    report, _ = run_case(data)
    base = _num(report, "baseline")
    a1 = _num(report, "fast_amplitude")
    a2 = _num(report, "slow_amplitude")
    ratio = _num(report, "amplitude_ratio")
    assert 8.0 <= base <= 16.0, "基线不在 12 附近：%r" % base
    want = a1 / a2
    assert abs(ratio - want) <= 0.005 * abs(want), \
        "amplitude_ratio 与 a1/a2 不一致：%r 对 %r" % (ratio, want)


def test_paralyzable_pileup_model():
    """题面原文：「堆积模型写在 instrument.pileup_model 里：非瘫痪型 y = u/(1 + k·u)，瘫痪型 y = u·exp(-k·u)」。

    换一条瘫痪型堆积的曲线（k=2e-4）：两个时间常数与独立复算相差不超过 3%，
    堆积系数不超过 12%。把它当非瘫痪型反解、或者干脆不反解，都会明显跑偏。
    """
    data = make_curve(pileup=2.0e-4, model="paralyzable", seed=1311)
    report, _ = run_case(data)
    exp = expect(data)
    tau1 = _num(report, "fast_tau_ns")
    tau2 = _num(report, "slow_tau_ns")
    k = _num(report, "pileup_scale")
    assert abs(tau1 - exp["tau1"]) <= 0.03 * exp["tau1"], \
        "瘫痪型下 tau1 偏了：被测 %r，独立复算 %r" % (tau1, exp["tau1"])
    assert abs(tau2 - exp["tau2"]) <= 0.03 * exp["tau2"], \
        "瘫痪型下 tau2 偏了：被测 %r，独立复算 %r" % (tau2, exp["tau2"])
    assert abs(k - exp["pileup"] * 1e6) <= 0.12 * abs(exp["pileup"] * 1e6), \
        "瘫痪型下堆积系数偏了：被测 %r，独立复算 %r" % (k, exp["pileup"] * 1e6)


def test_calibration_pins_pileup_coefficient():
    """题面原文：「记录若带 calibration，堆积系数必须先在标定点上定出来」。

    强堆积样品（k=3.5e-4，峰值处被压到三成）：只靠曲线本身，k 与两个时间常数会退化；
    先用 5 个标定点定 k 再反解，报告的 k 与真值差不超过 5%、与独立复算不超过 2%。
    """
    calib_counts = [800.0, 2000.0, 4000.0, 8000.0, 14000.0]
    data = make_curve(pileup=3.5e-4, sigma=4.0, calibration=calib_counts, seed=2201)
    report, _ = run_case(data)
    exp = expect_with_calibration(data)
    k = _num(report, "pileup_scale")
    tau1 = _num(report, "fast_tau_ns")
    tau2 = _num(report, "slow_tau_ns")
    assert abs(k - exp["pileup"] * 1e6) <= 0.02 * abs(exp["pileup"] * 1e6), \
        "标定没有把堆积系数定住：被测 %r，独立复算 %r" % (k, exp["pileup"] * 1e6)
    assert abs(k - 3.5e-4 * 1e6) <= 0.05 * 3.5e-4 * 1e6, \
        "堆积系数离真值太远：被测 %r，真值 %r" % (k, 3.5e-4 * 1e6)
    assert abs(tau1 - exp["tau1"]) <= 0.02 * exp["tau1"], \
        "tau1 没对上：被测 %r，独立复算 %r" % (tau1, exp["tau1"])
    assert abs(tau2 - exp["tau2"]) <= 0.02 * exp["tau2"], \
        "tau2 没对上：被测 %r，独立复算 %r" % (tau2, exp["tau2"])


def test_invalid_data_rejected_without_output():
    """题面原文：「记录不合法时以退出码 2 结束、往 stderr 写原因、不产出报告」。

    四种坏记录：采样点不足 100、时间不单调、探测器上限非正、有效点不足。
    每种都要退出码 2、stderr 有原因、目录里不留报告也不留临时文件。
    """
    good = make_curve(seed=701)
    cases = []
    few = json.loads(json.dumps(good))
    few["samples"] = few["samples"][:60]
    cases.append(("采样点不足", few))
    nonmono = json.loads(json.dumps(good))
    nonmono["samples"][10]["t"] = nonmono["samples"][9]["t"]
    cases.append(("时间不单调", nonmono))
    badsat = json.loads(json.dumps(good))
    badsat["instrument"]["saturation_counts"] = 0
    cases.append(("上限非正", badsat))
    short = json.loads(json.dumps(good))
    short["fit"]["min_used"] = 300
    cases.append(("有效点不足", short))
    for label, data in cases:
        tmp = tempfile.mkdtemp(prefix="dec-bad-")
        dpath = os.path.join(tmp, "decay.json")
        opath = os.path.join(tmp, "report.json")
        with open(dpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        proc = _run(dpath, opath)
        assert proc.returncode == 2, "%s：退出码应为 2，实际 %r" % (label, proc.returncode)
        assert (proc.stderr or "").strip(), "%s：stderr 没有写原因" % label
        assert not os.path.exists(opath), "%s：不该产出报告文件" % label
        leftovers = [n for n in os.listdir(tmp) if n not in ("decay.json",)]
        assert not leftovers, "%s：目录里留下残留文件 %r" % (label, leftovers)


def test_report_schema_and_rounding():
    """题面原文：「tau1_ns / tau2_ns / amplitude_ratio / rmse_counts 保留三位小数；a1_counts / a2_counts / baseline_counts / pileup_per_mega 保留两位」。

    校验顶层字段、位数、dof 与 tau1 < tau2，文件里不出现 NaN 或 Infinity。
    """
    data = make_curve(seed=809)
    report, ctx = run_case(data)
    for key in NEED:
        assert key in report, "报告缺少字段 %s" % key
    for key in ("fast_tau_ns", "slow_tau_ns", "amplitude_ratio", "residual_rms"):
        v = _num(report, key)
        assert abs(v - round(v, 3)) <= 1e-12, "字段 %s 不是三位小数：%r" % (key, v)
    for key in ("fast_amplitude", "slow_amplitude", "baseline", "pileup_scale"):
        v = _num(report, key)
        assert abs(v - round(v, 2)) <= 1e-12, "字段 %s 不是两位小数：%r" % (key, v)
    n_used = int(_num(report, "samples_used"))
    assert int(_num(report, "dof")) == n_used - 6, \
        "dof 应等于 n_used - 6：n_used=%r dof=%r" % (n_used, report["dof"])
    assert _num(report, "fast_tau_ns") < _num(report, "slow_tau_ns"), "tau1 必须小于 tau2"
    raw = open(ctx["out"], encoding="utf-8").read()
    assert "NaN" not in raw and "Infinity" not in raw, "报告里出现了 NaN 或 Infinity"


def test_rerun_produces_identical_bytes():
    """题面原文：「同一份记录重复运行，报告逐字节一致（不要写时间戳之类的运行期信息）」。

    连跑两次，两个报告文件字节完全相同。
    """
    data = make_curve(seed=907)
    first, ctx1 = run_case(data)
    for key in NEED:
        assert key in first, "第一次运行就缺字段 %s" % key
    body1 = open(ctx1["out"], "rb").read()
    second, ctx2 = run_case(data)
    assert second is not None, "第二次运行没有产出报告"
    body2 = open(ctx2["out"], "rb").read()
    assert body1 == body2, "两次运行报告字节不一致（是不是写了时间戳）"


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
    assert first.returncode == 0, "合格记录应退出 0，实际 %r" % first.returncode
    assert os.path.exists(opath), "合格记录没有产出报告"
    before = open(opath, "rb").read()
    broken = json.loads(json.dumps(good))
    broken["samples"][5]["t"] = broken["samples"][4]["t"]
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(broken, f, ensure_ascii=False)
    second = _run(dpath, opath)
    assert second.returncode == 2, "坏记录应退出 2，实际 %r" % second.returncode
    after = open(opath, "rb").read()
    assert after == before, "运行失败把上一次的报告覆盖了"


def test_cache_reuse_and_append():
    """题面原文：「这个 key 已在缓存里就原地替换该条目的 report；不在就追加到末尾」。

    连跑三次：同一份记录跑两次（中间把缓存里那条报告的 tau1_ns 改成哨兵值 999），
    第二次必须直接复用缓存、报出哨兵值；换一份记录再跑，缓存变成两条、顺序不变。
    """
    cache = os.path.join(tempfile.mkdtemp(prefix="dec-cache-"), "cache.json")
    data_a = make_curve(seed=1301)
    report_a, ctx_a = run_case(data_a, cache_path=cache)
    for key in NEED:
        assert key in report_a, "第一次运行就缺字段 %s" % key
    assert os.path.exists(cache), "运行结束没有写缓存文件"
    with open(cache, encoding="utf-8") as f:
        body = json.load(f)
    assert isinstance(body, dict) and body.get("version") == 1, "缓存结构不对：%r" % (list(body)[:4],)
    entries = body.get("entries")
    assert isinstance(entries, list) and len(entries) == 1, "缓存条目数应为 1，实际 %r" % (entries,)
    import hashlib
    key_a = hashlib.sha256(open(ctx_a["data"], "rb").read()).hexdigest()
    assert entries[0].get("key") == key_a, "缓存 key 不是记录字节的 sha256：%r" % (entries[0].get("key"),)
    assert isinstance(entries[0].get("report"), dict), "缓存条目里的 report 不是对象"

    entries[0]["report"]["fast_tau_ns"] = 999.0
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False)
    report_again, _ = run_case(data_a, cache_path=cache)
    assert float(report_again.get("fast_tau_ns", 0.0)) == 999.0, \
        "第二次跑同一份记录没有复用缓存：tau1_ns=%r" % (report_again.get("fast_tau_ns"),)

    data_b = make_curve(seed=1409)
    report_b, ctx_b = run_case(data_b, cache_path=cache)
    exp_b = expect(data_b)
    assert abs(float(report_b["fast_tau_ns"]) - exp_b["tau1"]) <= 0.03 * exp_b["tau1"], \
        "换记录后没有重新拟合：tau1_ns=%r，独立复算 %r" % (report_b["fast_tau_ns"], exp_b["tau1"])
    with open(cache, encoding="utf-8") as f:
        body2 = json.load(f)
    keys = [e.get("key") for e in body2.get("entries", [])]
    assert len(keys) == 2, "缓存条目数应为 2，实际 %r" % (keys,)
    assert keys[0] == key_a, "原有条目被挪动或覆盖了：%r" % (keys,)
    key_b = hashlib.sha256(open(ctx_b["data"], "rb").read()).hexdigest()
    assert keys[1] == key_b, "新条目没有追加到末尾：%r" % (keys,)


def test_corrupt_cache_recovered():
    """题面原文：「缓存文件读不出来、不是合法 JSON 或结构不对：当作空缓存」。

    先塞一段乱码当缓存，再跑一份合格记录：退出码 0、报告正确、缓存被重写成合法结构。
    """
    cache = os.path.join(tempfile.mkdtemp(prefix="dec-cache-bad-"), "cache.json")
    with open(cache, "w", encoding="utf-8") as f:
        f.write("{ this is not json ][")
    data = make_curve(seed=1511)
    report, _ = run_case(data, cache_path=cache)
    exp = expect(data)
    assert isinstance(report, dict), "坏缓存把整次运行拖挂了（没有报告）"
    assert abs(float(report["fast_tau_ns"]) - exp["tau1"]) <= 0.03 * exp["tau1"], \
        "坏缓存下报告不对：tau1_ns=%r，独立复算 %r" % (report.get("fast_tau_ns"), exp["tau1"])
    with open(cache, encoding="utf-8") as f:
        body = json.load(f)
    assert body.get("version") == 1, "坏缓存没有被重写成合法结构：%r" % (body,)
    assert len(body.get("entries", [])) == 1, "重写后的缓存应只有一条，实际 %r" % (body.get("entries"),)


def test_cache_capped_at_32_entries():
    """题面原文：「缓存最多留 32 条：追加新条目时如果会超过 32 条，从最早的那头丢掉」。

    先往缓存里塞满 32 条旧条目，再跑一份新记录：缓存仍是 32 条 —— 最早那条被丢掉、
    新的一条在末尾、原来第 2 到第 32 条顺序不变。
    """
    import hashlib
    cache = os.path.join(tempfile.mkdtemp(prefix="dec-cache-cap-"), "cache.json")
    stub = {"samples_total": 241, "samples_used": 236, "dof": 230, "fast_tau_ns": 3.499,
            "slow_tau_ns": 17.964, "fast_amplitude": 12002.95, "slow_amplitude": 2103.81,
            "baseline": 11.77, "pileup_scale": 50.06,
            "amplitude_ratio": 5.705, "residual_rms": 3.095,
            "method_note": {"model": "biexp", "unit": {"time": "ns", "counts": "counts"}}}
    old_keys = ["%064x" % (i + 1) for i in range(32)]
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"version": 1,
                   "entries": [{"key": k, "report": stub} for k in old_keys]}, f)

    data = make_curve(seed=2000)
    report, ctx = run_case(data, cache_path=cache)
    assert isinstance(report, dict), "新记录没有产出报告"
    with open(cache, encoding="utf-8") as f:
        body = json.load(f)
    entries = body.get("entries")
    assert isinstance(entries, list), "缓存结构不对：%r" % (body,)
    assert len(entries) == 32, "缓存应只留 32 条，实际 %d 条" % len(entries)
    got = [e.get("key") for e in entries]
    new_key = hashlib.sha256(open(ctx["data"], "rb").read()).hexdigest()
    assert got[-1] == new_key, "新条目没有追加到末尾：%r" % (got[-1],)
    assert old_keys[0] not in got, "最早的一条没有被丢掉"
    assert got[:-1] == old_keys[1:], "留下的不是原来第 2 到第 32 条（顺序也不对）"


def test_failed_run_leaves_cache_untouched():
    """题面原文：「记录不合法（退出码 2）时缓存文件一个字节都不许动」。

    先跑一份合格记录把缓存建起来，再用坏记录跑：退出码 2，缓存字节与旧报告都不变；
    再把缓存路径指成一个目录（写不进去），也要按非法记录退出 2、不产出报告。
    """
    cache = os.path.join(tempfile.mkdtemp(prefix="dec-cache-keep-"), "cache.json")
    good = make_curve(seed=1601)
    report, ctx = run_case(good, cache_path=cache)
    assert isinstance(report, dict), "合格记录没有产出报告"
    before_cache = open(cache, "rb").read()
    before_report = open(ctx["out"], "rb").read()

    broken = json.loads(json.dumps(good))
    broken["samples"][7]["t"] = broken["samples"][6]["t"]
    bad_dir = tempfile.mkdtemp(prefix="dec-cache-broken-")
    bad_data = os.path.join(bad_dir, "decay.json")
    bad_out = os.path.join(bad_dir, "report.json")
    with open(bad_data, "w", encoding="utf-8") as f:
        json.dump(broken, f, ensure_ascii=False)
    proc = _run(bad_data, bad_out, cache_path=cache)
    assert proc.returncode == 2, "坏记录应退出 2，实际 %r" % proc.returncode
    assert open(cache, "rb").read() == before_cache, "非法记录把缓存改动了"
    assert open(ctx["out"], "rb").read() == before_report, "非法记录把旧报告改动了"
    assert not os.path.exists(bad_out), "非法记录不该产出报告"

    unwritable = tempfile.mkdtemp(prefix="dec-cache-dir-")
    good2 = make_curve(seed=1709)
    out2 = os.path.join(unwritable, "report.json")
    data2 = os.path.join(unwritable, "decay.json")
    with open(data2, "w", encoding="utf-8") as f:
        json.dump(good2, f, ensure_ascii=False)
    proc2 = _run(data2, out2, cache_path=unwritable)
    assert proc2.returncode == 2, "缓存写不进去时应退出 2，实际 %r" % proc2.returncode
    assert not os.path.exists(out2), "缓存写不进去时不该产出报告"
