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
               pileup=5.0e-5, seed=101):
    """造一条带堆积与限幅的双指数曲线。"""
    rng = _Lcg(seed)
    n = int(round(duration_ns / dt_ns)) + 1
    samples = []
    for i in range(n):
        t = round(i * dt_ns, 2)
        u = a1 * math.exp(-t / tau1) + a2 * math.exp(-t / tau2) + base
        u += (rng.next_u() + rng.next_u() - 1.0) * math.sqrt(6.0) * sigma
        y = u / (1.0 + pileup * u)
        if y > saturation:
            y = saturation
        samples.append({"t": t, "y": round(y, 3)})
    return {"unit": {"time": "ns", "counts": "counts"},
            "instrument": {"saturation_counts": saturation, "time_step_ns": dt_ns},
            "samples": samples,
            "fit": {"saturation_margin": 0.5, "min_used": 40}}


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


def _sse_params(ts, ys, p):
    t1, t2, a1, a2, c, k = p
    if t1 <= 0 or t2 <= t1 or k < 0:
        return None
    acc = 0.0
    for t, y in zip(ts, ys):
        u = a1 * math.exp(-t / t1) + a2 * math.exp(-t / t2) + c
        acc += (y - u / (1.0 + k * u)) ** 2
    return acc


def _lm(ts, ys, p, iters=90):
    """实测读数上的 Levenberg-Marquardt（本轮不用矩阵库，自己解正规方程）。"""
    p = list(p)
    sse = _sse_params(ts, ys, p)
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
            w = 1.0 / (1.0 + k * u)
            w2 = w * w
            jac = [w2 * a1 * e1 * t / (t1 * t1),
                   w2 * a2 * e2 * t / (t2 * t2),
                   w2 * e1, w2 * e2, w2, -u * u * w2]
            r = y - u * w
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
            new = _sse_params(ts, ys, trial)
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
    for k in (0.0, 1.2e-5, 3.5e-5, 6.5e-5, 1.1e-4):
        d = 1.0 - k * max(ys)
        if d < 0.05:
            continue
        for i in range(13):
            t1 = lo * (hi / lo) ** (i / 12.0)
            for j in range(13):
                t2 = t1 * 1.2 * (200.0 ** (j / 12.0))
                if t2 <= t1:
                    continue
                coef = _fit_lin(ts, [y / (1.0 - k * y) for y in ys], t1, t2)
                if coef is None:
                    continue
                cand = [t1, t2, coef[0], coef[1], coef[2], k]
                val = _sse_params(ts, ys, cand)
                if val is not None and (best is None or val < best[0]):
                    best = (val, cand)
    p, sse = _lm(ts, ys, best[1])
    if p is None:
        p, sse = best[1], best[0]
    t1, t2, a1, a2, c, k = p
    dof = len(ts) - 6
    return {"tau1": t1, "tau2": t2, "a1": a1, "a2": a2, "base": c, "pileup": k,
            "rmse": math.sqrt(sse / dof), "n_used": len(ts), "dof": dof}


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


NEED = {"n_total", "n_used", "dof", "tau1_ns", "tau2_ns", "a1_counts",
        "a2_counts", "baseline_counts", "pileup_per_mega", "amplitude_ratio",
        "rmse_counts"}


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
    tau1 = _num(report, "tau1_ns")
    tau2 = _num(report, "tau2_ns")
    a1 = _num(report, "a1_counts")
    a2 = _num(report, "a2_counts")
    assert abs(tau1 - exp["tau1"]) <= 0.02 * exp["tau1"], \
        "tau1 差得太多：被测 %r，独立复算 %r" % (tau1, exp["tau1"])
    assert abs(tau2 - exp["tau2"]) <= 0.02 * exp["tau2"], \
        "tau2 差得太多：被测 %r，独立复算 %r" % (tau2, exp["tau2"])
    assert abs(a1 - exp["a1"]) <= 0.05 * abs(exp["a1"]), \
        "a1 差得太多：被测 %r，独立复算 %r" % (a1, exp["a1"])
    assert abs(a2 - exp["a2"]) <= 0.05 * abs(exp["a2"]), \
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
        got = _num(report, "pileup_per_mega")
        want = exp["pileup"] * 1e6
        assert abs(got - want) <= 0.10 * abs(want), \
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
    tau1 = _num(report, "tau1_ns")
    tau2 = _num(report, "tau2_ns")
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
    n_used = int(_num(report, "n_used"))
    n_total = int(_num(report, "n_total"))
    assert n_total == len(data["samples"]), \
        "n_total 应为采样点总数 %d，实际 %d" % (len(data["samples"]), n_total)
    assert n_used == exp["n_used"], \
        "n_used 应为剔除限幅点后的 %d，实际 %d" % (exp["n_used"], n_used)
    tau1 = _num(report, "tau1_ns")
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
    tau1 = _num(report, "tau1_ns")
    tau2 = _num(report, "tau2_ns")
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
    rmse = _num(report, "rmse_counts")
    assert 2.0 <= rmse <= 5.0, "rmse 不在噪声量级：%r" % rmse
    assert abs(rmse - exp["rmse"]) <= 0.20 * exp["rmse"], \
        "rmse 与独立复算差得太多：被测 %r，独立复算 %r" % (rmse, exp["rmse"])


def test_baseline_and_ratio_consistent():
    """题面原文：「amplitude_ratio = a1 / a2」，基线是本底计数。

    基线在 8 到 16 counts 之间（真值 12），amplitude_ratio 与 a1/a2 一致（0.5%）。
    """
    data = make_curve(seed=601)
    report, _ = run_case(data)
    base = _num(report, "baseline_counts")
    a1 = _num(report, "a1_counts")
    a2 = _num(report, "a2_counts")
    ratio = _num(report, "amplitude_ratio")
    assert 8.0 <= base <= 16.0, "基线不在 12 附近：%r" % base
    want = a1 / a2
    assert abs(ratio - want) <= 0.005 * abs(want), \
        "amplitude_ratio 与 a1/a2 不一致：%r 对 %r" % (ratio, want)


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
    for key in ("tau1_ns", "tau2_ns", "amplitude_ratio", "rmse_counts"):
        v = _num(report, key)
        assert abs(v - round(v, 3)) <= 1e-12, "字段 %s 不是三位小数：%r" % (key, v)
    for key in ("a1_counts", "a2_counts", "baseline_counts", "pileup_per_mega"):
        v = _num(report, key)
        assert abs(v - round(v, 2)) <= 1e-12, "字段 %s 不是两位小数：%r" % (key, v)
    n_used = int(_num(report, "n_used"))
    assert int(_num(report, "dof")) == n_used - 6, \
        "dof 应等于 n_used - 6：n_used=%r dof=%r" % (n_used, report["dof"])
    assert _num(report, "tau1_ns") < _num(report, "tau2_ns"), "tau1 必须小于 tau2"
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
