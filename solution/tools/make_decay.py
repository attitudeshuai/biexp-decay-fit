#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成闪烁体衰减曲线（双指数 + 基线 + 读出堆积 + 探测器饱和 + 测量噪声）。

数据口径：业务模型 + 确定性生成规则（固定伪随机序列，不用真随机）——
  u(t) = A1·exp(-t/τ1) + A2·exp(-t/τ2) + c；真实计数先过一遍读出堆积
  y = u/(1 + k·u)，前段若超过探测器上限再被限幅在饱和度上
  （真实采集就是这样丢峰的），拟合时必须把限幅点剔除并反解堆积。
"""
import json
import math
import sys


class Lcg(object):
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_u(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return (self.state % 1000003) / 1000003.0


def make_curve(a1=12000.0, tau1=3.5, a2=2100.0, tau2=18.0, base=12.0,
               dt_ns=0.5, duration_ns=120.0, sigma=3.0, saturation=6000.0,
               pileup=5.0e-5, seed=20261026):
    rng = Lcg(seed)
    n = int(round(duration_ns / dt_ns)) + 1
    samples = []
    for i in range(n):
        t = round(i * dt_ns, 2)
        y = a1 * math.exp(-t / tau1) + a2 * math.exp(-t / tau2) + base
        y += (rng.next_u() + rng.next_u() - 1.0) * math.sqrt(6.0) * sigma
        # 非瘫痪型堆积：高计数处读数被压缩，y_obs = y_true / (1 + k·y_true)
        y = y / (1.0 + pileup * y)
        if y > saturation:
            y = saturation
        samples.append({"t": t, "y": round(y, 3)})
    return {
        "unit": {"time": "ns", "counts": "counts"},
        "instrument": {"saturation_counts": saturation, "time_step_ns": dt_ns},
        "samples": samples,
        "fit": {"saturation_margin": 0.5, "min_used": 40},
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "decay.json"
    data = make_curve()
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print("wrote %s  samples=%d" % (out, len(data["samples"])))
