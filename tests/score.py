#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""加权评分器：读 scoring.json 与 pytest 的 junit 报告，折算 reward 并落结构化报告。

用法：
  python3 score.py --config scoring.json --junit junit.xml \
      --report-out scoring-report.json --reward-out reward.txt

口径：
  · gate_tests 里任何一条没通过 → 整份记 0（status=FAILED_GATE）；
  · 有测试没出现在 junit 里（跑了但没收集到 / 被 skip）→ 记 0（status=INCOMPLETE）；
  · 其余情况 reward = 通过用例的权重之和（权重来自 scoring.json，和为 1）。
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("schema_version") != 1:
        raise SystemExit("scoring.json schema_version must be 1")
    gates = cfg.get("gate_tests") or []
    tests = cfg.get("tests") or {}
    if not gates or not tests:
        raise SystemExit("scoring.json 需要非空的 gate_tests 与 tests")
    return gates, tests


def load_outcomes(path):
    tree = ET.parse(path)
    root = tree.getroot()
    outcomes = {}
    for case in root.iter("testcase"):
        name = (case.get("name") or "").split("[")[0]
        cls = case.get("classname") or ""
        module = cls.split(".")[-1] if cls else ""
        cid = "%s.py::%s" % (module, name) if module else name
        failed = any(child.tag in ("failure", "error") for child in case)
        skipped = any(child.tag == "skipped" for child in case)
        status = "fail" if failed else ("skip" if skipped else "pass")
        # 同一用例的参数化变体：任一失败即算失败
        if outcomes.get(cid) != "fail":
            outcomes[cid] = status if outcomes.get(cid) != "fail" else "fail"
        if status == "fail":
            outcomes[cid] = "fail"
    return outcomes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--junit", required=True)
    ap.add_argument("--report-out", required=True)
    ap.add_argument("--reward-out", required=True)
    args = ap.parse_args(argv)

    gates, tests = load_config(args.config)
    try:
        outcomes = load_outcomes(args.junit)
    except (OSError, ET.ParseError) as exc:
        report = {"schema_version": 1, "status": "INCOMPLETE", "reward": 0.0,
                  "reason": "junit 读不出来：%s" % exc, "cases": {}}
        payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        with open(args.report_out, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
        with open(args.reward_out, "w", encoding="utf-8", newline="\n") as f:
            f.write("0.000000\n")
        return 0

    cases = {}
    missing = []
    gate_failed = []
    total = 0.0
    for cid in gates:
        status = outcomes.get(cid)
        cases[cid] = {"kind": "gate", "status": status or "missing"}
        if status is None:
            missing.append(cid)
        elif status != "pass":
            gate_failed.append(cid)
    for cid, meta in tests.items():
        status = outcomes.get(cid)
        cases[cid] = {"kind": "scored", "weight": meta["weight"],
                      "rationale": meta["rationale"], "status": status or "missing"}
        if status is None:
            missing.append(cid)
        elif status == "pass":
            total += float(meta["weight"])

    if missing:
        status, reward = "INCOMPLETE", 0.0
    else:
        status, reward = "COMPLETED", round(0.0 if gate_failed else total, 6)

    report = {"schema_version": 1, "status": status, "reward": reward,
              "gate_passed": not gate_failed, "missing": missing,
              "gate_failed": gate_failed, "cases": cases}
    with open(args.report_out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with open(args.reward_out, "w", encoding="utf-8", newline="\n") as f:
        f.write("%.6f\n" % reward)
    return 0


if __name__ == "__main__":
    sys.exit(main())
