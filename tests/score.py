#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 pytest 的 junitxml 结果按 WEIGHTS 折算成总分（0~1）。

用法：python3 /tests/score.py --junit /tmp/results.xml --report-out /tmp/scoring-report.json \
          --reward-out /logs/verifier/reward.txt
WEIGHTS 从同目录 test_outputs.py 里静态读出（ast，不执行被测代码）。
junit 缺失/为空/解析失败：reward 写 0，报告里记 INCOMPLETE。
"""
import argparse
import ast
import json
import os
import sys
import xml.etree.ElementTree as ET


def load_weights(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "WEIGHTS":
                    return ast.literal_eval(node.value)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--junit", required=True)
    ap.add_argument("--report-out", required=True)
    ap.add_argument("--reward-out", required=True)
    a = ap.parse_args()

    weights = load_weights(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "test_outputs.py"))
    report = {"status": "INCOMPLETE", "total": 0.0, "cases": {}}
    reward = 0.0

    xml_ok = False
    if os.path.exists(a.junit) and os.path.getsize(a.junit) > 0:
        try:
            root = ET.parse(a.junit).getroot()
            xml_ok = True
        except ET.ParseError:
            xml_ok = False

    if xml_ok and weights:
        failed = set()
        seen = set()
        for tc in root.iter("testcase"):
            name = tc.get("name", "")
            seen.add(name)
            if list(tc):
                failed.add(name)
        missing = [n for n in weights if n not in seen]
        if missing:
            # 有测试根本没跑到（收集失败/被跳过），不能按全过计
            report["status"] = "INCOMPLETE"
            report["missing"] = missing
        else:
            total = 0.0
            for name, w in weights.items():
                passed = name not in failed
                report["cases"][name] = {"weight": w, "status": "PASS" if passed else "FAIL"}
                total += w * passed
                print("%s: %s (权重 %.3f)" % (name, report["cases"][name]["status"], w),
                      file=sys.stderr)
            report["status"] = "COMPLETE"
            report["total"] = total
            reward = total

    with open(a.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    with open(a.reward_out, "w", encoding="utf-8") as f:
        f.write("%.6f" % reward)
    print("%.6f" % reward)


if __name__ == "__main__":
    main()
