#!/bin/bash
# 评分脚本：pytest 跑 /tests 下的测试，加权分由 tests/conftest.py 写进 weighted.txt。
# reward 只写 /logs/verifier/reward.txt；开头先覆盖为 0，崩溃/超时/信号都归零。
# 写分数之前先确认结构化报告 pytest.xml 非空、weighted.txt 非空、用例条数够，
# 三者缺一即按 0 记（INCOMPLETE 不许拿满分）。
set -u

LOG_DIR=/logs/verifier
mkdir -p "$LOG_DIR"
echo 0 > "$LOG_DIR/reward.txt"
DONE=0
trap '[ "$DONE" = 1 ] || echo 0 > "$LOG_DIR/reward.txt"' EXIT
trap 'exit 1' INT TERM HUP

cd /app

# 判分工具不预装进镜像：pytest 随包放在 /tests/wheels，验证阶段离线安装
pip install --no-index --find-links /tests/wheels --quiet pytest > "$LOG_DIR/pip.log" 2>&1 || true

rm -f "$LOG_DIR/weighted.txt"
timeout 240 python3 -m pytest /tests -q -p no:cacheprovider -o addopts="" \
    --junitxml="$LOG_DIR/pytest.xml" > "$LOG_DIR/pytest.log" 2>&1 || true

SCORE=0
if [ -s "$LOG_DIR/pytest.xml" ] && [ -s "$LOG_DIR/weighted.txt" ]; then
  RAW=$(tr -d '\r\n' < "$LOG_DIR/weighted.txt")
  case "$RAW" in
    ''|*[!0-9.]*) SCORE=0 ;;
    *) SCORE=$RAW ;;
  esac
  COUNT=$(grep -c '<testcase' "$LOG_DIR/pytest.xml" 2>/dev/null) || COUNT=0
  if [ "$COUNT" -lt 11 ]; then
    SCORE=0
  fi
fi
echo "$SCORE" > "$LOG_DIR/reward.txt"
DONE=1
chmod 0444 "$LOG_DIR/reward.txt" 2>/dev/null || true
