#!/bin/bash
# 评分脚本：pytest 跑 /tests 下的测试，加权分由 tests/conftest.py 写进 weighted.txt。
# reward 只写 /logs/verifier/reward.txt；开头先覆盖为 0，崩溃/超时/信号都归零。
# 写分数之前先确认结构化报告 pytest.xml 非空、weighted.txt 非空、用例条数够，
# 三者缺一即按 0 记（INCOMPLETE 不许拿满分）。
set -u

# 评分脚本：先把评分环境隔离（不让被测目录里同名模块劫持 import），再跑测试。
#   · cd / 且 python3 -P：不让工作目录进 sys.path，挡住 /app/pytest.py 这类同名劫持
#   · env -u PYTHON*：挡住 PYTHONPATH / sitecustomize 注入
#   · 解释器走绝对路径：挡住 PATH 上的同名 python3

LOG_DIR=/logs/verifier
mkdir -p "$LOG_DIR"
echo 0 > "$LOG_DIR/reward.txt"
DONE=0
trap '[ "$DONE" = 1 ] || echo 0 > "$LOG_DIR/reward.txt"' EXIT
trap 'exit 1' INT TERM HUP

cd /

# 判分工具不预装进镜像：pytest 随包放在 /tests/wheels，验证阶段离线安装
/usr/local/bin/python3 -m pip install --no-index --find-links /tests/wheels --quiet pytest > "$LOG_DIR/pip.log" 2>&1 || true

rm -f "$LOG_DIR/weighted.txt" "$LOG_DIR/pytest.xml"
timeout 240 env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP /usr/local/bin/python3 -P -m pytest /tests -q -p no:cacheprovider -o addopts="" \
    --junitxml="$LOG_DIR/pytest.xml" > "$LOG_DIR/pytest.log" 2>&1 || true

SCORE=0
if [ -s "$LOG_DIR/pytest.xml" ] && [ -s "$LOG_DIR/weighted.txt" ]; then
  RAW=$(tr -d '\r\n' < "$LOG_DIR/weighted.txt")
  case "$RAW" in
    ''|*[!0-9.]*) SCORE=0 ;;
    *) SCORE=$RAW ;;
  esac
  COUNT=$(grep -o 'tests="[0-9]*"' "$LOG_DIR/pytest.xml" 2>/dev/null | head -n 1 | tr -cd '0-9')
  if [ -z "$COUNT" ]; then
    COUNT=$(grep -o '<testcase' "$LOG_DIR/pytest.xml" 2>/dev/null | wc -l | tr -d ' \r\n')
  fi
  if [ -z "$COUNT" ]; then
    COUNT=0
  fi
  if [ "$COUNT" -lt 14 ]; then
    SCORE=0
  fi
fi
echo "$SCORE" > "$LOG_DIR/reward.txt"
DONE=1
chmod 0444 "$LOG_DIR/reward.txt" 2>/dev/null || true
