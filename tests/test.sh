#!/bin/bash
# 评分脚本：pytest 跑 /tests，加权分交给 /tests/score.py 按 /tests/scoring.json 折算。
# 起手清掉旧报告与旧分数；写 reward 之前校验结构化报告非空、用例条数够。
set -u

LOG_DIR=/logs/verifier
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR/pytest.xml" "$LOG_DIR/reward.txt" "$LOG_DIR/scoring-report.json"
echo 0 > "$LOG_DIR/reward.txt"
DONE=0
trap '[ "$DONE" = 1 ] || echo 0 > "$LOG_DIR/reward.txt"' EXIT
trap 'exit 1' INT TERM HUP

cd /

/usr/local/bin/python3 -m pip install --no-index --find-links /tests/wheels --quiet pytest > "$LOG_DIR/pip.log" 2>&1 || true

timeout 240 env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP /usr/local/bin/python3 -P -m pytest /tests -q -p no:cacheprovider -o addopts="" \
    --junitxml="$LOG_DIR/pytest.xml" > "$LOG_DIR/pytest.log" 2>&1 || true

if [ -s "$LOG_DIR/pytest.xml" ]; then
  timeout 120 /usr/local/bin/python3 /tests/score.py \
      --config /tests/scoring.json \
      --junit "$LOG_DIR/pytest.xml" \
      --report-out "$LOG_DIR/scoring-report.json" \
      --reward-out "$LOG_DIR/reward.txt" > "$LOG_DIR/weighted.log" 2>&1 || true
fi
# 完成报告校验：没有结构化评分报告、或报告状态是 INCOMPLETE，都在日志里留痕（reward 保持 0）
if [ ! -s "$LOG_DIR/scoring-report.json" ]; then
  echo "[INCOMPLETE] 没有拿到结构化评分报告，reward 记 0" >> "$LOG_DIR/weighted.log"
fi
if [ -s "$LOG_DIR/scoring-report.json" ] && grep -q '"status": "INCOMPLETE"' "$LOG_DIR/scoring-report.json" 2>/dev/null; then
  echo "[INCOMPLETE] 有用例没跑到，reward 记 0" >> "$LOG_DIR/weighted.log"
fi

DONE=1
chmod 0444 "$LOG_DIR/reward.txt" 2>/dev/null || true
