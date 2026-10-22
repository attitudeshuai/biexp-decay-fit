#!/bin/bash
# 评分脚本：reward 只写 /logs/verifier/reward.txt；开头先覆盖为 0，崩溃/超时/信号都归零。
# 退出前再补一次 fail-closed（DONE 未置位说明没走到正常收尾），最后把 reward 置为只读。
set -u

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
DONE=0
trap '[ "$DONE" = 1 ] || echo 0 > /logs/verifier/reward.txt' EXIT
trap 'exit 1' INT TERM HUP

cd /app

SCORE=$(timeout 240 python3 /tests/test_outputs.py) || SCORE=0
case "$SCORE" in
  ''|*[!0-9.]*) SCORE=0 ;;
esac
echo "$SCORE" > /logs/verifier/reward.txt
DONE=1
chmod 0444 /logs/verifier/reward.txt
