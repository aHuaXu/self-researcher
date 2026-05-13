#!/usr/bin/env bash
# Every 30s: print time, PPO alive?, Ray head line, last step-ish lines from remote GRPO log.
# Usage: bash scripts/train/monitor_grpo_7b_remote.sh
# Override: REMOTE=user@host LOG=/path/to/deepresearcher_qwen2.5_7b_grpo.log

set -euo pipefail
REMOTE="${REMOTE:-zjx@10.35.2.238}"
LOG="${LOG:-/home/zjx/ahua_llm/self-researcher/deepresearcher_qwen2.5_7b_grpo.log}"

while true; do
  echo "======== $(date -Iseconds) ========"
  # Pass log path via env so the remote script can stay quoted (no accidental local $(...) expansion).
  ssh -o BatchMode=yes -o ConnectTimeout=20 "${REMOTE}" \
    env REMOTE_LOG="${LOG}" bash -s <<'EOF' || true
if pgrep -f "python3 -m verl.trainer.main_ppo" >/dev/null 2>&1; then
  echo "[PPO] running"
else
  echo "[PPO] NOT RUNNING"
fi
source /home/zjx/anaconda3/bin/activate deepresearcher 2>/dev/null || true
ray status 2>/dev/null | head -8 || echo "[Ray] ray status unavailable"
echo "--- step / rollout (last 12 matches) ---"
grep -E 'node 0 step|rollout_step|global_step|Training|epoch' "${REMOTE_LOG}" 2>/dev/null | tail -n 12 || true
echo "--- log tail ---"
tail -n 5 "${REMOTE_LOG}" 2>/dev/null
EOF
  sleep 20
done
