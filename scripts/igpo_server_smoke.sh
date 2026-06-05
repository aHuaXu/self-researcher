#!/usr/bin/env bash
# =============================================================================
# Hi-IGPO 服务器一键检查脚本 (Phase 1: 环境 + 单测 + belief 数值验证)
#
# 在【你 VPN 能连到服务器的本地机器】上运行(不是服务器上)。
# 它会:rsync 本仓库 → 服务器、查 GPU、在服务器 conda env 里跑 hi_igpo 单测,
# 并(可选)做 belief 数值验证。全部输出同时打印到屏幕。
#
# 用法:
#   1) 先只查 GPU + 跑 CPU 单测(不占 GPU):
#        bash scripts/igpo_server_smoke.sh
#   2) 看完 GPU 输出、挑一张空闲卡(如 2),再做 belief 验证:
#        IGPO_GPU=2 bash scripts/igpo_server_smoke.sh
# =============================================================================
set -uo pipefail

SERVER="${IGPO_SERVER:-zjx@10.35.2.238}"
REMOTE_DIR="${IGPO_REMOTE_DIR:-/home/zjx/self_llm/self-researcher}"
CONDA_ACT="${IGPO_CONDA_ACT:-source /home/zjx/anaconda3/bin/activate deepresearcher}"
MODEL_PATH="${IGPO_MODEL_PATH:-/home/zjx/self_llm/self-researcher/models/Qwen3-4B-Instruct}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "================ Hi-IGPO server smoke (Phase 1) ================"
echo "server      : $SERVER"
echo "remote dir  : $REMOTE_DIR"
echo "local dir   : $LOCAL_DIR"
echo "model       : $MODEL_PATH"
echo "IGPO_GPU    : ${IGPO_GPU:-<unset → 只查GPU+CPU单测,跳过belief>}"
echo "==============================================================="

# ---- 1) rsync 本地 → 服务器(排除大目录/本地虚拟环境) ----
echo; echo "### [1/4] rsync 本地 → 服务器 ..."
rsync -avz --delete \
  --exclude '.git' --exclude '.venv' --exclude '.venv-test' \
  --exclude 'models' --exclude 'data' --exclude 'outputs' --exclude 'downloads' \
  --exclude 'tmp' --exclude '*.log' --exclude '__pycache__' --exclude '.pytest_cache' \
  "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/" || { echo "rsync 失败"; exit 1; }

# ---- 2) 查 GPU(只读,绝不占用) ----
echo; echo "### [2/4] GPU 状态(挑一张 memory.used≈0 / util 0% 的空闲卡):"
ssh "$SERVER" 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv'

# ---- 3) 服务器上跑 hi_igpo CPU 单测(优势 + 散射,9 passed 预期) ----
echo; echo "### [3/4] 服务器跑 hi_igpo CPU 单测 ..."
ssh "$SERVER" "$CONDA_ACT && cd $REMOTE_DIR && \
  python -m pytest tests/hi_igpo/test_igpo_advantage.py tests/hi_igpo/test_igpo_reward_scatter.py -v 2>&1 | tail -25"

# ---- 4) belief 数值验证(仅当指定了空闲 GPU) ----
echo; echo "### [4/4] belief 数值验证 ..."
if [ -z "${IGPO_GPU:-}" ]; then
  echo "跳过(未设 IGPO_GPU)。看完上面 GPU 状态,挑一张空闲卡,重跑:"
  echo "    IGPO_GPU=<卡号> bash scripts/igpo_server_smoke.sh"
else
  echo "在 GPU $IGPO_GPU 上加载 $MODEL_PATH 验证 belief(Bel∈(0,1) 且含答案上下文更高)..."
  ssh "$SERVER" "$CONDA_ACT && cd $REMOTE_DIR && \
    CUDA_VISIBLE_DEVICES=$IGPO_GPU IGPO_BELIEF_TEST_MODEL=$MODEL_PATH \
    python -m pytest tests/hi_igpo/test_belief_smoke.py -v -s 2>&1 | tail -40"
fi
echo; echo "================ 完成。把上面全部输出贴回来即可。 ================"
