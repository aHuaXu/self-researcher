#!/bin/bash
# Hi-IGPO Phase 2b: interleaved Planner(LoRA, trained) <-> frozen Executor.
# base = SFT'd Qwen3-4B-Thinking-2507 for BOTH planner & executor LoRA (single base +
# dual LoRA); search_engine=drvenus + enable_think=true => search/visit native format end to end.
#   - Planner: thinking + <subtask>/<answer>; belief GT wrapper closes think.
#   - Executor (FROZEN): the single-agent trained model (Phase 0 IGPO output), search/visit
#     (mapped to web_search/browse).
# See docs/design/hi_igpo_design.md §13.6.
#
# STAGE 1: VAL_BEFORE_TRAIN=false + test_freq=-1; observe training rollout dumps
#   (planner_rollout_step_N.json: planner single <subtask>/think ok, executor multi-turn, belief IG).
# Prereqs: models/Qwen3-4B-Thinking-2507-SFT (single-agent trained ckpt), SearXNG up, 4 idle GPUs.
set -euo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# TOOL_CONTENT_MAX_CHARS sourced from .env (single source of truth).
export PET_NODE_RANK=${PET_NODE_RANK:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,3,4,6}   # pick 4 IDLE gpus (check nvidia-smi!)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export project_name="deepresearcher"
export experiment_name="qwen3_4b_hi_igpo_phase2b"
IFS=',' read -ra _D <<< "${CUDA_VISIBLE_DEVICES}"; NGPU=${#_D[@]}

BASE=/home/zjx/self_llm/self-researcher
cd ${BASE}
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# Tool-side webpage fetching needs the local proxy on this server. Set it explicitly
# so Ray workers inherit it even when the launcher is not an interactive/login shell.
export http_proxy="${http_proxy-http://127.0.0.1:7890}"
export https_proxy="${https_proxy-http://127.0.0.1:7890}"
export ftp_proxy="${ftp_proxy-http://127.0.0.1:7890}"
export no_proxy="${no_proxy-localhost,127.0.0.1,10.*,192.168.*,*.local,*.internal}"
export HTTP_PROXY="${HTTP_PROXY-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY-${https_proxy}}"
export FTP_PROXY="${FTP_PROXY-${ftp_proxy}}"
export NO_PROXY="${NO_PROXY-${no_proxy}}"
export SEARXNG_ENGINE_PRIORITY="${SEARXNG_ENGINE_PRIORITY_OVERRIDE:-google,bing,duckduckgo,brave}"

source /home/zjx/anaconda3/bin/activate deepresearcher
export LLM_BASE_URL="${LLM_BASE_URL:-https://api.minimaxi.com/v1}"
export LLM_MODEL="${LLM_MODEL:-MiniMax-M2.7}"
export LLM_API_KEY="${LLM_API_KEY:-${JUDGE_API_KEY:-}}"

ray stop --force 2>/dev/null || true
sleep 2
export RAY_memory_monitor_refresh_ms=0
ray start --head
sleep 2

BASE_MODEL=${BASE_MODEL:-${BASE}/models/Qwen3-4B-Thinking-2507-SFT}

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=${BASE}/data/deepresearch_phase1.parquet \
    data.val_files=${BASE}/data/deepresearch_phase1_val.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=3096 \
    data.max_response_length=1536 \
    max_seq_len_for_training=7168 \
    algorithm.adv_estimator=igpo \
    +algorithm.use_vectorized_gt_logprob=true \
    +algorithm.info_gain_type=log_prob_diff \
    +algorithm.info_gain_norm_mode=separate \
    +algorithm.ig_group_mode=turn_group \
    +algorithm.format_penalty=1.0 \
    +algorithm.enable_think=true \
    algorithm.gamma=1.0 \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=7168 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.max_model_len=9216 \
    actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    ++actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp16 \
    ++actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    ++actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=7168 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    +trainer.val_before_train=${VAL_BEFORE_TRAIN:-false} \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=${NGPU} \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=-1 \
    trainer.resume_mode=auto \
    trainer.remove_previous_ckpt_in_save=true \
    multi_agent.enable=true \
    multi_agent.max_planner_turns=4 \
    multi_agent.freeze_executor=true \
    multi_agent.planner_findings_max_chars=1000 \
    multi_agent.base_model=${BASE_MODEL} \
    multi_agent.lora_save_dir=${BASE}/tmp_lora_adapters_hi_igpo_phase2b \
    multi_agent.lora.rank=64 \
    multi_agent.lora.alpha=128 \
    multi_agent.lora.dropout=0.05 \
    multi_agent.agents.executor.max_turns=6 \
    agent_grpo.n=8 \
    max_turns=6 \
    search_engine=drvenus \
    trainer.total_epochs=1 \
    trainer.total_training_steps=100 2>&1 | tee ${BASE}/${project_name}_${experiment_name}.log
