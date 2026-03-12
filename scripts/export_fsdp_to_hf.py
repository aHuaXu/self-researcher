#!/usr/bin/env python3
"""
Export a single-GPU FSDP checkpoint (from train_grpo.sh Stage 1) to HuggingFace format.
This enables using Stage 1 output as the base model for Stage 2 multi-agent training.

Usage:
    python scripts/export_fsdp_to_hf.py \
        --ckpt_dir  ./ckpts/deepresearcher/grpo_exp/global_step_200/actor \
        --base_model ./models/Qwen2.5-0.5B-Instruct \
        --output_dir ./ckpts/deepresearcher/grpo_exp/exported_hf

Then in train_multi_agent.sh set:
    actor_rollout_ref.model.path=<output_dir>

Notes:
    - Only supports single-GPU (world_size=1) checkpoints produced by train_grpo.sh.
    - Does NOT handle LoRA checkpoints from train_multi_agent.sh.
    - Requires the same Python env as training (deepresearcher conda env).
"""

import argparse
import glob
import os
import shutil

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp import FullStateDictConfig, ShardedStateDictConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Export FSDP checkpoint to HuggingFace format')
    parser.add_argument('--ckpt_dir', required=True,
                        help='Actor checkpoint directory containing model_world_size_N_rank_K.pt')
    parser.add_argument('--base_model', required=True,
                        help='Base HF model path for structure initialization (e.g. ./models/Qwen2.5-0.5B-Instruct)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for the exported HF model')
    parser.add_argument('--dtype', default='auto', choices=['auto', 'float16', 'bfloat16', 'float32'],
                        help='Model dtype (default: auto, uses bfloat16 if CUDA available else float32)')
    return parser.parse_args()


def find_checkpoint(ckpt_dir):
    """Find and validate the model checkpoint file."""
    # Look for single-GPU checkpoint first
    single_gpu_path = os.path.join(ckpt_dir, 'model_world_size_1_rank_0.pt')
    if os.path.exists(single_gpu_path):
        return single_gpu_path, 1

    # Check for multi-GPU checkpoint (not supported but give a helpful error)
    shards = glob.glob(os.path.join(ckpt_dir, 'model_world_size_*_rank_*.pt'))
    if not shards:
        raise FileNotFoundError(
            f'No model checkpoint found in {ckpt_dir}.\n'
            f'Expected: model_world_size_1_rank_0.pt'
        )

    # Parse world_size from filename
    import re
    world_sizes = set()
    for s in shards:
        m = re.search(r'model_world_size_(\d+)_rank_\d+\.pt', s)
        if m:
            world_sizes.add(int(m.group(1)))

    if len(world_sizes) != 1:
        raise ValueError(f'Ambiguous shards in {ckpt_dir}: {shards}')

    world_size = world_sizes.pop()
    if world_size > 1:
        raise ValueError(
            f'Found world_size={world_size} checkpoint. '
            f'This script only supports single-GPU (world_size=1) checkpoints.\n'
            f'For multi-GPU checkpoints, run the script with torchrun --nproc_per_node={world_size}.'
        )

    return single_gpu_path, 1


def init_single_process_distributed():
    """Initialize torch.distributed for a single process (required by FSDP)."""
    if not torch.distributed.is_initialized():
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29501')
        torch.distributed.init_process_group(
            backend=backend,
            world_size=1,
            rank=0,
        )


def main():
    args = parse_args()

    print('=' * 60)
    print('FSDP → HuggingFace Checkpoint Export')
    print('=' * 60)
    print(f'  ckpt_dir  : {args.ckpt_dir}')
    print(f'  base_model: {args.base_model}')
    print(f'  output_dir: {args.output_dir}')
    print()

    # Validate checkpoint
    ckpt_path, world_size = find_checkpoint(args.ckpt_dir)
    print(f'[1/5] Found checkpoint: {ckpt_path}')

    # Initialize distributed
    init_single_process_distributed()
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device_id = 0
        print('[2/5] Using CUDA device 0')
    else:
        device_id = None
        print('[2/5] No CUDA available, using CPU (slower)')

    # Determine dtype
    if args.dtype == 'auto':
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        torch_dtype = getattr(torch, args.dtype)
    print(f'      dtype: {torch_dtype}')

    # Load model structure from base model (rank 0 loads real weights)
    print(f'[3/5] Loading model structure from {args.base_model}...')
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # Wrap with minimal FSDP (world_size=1, effectively a no-op for sharding)
    model_fsdp = FSDP(
        model,
        device_id=device_id,
        use_orig_params=True,
        sync_module_states=True,
    )

    # Load FSDP sharded state dict
    print(f'[4/5] Loading FSDP checkpoint weights...')
    model_state_dict = torch.load(ckpt_path, map_location='cpu')
    state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model_fsdp, StateDictType.SHARDED_STATE_DICT, state_dict_cfg):
        model_fsdp.load_state_dict(model_state_dict)
    del model_state_dict

    # Collect full state dict on CPU
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model_fsdp, StateDictType.FULL_STATE_DICT, full_cfg):
        full_state_dict = model_fsdp.state_dict()

    # Save HF model
    print(f'[5/5] Saving HuggingFace model to {args.output_dir}...')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load full weights into the unwrapped model and save
    unwrapped = model_fsdp._fsdp_wrapped_module
    unwrapped.load_state_dict(full_state_dict)
    unwrapped.save_pretrained(args.output_dir, safe_serialization=True)
    del full_state_dict

    # Copy tokenizer — prefer from checkpoint's huggingface/ subdir, fallback to base_model
    hf_subdir = os.path.join(args.ckpt_dir, 'huggingface')
    tokenizer_src = hf_subdir if os.path.exists(hf_subdir) else args.base_model
    print(f'      Copying tokenizer from {tokenizer_src}...')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    print()
    print('=' * 60)
    print(f'Export complete: {args.output_dir}')
    print()
    print('Next step — set in train_multi_agent.sh:')
    print(f'  actor_rollout_ref.model.path={os.path.abspath(args.output_dir)}')
    print('=' * 60)


if __name__ == '__main__':
    main()
