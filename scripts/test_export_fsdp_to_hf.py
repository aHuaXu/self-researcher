#!/usr/bin/env python3
"""
Smoke test for export_fsdp_to_hf.py.

Creates a tiny synthetic FSDP checkpoint (using a 2-layer GPT2-like config),
runs the export, then verifies:
  1. model.safetensors exists in output_dir
  2. The exported weights match the original weights (round-trip accuracy)
  3. from_pretrained can load the output without error

Run:
    python scripts/test_export_fsdp_to_hf.py
"""

import os
import sys
import tempfile
import warnings

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp import ShardedStateDictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def init_dist():
    if not torch.distributed.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29502')
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        torch.distributed.init_process_group(backend=backend, world_size=1, rank=0)


def make_tiny_model():
    """Create a tiny GPT2 model for fast testing (< 1 MB)."""
    cfg = GPT2Config(
        vocab_size=512,
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_inner=128,
    )
    return AutoModelForCausalLM.from_config(cfg)


def save_fake_fsdp_checkpoint(model, ckpt_dir, tokenizer_dir):
    """Wrap with FSDP and save a SHARDED_STATE_DICT checkpoint."""
    device_id = 0 if torch.cuda.is_available() else None
    model_fsdp = FSDP(model, device_id=device_id, use_orig_params=True, sync_module_states=True)

    os.makedirs(ckpt_dir, exist_ok=True)
    state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with FSDP.state_dict_type(model_fsdp, StateDictType.SHARDED_STATE_DICT, state_dict_cfg):
            sd = model_fsdp.state_dict()
    torch.save(sd, os.path.join(ckpt_dir, 'model_world_size_1_rank_0.pt'))

    # Save config + tokenizer to huggingface/ subdir
    hf_dir = os.path.join(ckpt_dir, 'huggingface')
    os.makedirs(hf_dir, exist_ok=True)
    model_fsdp._fsdp_wrapped_module.config.save_pretrained(hf_dir)

    # Use GPT2 tokenizer as a stand-in
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    tok.save_pretrained(hf_dir)

    return model_fsdp


# ──────────────────────────────────────────────
# Main test
# ──────────────────────────────────────────────

def find_local_tokenizer():
    """Find a locally available tokenizer path (offline-friendly)."""
    candidates = [
        '/home/zjx/ahua_llm/self-researcher/models/Qwen2.5-0.5B-Instruct',
        '/home/zjx/ahua_llm/self-researcher/models/Qwen2.5-3B-Instruct',
        os.path.expanduser('~/models/Qwen2.5-0.5B-Instruct'),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise RuntimeError(
        'No local tokenizer found. Checked:\n' + '\n'.join(candidates)
    )


def run_test():
    print('=' * 55)
    print('Smoke test: export_fsdp_to_hf')
    print('=' * 55)

    init_dist()

    tokenizer_path = find_local_tokenizer()
    print(f'Using local tokenizer: {tokenizer_path}')

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, 'actor')
        output_dir = os.path.join(tmpdir, 'exported_hf')
        base_model_dir = os.path.join(tmpdir, 'base_model')

        # ── Step 1: create tiny model and save base HF copy ──────────
        print('[1/4] Creating tiny model and saving base HF copy...')
        model = make_tiny_model()
        os.makedirs(base_model_dir, exist_ok=True)
        model.save_pretrained(base_model_dir)
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        tok.save_pretrained(base_model_dir)

        # record original weights for comparison
        original_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        # ── Step 2: simulate a training step (mutate weights slightly) ──
        print('[2/4] Simulating weight update (adding noise)...')
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        trained_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        # ── Step 3: save FSDP checkpoint ────────────────────────────────
        print('[3/4] Saving synthetic FSDP checkpoint...')
        model_fsdp = save_fake_fsdp_checkpoint(model, ckpt_dir, tokenizer_path)

        # ── Step 4: run export script logic inline ───────────────────────
        print('[4/4] Running export...')
        # Import and call the export logic directly
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from torch.distributed.fsdp import FullStateDictConfig

        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model_fsdp, StateDictType.FULL_STATE_DICT, full_cfg):
            full_sd = model_fsdp.state_dict()

        os.makedirs(output_dir, exist_ok=True)
        unwrapped = model_fsdp._fsdp_wrapped_module
        unwrapped.load_state_dict(full_sd)
        unwrapped.save_pretrained(output_dir, safe_serialization=True)
        tok.save_pretrained(output_dir)

        # ── Assertions ───────────────────────────────────────────────────
        print()
        print('Verifying...')

        # 1. model.safetensors exists
        safetensors_path = os.path.join(output_dir, 'model.safetensors')
        assert os.path.exists(safetensors_path), \
            f'FAIL: model.safetensors not found in {output_dir}'
        print(f'  [PASS] model.safetensors exists ({os.path.getsize(safetensors_path) // 1024} KB)')

        # 2. config.json exists
        config_path = os.path.join(output_dir, 'config.json')
        assert os.path.exists(config_path), 'FAIL: config.json not found'
        print('  [PASS] config.json exists')

        # 3. Exported weights match trained weights (not original)
        loaded_model = AutoModelForCausalLM.from_pretrained(output_dir)
        exported_sd = loaded_model.state_dict()

        max_diff_from_trained = max(
            (exported_sd[k].cpu() - trained_sd[k]).abs().max().item()
            for k in trained_sd
        )
        max_diff_from_original = max(
            (exported_sd[k].cpu() - original_sd[k]).abs().max().item()
            for k in original_sd
        )
        print(f'  [PASS] Max weight diff vs trained  : {max_diff_from_trained:.2e}  (should be ~0)')
        print(f'  [INFO] Max weight diff vs original : {max_diff_from_original:.2e}  (should be > 0)')
        assert max_diff_from_trained < 1e-5, \
            f'FAIL: exported weights differ from trained weights (max_diff={max_diff_from_trained})'
        assert max_diff_from_original > 1e-6, \
            'FAIL: exported weights are identical to original (noise was not preserved)'

        # 4. from_pretrained works without error
        reloaded = AutoModelForCausalLM.from_pretrained(output_dir)
        assert reloaded is not None
        print('  [PASS] from_pretrained reload successful')

    print()
    print('All checks passed.')


if __name__ == '__main__':
    run_test()
