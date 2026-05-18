# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward
from transformers.utils import logging

from verl.utils.ulysses import gather_heads_scatter_seq, gather_seq_scatter_heads, get_ulysses_sequence_parallel_world_size

logger = logging.get_logger(__name__)


def _flash_attention_2_is_supported() -> bool:
    if "flash_attention_2" not in ALL_ATTENTION_FUNCTIONS:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def qwen3_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Qwen3 attention forward with Ulysses sequence parallelism.

    Qwen3 computes RoPE outside the attention layer. With remove-padding + Ulysses,
    hidden_states are local sequence shards while position_embeddings remain global.
    Gather sequence before RoPE so q/k and cos/sin share the same sequence length,
    then scatter the attention output back to local sequence shards.
    """
    if past_key_values is None and "past_key_value" in kwargs:
        past_key_values = kwargs.pop("past_key_value")

    bsz, q_len, _ = hidden_states.shape
    hidden_shape = (bsz, q_len, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    if ulysses_sp_size > 1:
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)

    full_q_len = query_states.size(2)
    attn_impl = self.config._attn_implementation

    # transformers>=4.57 builds the causal mask before this attention hook.
    # Under Ulysses, q/k are gathered to full sequence length here, while the
    # pre-built attention_mask can remain local-shard length, causing SDPA shape
    # mismatch (e.g. 20208 vs 5052). In this case we drop the stale mask and
    # rely on causal attention path.
    if ulysses_sp_size > 1 and attention_mask is not None and attention_mask.ndim == 4:
        mask_q_len = attention_mask.shape[-2]
        mask_k_len = attention_mask.shape[-1]
        full_k_len = key_states.size(2)
        if mask_q_len != full_q_len or mask_k_len != full_k_len:
            logger.warning_once(
                "Qwen3 Ulysses: detected stale local attention_mask shape "
                f"{tuple(attention_mask.shape)} vs gathered q/k ({full_q_len}, {full_k_len}); "
                "falling back to causal path by ignoring attention_mask."
            )
            attention_mask = None
            if attn_impl == "sdpa":
                if _flash_attention_2_is_supported():
                    attn_impl = "flash_attention_2"
                    logger.warning_once(
                        "Qwen3 Ulysses: switching attention interface from sdpa to flash_attention_2 "
                        "to avoid sdpa O(seq^2) memory blowup on gathered sequence."
                    )
                else:
                    logger.warning_once(
                        "Qwen3 Ulysses: keep sdpa because flash_attention_2 is unsupported "
                        "on current GPU (requires Ampere+)."
                    )

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward
    if attn_impl != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, full_q_len, -1, self.head_dim).contiguous()
    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
