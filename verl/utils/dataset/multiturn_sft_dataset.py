# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0.
"""Multi-turn SFT dataset for chat ``messages`` parquet files.

This is adapted from DR-Venus' multi-turn SFT dataset, with one important
compatibility detail for this repo's current ``fsdp_sft_trainer``: the trainer
uses ``loss_mask[:, :-1]`` against next-token labels, so this dataset returns a
pre-shifted input-position mask. Assistant tokens are still the only labels
trained; system/user/tool observations are masked out.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local


def _to_regular(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_regular(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_regular(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_regular(obj.tolist())
    return obj


class MultiTurnSFTDataset(Dataset):
    """Dataset for multi-turn conversations.

    Expected parquet schema:
      - ``messages``: list[{"role": "system"|"user"|"assistant"|"tool", "content": str}]
      - optional ``tools``: chat-template tool schema
      - optional ``enable_thinking``: passed through to Qwen-style chat templates

    Loss is applied to every assistant message, not just the final assistant
    message. Non-assistant messages are context only.
    """

    def __init__(self, parquet_files: str | list[str], tokenizer, config=None):
        config = config or {}
        self.max_length = int(config.get("max_length", 4096))
        self.truncation = config.get("truncation", "error")
        assert self.truncation in ["error", "left", "right"]

        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", "enable_thinking")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        self.parquet_files = list(parquet_files)
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files(self):
        def series_to_item(value):
            import numpy
            import pandas

            while isinstance(value, pandas.core.series.Series | numpy.ndarray) and len(value) == 1:
                value = value[0]
            return value

        dataframes = [pd.read_parquet(path) for path in self.parquet_files]
        self.dataframe = pd.concat(dataframes)
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).apply(_to_regular).tolist()
        self.tools = (
            self.dataframe[self.tools_key].apply(series_to_item).apply(_to_regular).tolist()
            if self.tools_key in self.dataframe.columns
            else None
        )
        self.enable_thinking = (
            self.dataframe[self.enable_thinking_key].tolist()
            if self.enable_thinking_key in self.dataframe.columns
            else None
        )

    def __len__(self):
        return len(self.messages)

    def _apply_template(self, messages, *, tools=None, enable_thinking=None, add_generation_prompt=False):
        kwargs = dict(self.apply_chat_template_kwargs)
        if tools is not None:
            kwargs["tools"] = tools
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def _message_tokens(
        self,
        messages: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        *,
        is_assistant: bool,
        tools=None,
        enable_thinking=None,
    ) -> tuple[list[int], list[int], list[int]]:
        if start_idx > 0:
            prev_text = self._apply_template(
                messages[:start_idx],
                tools=tools,
                enable_thinking=enable_thinking,
                add_generation_prompt=False,
            )
            if is_assistant:
                prev_gen_text = self._apply_template(
                    messages[:start_idx],
                    tools=tools,
                    enable_thinking=enable_thinking,
                    add_generation_prompt=True,
                )
        else:
            prev_text = ""
            prev_gen_text = ""

        cur_text = self._apply_template(
            messages[:end_idx],
            tools=tools,
            enable_thinking=enable_thinking,
            add_generation_prompt=False,
        )

        if is_assistant:
            gen_prompt_text = prev_gen_text[len(prev_text) :]
            gen_prompt_tokens = self.tokenizer.encode(gen_prompt_text, add_special_tokens=False)
            assistant_tokens = self.tokenizer.encode(cur_text[len(prev_gen_text) :], add_special_tokens=False)
            tokens = gen_prompt_tokens + assistant_tokens
            label_loss_mask = [0] * len(gen_prompt_tokens) + [1] * len(assistant_tokens)
        else:
            tokens = self.tokenizer.encode(cur_text[len(prev_text) :], add_special_tokens=False)
            label_loss_mask = [0] * len(tokens)

        attention_mask = [1] * len(tokens)
        return tokens, label_loss_mask, attention_mask

    def _build_sequence(self, messages, *, tools=None, enable_thinking=None):
        full_tokens = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,
            **({"enable_thinking": enable_thinking} if enable_thinking is not None else {}),
            **self.apply_chat_template_kwargs,
        )[0].tolist()

        tokens: list[int] = []
        label_loss_mask: list[int] = []
        attention_mask: list[int] = []
        i = 0
        while i < len(messages):
            role = messages[i]["role"]
            if role == "assistant":
                cur_tokens, cur_loss, cur_attn = self._message_tokens(
                    messages, i, i + 1, is_assistant=True, tools=tools, enable_thinking=enable_thinking
                )
                i += 1
            elif role == "tool":
                start = i
                i += 1
                while i < len(messages) and messages[i]["role"] == "tool":
                    i += 1
                cur_tokens, cur_loss, cur_attn = self._message_tokens(
                    messages, start, i, is_assistant=False, tools=tools, enable_thinking=enable_thinking
                )
            elif role in ["system", "user"]:
                if role == "system" and i != 0:
                    raise ValueError("system message should be the first message")
                cur_tokens, cur_loss, cur_attn = self._message_tokens(
                    messages, i, i + 1, is_assistant=False, tools=tools, enable_thinking=enable_thinking
                )
                i += 1
            else:
                raise ValueError(f"Unknown role: {role}")

            tokens.extend(cur_tokens)
            label_loss_mask.extend(cur_loss)
            attention_mask.extend(cur_attn)

        if len(tokens) != len(full_tokens) or any(a != b for a, b in zip(tokens, full_tokens, strict=False)):
            logging.warning(
                "Token mismatch in MultiTurnSFTDataset; using incremental tokenization. "
                "full_len=%s concat_len=%s",
                len(full_tokens),
                len(tokens),
            )

        # Current trainer masks next-token losses with loss_mask[:, :-1].
        # Shift label-position mask left so label token i is trained from input position i-1.
        input_loss_mask = [0] * len(label_loss_mask)
        for idx in range(1, len(label_loss_mask)):
            input_loss_mask[idx - 1] = label_loss_mask[idx]
        return tokens, input_loss_mask, attention_mask

    def __getitem__(self, item):
        messages = self.messages[item]
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None

        try:
            input_ids, loss_mask, attention_mask = self._build_sequence(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
            )
        except Exception:
            logging.exception("Failed to tokenize multi-turn SFT messages: %s", messages)
            raise

        sequence_length = len(input_ids)
        if sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            else:
                raise ValueError(f"sequence_length={sequence_length} is larger than max_length={self.max_length}")

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            loss_mask = loss_mask + [0] * pad_len

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.long)
        position_ids = torch.arange(len(input_ids), dtype=torch.long) * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
