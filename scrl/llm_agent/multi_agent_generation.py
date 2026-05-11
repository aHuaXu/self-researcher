"""
Multi-Agent Generation Manager for three-stage LoRA rollout.

Implements Planner -> Executor -> Writer pipeline where each agent
uses a separate LoRA adapter on the same base model.
"""

import re
import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from verl import DataProto
from scrl.llm_agent.generation import LLMGenerationManager, GenerationConfig

from research_agent.prompts.planner import get_planner_prompt
from research_agent.prompts.executor import get_executor_prompt, EXECUTOR_TOOLS
from research_agent.prompts.writer import get_writer_prompt

class MultiAgentGenerationManager(LLMGenerationManager):
    """Three-stage rollout manager: Planner -> Executor -> Writer.

    Each stage uses a different LoRA adapter loaded on the same base model.
    The executor stage is multi-turn (tool-calling), while planner and writer
    are single-turn generation.
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        lora_save_dir: str = './tmp_lora_adapters',
        is_validation: bool = False,
    ):
        super().__init__(tokenizer, actor_rollout_wg, config, is_validation)
        self.lora_save_dir = lora_save_dir
        self._lora_step = 0

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def run_multi_agent_loop(
        self, gen_batch: DataProto, global_steps: int,
        agent_grpo_idx: Optional[List[int]] = None,
    ) -> dict:
        """Run the three-stage multi-agent rollout.

        Args:
            gen_batch: Input DataProto containing tokenized questions.
            global_steps: Current training step (for logging).
            agent_grpo_idx: GRPO group index for each sample in gen_batch.
                Samples with the same index are different rollouts of the
                same original question.

        Returns:
            Dict with 'planner', 'executor', 'writer' DataProto outputs
            and 'metadata' dict with decoded text artifacts.
        """
        node_rank = int(os.environ.get("PET_NODE_RANK", 0))
        questions = self._extract_questions(gen_batch)
        num_questions = len(questions)
        self._lora_step += 1
        print(
            f"[MultiAgent] node {node_rank}, {num_questions} questions, "
            f"step {global_steps}",
            flush=True,
        )

        # ------------------------------------------------------------------
        # Stage 1: Planner (single-turn generation)
        # ------------------------------------------------------------------
        print(f"[MultiAgent] Stage 1: Planner", flush=True)
        planner_batch = self._build_planner_batch(questions, gen_batch)
        plan_outputs = self._generate_with_gpu_padding(
            planner_batch, lora_adapter_name="planner"
        )
        plan_texts = self._decode_outputs(plan_outputs)
        parsed_todos = [self._parse_todos(text) for text in plan_texts]

        # Log planner outputs
        self._log_stage_outputs(
            "planner", global_steps, questions, plan_texts, parsed_todos
        )

        # ------------------------------------------------------------------
        # Stage 2: Executor (multi-turn tool-calling)
        # ------------------------------------------------------------------
        print(f"[MultiAgent] Stage 2: Executor", flush=True)
        executor_batch, todo_mapping = self._build_executor_batch(
            questions, parsed_todos, gen_batch
        )

        if len(executor_batch) > 0:
            # run_llm_loop returns (message_string_list, message_tensor)
            exec_msg_strings, exec_outputs = self.run_llm_loop(
                executor_batch,
                global_steps,
                lora_adapter_name="executor",
            )
        else:
            print(
                "[MultiAgent] WARNING: executor batch is empty — "
                "planner produced no TODOs",
                flush=True,
            )
            exec_msg_strings = []
            exec_outputs = DataProto.from_dict({
                'input_ids': torch.zeros((0, 1), dtype=torch.long),
                'attention_mask': torch.zeros((0, 1), dtype=torch.long),
                'position_ids': torch.zeros((0, 1), dtype=torch.long),
            })

        grouped_findings = self._group_findings(
            exec_msg_strings, todo_mapping, num_questions
        )
        exec_trajectories = self._build_exec_trajectories(
            exec_msg_strings, todo_mapping, num_questions
        )

        # ------------------------------------------------------------------
        # Stage 3: Writer (single-turn generation)
        # ------------------------------------------------------------------
        print(f"[MultiAgent] Stage 3: Writer", flush=True)
        writer_batch = self._build_writer_batch(
            questions, plan_texts, grouped_findings, gen_batch
        )
        writer_outputs = self._generate_with_gpu_padding(
            writer_batch, lora_adapter_name="writer"
        )
        final_reports = self._decode_outputs(writer_outputs)

        # Log writer outputs
        self._log_stage_outputs(
            "writer", global_steps, questions, final_reports
        )

        # Propagate GRPO group index to outputs so advantage computation
        # groups multiple rollouts of the same question together.
        if agent_grpo_idx is not None:
            # DataProto.check_consistency() requires non_tensor_batch arrays to have dtype=object
            plan_outputs.non_tensor_batch['agent_grpo_idx'] = np.array(agent_grpo_idx, dtype=object)
            writer_outputs.non_tensor_batch['agent_grpo_idx'] = np.array(agent_grpo_idx, dtype=object)
            # Executor has one output per TODO, not per question — build its index
            # from the todo_mapping (each todo maps back to a question index)
            if todo_mapping:
                exec_grpo_idx = [agent_grpo_idx[q_idx] for q_idx in todo_mapping]
                exec_outputs.non_tensor_batch['agent_grpo_idx'] = np.array(exec_grpo_idx, dtype=object)

        return {
            "planner": plan_outputs,
            "executor": exec_outputs,
            "writer": writer_outputs,
            "metadata": {
                "queries": questions,
                "plan_texts": plan_texts,
                "parsed_todos": parsed_todos,
                "exec_trajectories": exec_trajectories,
                "final_reports": final_reports,
                # todo_mapping[j] = question index for executor TODO j;
                # used by ray_trainer to expand per-question rewards to per-TODO
                "todo_mapping": todo_mapping,
            },
        }

    # -------------------------------------------------------------------------
    # Question extraction
    # -------------------------------------------------------------------------

    def _extract_questions(self, gen_batch: DataProto) -> List[str]:
        """Extract question strings from a DataProto batch.

        Uses the parent class parse_question which decodes input_ids and
        extracts the user message content between <|im_start|>user and
        <|im_end|>.
        """
        return self.parse_question(gen_batch.batch['input_ids'])

    # -------------------------------------------------------------------------
    # Batch builders
    # -------------------------------------------------------------------------

    def _build_planner_batch(
        self, questions: List[str], ref_batch: DataProto
    ) -> DataProto:
        """Build a DataProto batch of planner prompts."""
        messages_list = [get_planner_prompt(q) for q in questions]
        return self._tokenize_messages_to_batch(messages_list, ref_batch)

    def _build_executor_batch(
        self,
        questions: List[str],
        parsed_todos: List[List[Dict]],
        ref_batch: DataProto,
    ) -> Tuple[DataProto, List[int]]:
        """Flatten all TODO items into one executor batch.

        Returns:
            (batch, todo_mapping) where todo_mapping[i] is the index of
            the original question that executor prompt i belongs to.
        """
        messages_list = []
        todo_mapping = []

        for q_idx, (question, todos) in enumerate(
            zip(questions, parsed_todos)
        ):
            for todo in todos:
                sub_topic = todo.get("sub_topic", question)
                messages = get_executor_prompt(sub_topic)
                messages_list.append(messages)
                todo_mapping.append(q_idx)

        if not messages_list:
            # No TODOs at all: return empty batch
            empty_batch = DataProto.from_dict({
                'input_ids': torch.zeros((0, 1), dtype=torch.long),
                'attention_mask': torch.zeros((0, 1), dtype=torch.long),
                'position_ids': torch.zeros((0, 1), dtype=torch.long),
            })
            return empty_batch, todo_mapping

        return (
            self._tokenize_messages_to_batch(
                messages_list, ref_batch, tools=EXECUTOR_TOOLS
            ),
            todo_mapping,
        )

    def _build_writer_batch(
        self,
        questions: List[str],
        plan_texts: List[str],
        grouped_findings: List[str],
        ref_batch: DataProto,
    ) -> DataProto:
        """Build a DataProto batch of writer prompts.

        Each writer prompt includes the original question and all grouped
        research findings for that question.
        """
        messages_list = []
        for q_idx, question in enumerate(questions):
            # Combine plan and findings into a single findings block
            findings_block = (
                f"=== Research Plan ===\n{plan_texts[q_idx]}\n\n"
                f"=== Research Findings ===\n{grouped_findings[q_idx]}"
            )
            messages = get_writer_prompt(question, findings_block)
            messages_list.append(messages)

        return self._tokenize_messages_to_batch(messages_list, ref_batch)

    # -------------------------------------------------------------------------
    # Tokenization helper
    # -------------------------------------------------------------------------

    def _tokenize_messages_to_batch(
        self,
        messages_list: List[List[Dict]],
        ref_batch: DataProto,
        tools: Optional[List[Dict]] = None,
    ) -> DataProto:
        """Tokenize a list of chat message sequences into a padded DataProto.

        Uses tokenizer.apply_chat_template for proper chat formatting,
        then creates input_ids, attention_mask, and position_ids tensors.

        Args:
            messages_list: List of conversation message lists, each a list
                of dicts with 'role' and 'content'.
            ref_batch: Reference DataProto (unused currently but kept for
                future compatibility, e.g. copying meta_info).
            tools: Optional tool definitions to pass to apply_chat_template.
        """
        # apply_chat_template returns a list of strings when tokenize=False
        template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if tools is not None:
            template_kwargs["tools"] = tools

        formatted_strings = self.tokenizer.apply_chat_template(
            messages_list, **template_kwargs
        )

        # Tokenize all formatted strings with padding
        tokenized = self.tokenizer(
            formatted_strings,
            return_tensors="pt",
            padding=True,
        )

        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        # Left-pad: sort so that pad tokens come first
        pad_mask = input_ids != self.tokenizer.pad_token_id
        sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)
        input_ids = input_ids.gather(1, sorted_indices)
        attention_mask = attention_mask.gather(1, sorted_indices)

        position_ids = self.tensor_fn.create_position_ids(attention_mask)

        return DataProto.from_dict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        })

    # -------------------------------------------------------------------------
    # Output decoding
    # -------------------------------------------------------------------------

    def _decode_outputs(self, outputs: DataProto) -> List[str]:
        """Decode token IDs from output DataProto to strings.

        Expects the DataProto to contain a 'responses' key with the
        generated token IDs.
        """
        if 'responses' not in outputs.batch.keys():
            print(
                "[MultiAgent] WARNING: _decode_outputs found no 'responses' "
                "key in batch, returning empty list",
                flush=True,
            )
            return []

        responses = outputs.batch['responses']
        decoded = self.tokenizer.batch_decode(
            responses, skip_special_tokens=False
        )
        # Clean up padding tokens
        decoded = [
            text.replace(self.tokenizer.pad_token or "<|endoftext|>", "").strip()
            for text in decoded
        ]
        return decoded

    # -------------------------------------------------------------------------
    # TODO parsing
    # -------------------------------------------------------------------------

    def _parse_todos(self, plan_text: str) -> List[Dict[str, Any]]:
        """Parse planner output into a list of TODO dicts.

        Each dict has keys: index, priority, sub_topic.
        The planner only produces sub-topics; the executor decides
        what to search for on its own.
        """
        todos = []

        # Primary pattern: "1. [HIGH] The sub-topic description"
        pattern = r'(\d+)\.\s*\[(HIGH|MEDIUM|LOW)\]\s*(.+?)(?=\n\d+\.\s*\[|</todos>|$)'
        matches = re.findall(pattern, plan_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            sub_topic = match[2].strip()
            sub_topic = re.sub(
                r'^(?:Sub-topic|子主题|主题)[：:]\s*',
                '',
                sub_topic,
                flags=re.IGNORECASE,
            )
            sub_topic = sub_topic.rstrip('</todos>').strip()
            todos.append({
                "index": int(match[0]),
                "priority": match[1].lower(),
                "sub_topic": sub_topic,
            })

        if todos:
            return todos

        # Last-resort fallback: treat the whole text as a single TODO
        clean_text = plan_text.strip()
        if clean_text:
            print(
                f"[MultiAgent] WARNING: _parse_todos regex failed, "
                f"falling back to raw text: {clean_text[:80]!r}",
                flush=True,
            )
            todos.append({
                "index": 1,
                "priority": "high",
                "sub_topic": clean_text[:200],
            })

        return todos

    # -------------------------------------------------------------------------
    # Findings grouping
    # -------------------------------------------------------------------------

    def _group_findings(
        self,
        exec_msg_strings: List[str],
        todo_mapping: List[int],
        num_questions: int,
    ) -> List[str]:
        """Group executor output strings back to their original questions.

        Args:
            exec_msg_strings: List of full executor message strings (one per
                TODO item), as returned by run_llm_loop.
            todo_mapping: Mapping from executor index to question index.
            num_questions: Total number of original questions.

        Returns:
            List of concatenated findings strings, one per question.
        """
        grouped = ["" for _ in range(num_questions)]

        for exec_idx, q_idx in enumerate(todo_mapping):
            if exec_idx < len(exec_msg_strings):
                msg = exec_msg_strings[exec_idx]
                # Extract answer content if present
                answer_match = re.search(
                    r'<answer>(.*?)</answer>', msg, re.DOTALL
                )
                if answer_match:
                    finding = answer_match.group(1).strip()
                else:
                    # Use the last assistant response as finding
                    finding = self._extract_last_response(msg)

                grouped[q_idx] += f"\n--- Finding {exec_idx + 1} ---\n{finding}\n"

        # Fill in empty findings
        for i in range(num_questions):
            if not grouped[i].strip():
                grouped[i] = "[No findings available]"

        return grouped

    def _build_exec_trajectories(
        self,
        exec_msg_strings: List[str],
        todo_mapping: List[int],
        num_questions: int,
    ) -> List[List[Dict]]:
        """Build per-question executor trajectory metadata.

        Parses raw message strings into structured tool call entries
        with 'tool' and 'result' keys, matching the format expected
        by executor_rules().

        Returns:
            List of lists (one per question), each containing dicts with
            'tool' and 'result' keys.
        """
        trajectories = [[] for _ in range(num_questions)]

        for exec_idx, q_idx in enumerate(todo_mapping):
            raw = (
                exec_msg_strings[exec_idx]
                if exec_idx < len(exec_msg_strings)
                else ""
            )
            steps = self._parse_tool_steps(raw)
            trajectories[q_idx].extend(steps)

        return trajectories

    @staticmethod
    def _parse_tool_steps(raw_msg: str) -> List[Dict[str, str]]:
        """Extract structured tool call steps from a raw executor message.

        Looks for <tool_call>...</tool_call> followed by
        <observation>...</observation> pairs.
        """
        steps = []
        tool_call_pattern = re.compile(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
            r'.*?'
            r'<observation>(.*?)</observation>',
            re.DOTALL,
        )
        for match in tool_call_pattern.finditer(raw_msg):
            try:
                call = json.loads(match.group(1))
                tool_name = call.get("name", "")
            except (json.JSONDecodeError, AttributeError):
                tool_name = ""
            result = match.group(2).strip()
            steps.append({"tool": tool_name, "result": result})
        return steps

    def _extract_last_response(self, msg: str) -> str:
        """Extract the last meaningful response from a full message string.

        Looks for the last <think>...</think> block followed by content,
        or falls back to the last 500 characters.
        """
        # Try to find the last think block and what follows
        parts = msg.split("<think>")
        if len(parts) > 1:
            last_part = parts[-1]
            # Get content after </think>
            after_think = last_part.split("</think>")
            if len(after_think) > 1:
                return after_think[-1].strip()[:1000]

        # Fallback: return last portion of the message
        return msg[-500:].strip() if msg else ""

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_stage_outputs(
        self,
        stage_name: str,
        global_steps: int,
        questions: List[str],
        texts: List[str],
        parsed_todos: Optional[List[List[Dict]]] = None,
    ):
        """Log stage outputs to a JSON file for debugging."""
        output_dir = (
            f"./outputs/{self.config.project_name}/"
            f"{self.config.experiment_name}/rollout"
        )
        os.makedirs(output_dir, exist_ok=True)

        log_data = []
        for i, (question, text) in enumerate(zip(questions, texts)):
            entry = {
                "idx": i,
                "question": question,
                "output": text,
            }
            if parsed_todos is not None and i < len(parsed_todos):
                entry["parsed_todos"] = parsed_todos[i]
            log_data.append(entry)

        filepath = os.path.join(
            output_dir,
            f"{stage_name}_step_{global_steps}.json",
        )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

        print(f"[MultiAgent] {stage_name}_step_{global_steps}.json saved", flush=True)
