"""Hi-IGPO Phase 2b: interleaved Planner <-> frozen Executor generation.

Outer loop: the Planner (trained LoRA) proposes one subtask per turn, or emits
<answer> to stop (<= max_planner_turns). Inner: each subtask is executed by the
FROZEN Executor (Phase-1 model, no grad) to produce findings, which feed the next
Planner turn. After each Planner turn a belief Bel_t is computed; IG_t = Bel_t - Bel_{t-1}.

`run_loop_pure` is the tensor-free control flow — unit-testable and the skeleton for the
real `run_loop`. `run_loop` reuses MultiAgentGenerationManager's dual-LoRA rollout helpers
(_build_planner_batch / _generate_with_gpu_padding(lora=) / run_llm_loop(lora=) /
_tokenize_messages_to_batch / _decode_outputs) and is iterated on the server (GPU-only).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class InterleavedGenerationManager:
    """Pure-python control flow (no torch) — see run_loop_pure. The tensor rollout lives in
    InterleavedRolloutManager (subclasses MultiAgentGenerationManager) below to avoid importing
    heavy deps when only the control flow is needed (unit tests)."""

    def __init__(self, planner, executor, max_planner_turns=5, **kw):
        self.planner = planner
        self.executor = executor
        self.max_planner_turns = max_planner_turns

    def run_loop_pure(self, question):
        """Pure-python control flow (no tensors). Returns the interleaved trace.

        planner.step(ctx) -> (output_str, is_answer);  executor.run(subtask, ctx) -> findings_str
        Returns dict(answer, subtasks, findings).
        """
        subtasks, findings = [], []
        answer = None
        ctx = {"question": question, "findings": findings}
        for _ in range(self.max_planner_turns):
            out, is_answer = self.planner.step(ctx)
            if is_answer:
                answer = out.replace("<answer>", "").replace("</answer>", "").strip()
                break
            subtasks.append(out)
            findings.append(self.executor.run(out, ctx))
        return {"answer": answer, "subtasks": subtasks, "findings": findings}


@dataclass
class InterleavedTrace:
    """Per-sample interleaved rollout record (assembled into tensors by run_loop)."""
    question: str
    planner_texts: List[str] = field(default_factory=list)   # planner generation per turn (trained)
    subtasks: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)        # executor output per subtask (a:answer + b:evidence)
    answer: Optional[str] = None
    # token-level (filled during assembly): each planner turn's end-token position in the planner seq
    turn_end_positions: List[int] = field(default_factory=list)
    beliefs: List[float] = field(default_factory=list)       # Bel_0..Bel_T


def assemble_planner_sequence(planner_turn_ids, findings_ids):
    """Interleave Planner per-turn token ids with Executor findings into one response frame.

    planner_turn_ids: List[List[int]] — token ids of each Planner generation turn, in order;
        entries [0..T-1] are subtask turns, the LAST entry is the answer turn (T subtasks total).
    findings_ids: List[List[int]] — Executor findings observation per subtask
        (len == len(planner_turn_ids) - 1; the answer turn has no findings after it).

    Layout (response frame), findings masked out of the loss:
        [subtask_0][findings_0][subtask_1][findings_1]...[subtask_{T-1}][findings_{T-1}][answer]

    Returns (response_ids, loss_mask, turn_end_positions):
      - loss_mask: 1 on Planner tokens (trained), 0 on findings observations.
      - turn_end_positions: last-token index of each Planner turn in the response frame,
        ordered [subtask_0_end, ..., subtask_{T-1}_end, answer_end]; length == #subtasks + 1,
        matching scatter_planner_token_rewards (T IG turn-ends + 1 F1/answer position).
    """
    if not planner_turn_ids:
        raise ValueError("planner_turn_ids must contain at least the answer turn")
    if len(findings_ids) != len(planner_turn_ids) - 1:
        raise ValueError(
            f"findings_ids ({len(findings_ids)}) must equal #subtask turns "
            f"({len(planner_turn_ids) - 1}) = len(planner_turn_ids) - 1"
        )

    response_ids, loss_mask, turn_end_positions = [], [], []
    for t, turn_ids in enumerate(planner_turn_ids):
        if len(turn_ids) == 0:
            raise ValueError(f"planner turn {t} is empty")
        response_ids.extend(turn_ids)
        loss_mask.extend([1] * len(turn_ids))
        turn_end_positions.append(len(response_ids) - 1)   # last planner token of this turn
        # findings observation follows every subtask turn (not the final answer turn)
        if t < len(findings_ids):
            obs = findings_ids[t]
            response_ids.extend(obs)
            loss_mask.extend([0] * len(obs))
    return response_ids, loss_mask, turn_end_positions


def _parse_planner_turn(text: str):
    """(is_answer, payload). <answer>..</answer> -> answer; else the text is a subtask."""
    if "<answer>" in text and "</answer>" in text:
        return True, text.split("<answer>")[1].split("</answer>")[0].strip()
    # else: treat reasoning+content as the subtask instruction (align with planner prompt on server)
    return False, text.strip()


def build_interleaved_rollout_manager(tokenizer, actor_rollout_wg, config, lora_save_dir,
                                      max_planner_turns=5, gt_computer=None):
    """Factory: returns an InterleavedRolloutManager (defined lazily to avoid importing
    MultiAgentGenerationManager / torch when only run_loop_pure is needed)."""
    from scrl.llm_agent.multi_agent_generation import MultiAgentGenerationManager
    from research_agent.prompts.planner import get_planner_prompt
    from research_agent.prompts.executor import get_executor_prompt

    class InterleavedRolloutManager(MultiAgentGenerationManager):
        """Real interleaved rollout. Reuses the dual-LoRA helpers of MultiAgentGenerationManager;
        replaces its one-shot DAG `run_multi_agent_loop` with the Planner<->frozen-Executor
        interleave. Per-turn Planner belief is computed via the SAME actor-worker vectorized path
        as single-agent IGPO (prealigned_vectorized.compute_vectorized_gt_logprob over
        actor_rollout_wg) — NOT the in-process gt_computer (that path can't reach the FSDP/vLLM
        model during training; see design §11). gt_computer is kept only for offline belief smoke.

        ⚠ SERVER-ITERATED: belief collection + tensor assembly run only against the live runtime
        (GPU, Ray actor worker, real tokenizer). The model-independent core (assemble_planner_sequence)
        is unit-tested; everything touching DataProto/compute_log_prob is finalized on the server.
        """

        def __init__(self):
            super().__init__(tokenizer=tokenizer, actor_rollout_wg=actor_rollout_wg,
                             config=config, lora_save_dir=lora_save_dir)
            self.max_planner_turns = max_planner_turns
            self.gt_computer = gt_computer   # offline belief smoke only; training uses actor path

        # -- GT answer wrapping for belief (mirror igpo_generation lines ~528-572) --------
        def _build_pseudo_gt(self, ground_truths):
            """Return (pseudo_resps_with_gt: List[List[int]], gt_idx: List[List[int]]).

            Wraps each ground-truth answer with the no-think PREFIX/SUFFIX and locates the answer
            token range via offset_mapping — identical to the single-agent IGPO construction so the
            belief = P(gt_answer | context) is computed over the same span.
            """
            from scrl.llm_agent.vectorized_gt_logprob import (
                DEFAULT_GT_ANSWER_PREFIX, DEFAULT_GT_ANSWER_SUFFIX,
            )
            PREFIX, SUFFIX = DEFAULT_GT_ANSWER_PREFIX, DEFAULT_GT_ANSWER_SUFFIX
            pseudo_resps_with_gt, gt_idx = [], []
            for gt in ground_truths:
                gt_text = gt["ground_truth"]
                if "<|answer_split|>" in gt_text:
                    gt_text = gt_text.split("<|answer_split|>")[0]
                gt_text = gt_text.strip()
                full_text = f"{PREFIX}{gt_text}{SUFFIX}"
                enc = self.tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
                token_ids = enc["input_ids"].tolist()[0]
                offsets = enc["offset_mapping"].tolist()[0]
                pseudo_resps_with_gt.append(token_ids)
                if not token_ids:
                    gt_idx.append([0, 0]); continue
                gt_char_start, gt_char_end = len(PREFIX), len(PREFIX) + len(gt_text)
                tok_start = tok_end = None
                for ti, (cs, ce) in enumerate(offsets):
                    if tok_start is None and ce > gt_char_start:
                        tok_start = ti
                    if cs < gt_char_end and ce > 0:
                        tok_end = ti + 1
                gt_idx.append([tok_start if tok_start is not None else len(token_ids),
                               tok_end if tok_end is not None else len(token_ids)])
            return pseudo_resps_with_gt, gt_idx

        def _planner_belief_pseudo(self, planner_msgs_active, pseudo_resps_active, gen_batch):
            """Tokenize the active Planner contexts (no tools) and append the GT answer via
            pseudo_generate_sequences -> one pseudo DataProto for this turn's belief forward."""
            import torch
            from verl import DataProto
            rolled = self.tokenizer.apply_chat_template(
                planner_msgs_active, add_generation_prompt=True, tokenize=False)
            tok = self.tokenizer(rolled, return_tensors="pt", padding=True)
            pad_mask = tok["input_ids"] != self.tokenizer.pad_token_id
            order = pad_mask.to(torch.int64).argsort(dim=1, stable=True)   # left-pad
            input_ids = tok["input_ids"].gather(1, order)
            attn = tok["attention_mask"].gather(1, order)
            pos = self.tensor_fn.create_position_ids(attn)
            rollings = DataProto.from_dict(
                {"input_ids": input_ids, "attention_mask": attn, "position_ids": pos})
            return self.pseudo_generate_sequences(rollings, pseudo_resps_active)

        def run_loop(self, gen_batch, global_steps, ground_truths):
            """Interleaved Planner(LoRA)<->frozen Executor rollout.

            Returns (planner_output: DataProto, turn_end_positions: List[List[int]],
                     info_gain_rewards: List[List[float]], answers: List[str]).
            info_gain_rewards[i] holds Planner per-turn IG (from belief diffs, computed by the
            actor-worker vectorized path). ray_trainer then calls
            scatter_info_gain_rewards(info_gain_rewards, f1, turn_end_positions, L) -> (bs,L)
            token_level_rewards/turn_boundary_mask -> compute_igpo_turn_advantage.
            """
            import numpy as np  # noqa

            questions = self._extract_questions(gen_batch)
            n = len(questions)
            traces = [InterleavedTrace(question=q) for q in questions]
            planner_msgs: List[List[Dict]] = [list(get_planner_prompt(q)) for q in questions]
            active = list(range(n))

            # Belief setup: GT wrapping (per global sample) + per-turn pseudo collection.
            pseudo_resps_with_gt, gt_idx = self._build_pseudo_gt(ground_truths)
            pseudo_outputs_per_turn, activate_lists_per_turn = [], []

            for t in range(self.max_planner_turns):
                if not active:
                    break
                # --- Belief Bel_t: P(gt | planner context BEFORE this turn's generation) ---
                # context after t subtasks+findings; consecutive diffs => IG_t (vectorized after loop).
                pseudo_resps_active = [pseudo_resps_with_gt[i] for i in active]
                pseudo_outputs_per_turn.append(
                    self._planner_belief_pseudo([planner_msgs[i] for i in active],
                                                pseudo_resps_active, gen_batch))
                activate_lists_per_turn.append(list(active))

                # --- Planner: one generation per active sample ---
                batch = self._tokenize_messages_to_batch([planner_msgs[i] for i in active], gen_batch)
                plan_out = self._generate_with_gpu_padding(batch, lora_adapter_name="planner")
                texts = self._decode_outputs(plan_out)

                next_active, subtask_rows = [], []
                for k, i in enumerate(active):
                    is_answer, payload = _parse_planner_turn(texts[k])
                    traces[i].planner_texts.append(texts[k])
                    planner_msgs[i].append({"role": "assistant", "content": texts[k]})
                    if is_answer:
                        traces[i].answer = payload
                    else:
                        traces[i].subtasks.append(payload)
                        subtask_rows.append(i)
                        next_active.append(i)

                # --- Executor (FROZEN): run each subtask, inject findings back ---
                if subtask_rows:
                    exec_msgs = [get_executor_prompt(traces[i].subtasks[-1]) for i in subtask_rows]
                    exec_batch = self._tokenize_messages_to_batch(
                        exec_msgs, gen_batch, tools=getattr(self, "tools", None))
                    _, exec_out = self.run_llm_loop(
                        exec_batch, global_steps, lora_adapter_name="executor")
                    findings = self._decode_outputs(exec_out)   # a:answer + b:evidence (design §3)
                    for j, i in enumerate(subtask_rows):
                        traces[i].findings.append(findings[j])
                        planner_msgs[i].append({"role": "user", "content": findings[j]})
                active = next_active

            # ---- Belief: single batched actor forward over all collected turns (= IGPO path) ----
            from scrl.llm_agent.prealigned_vectorized import compute_vectorized_gt_logprob
            info_gain_type = getattr(self.config, "info_gain_type", "log_prob_diff")
            if pseudo_outputs_per_turn:
                vec = compute_vectorized_gt_logprob(
                    pseudo_outputs_per_turn=pseudo_outputs_per_turn,
                    activate_lists_per_turn=activate_lists_per_turn,
                    gt_idx=gt_idx,
                    actor_rollout_wg=self.actor_rollout_wg,
                    tokenizer=self.tokenizer,
                    info_gain_type=info_gain_type,
                )
                info_gain_rewards = vec["info_gain_rewards"]
            else:
                info_gain_rewards = [[] for _ in range(n)]

            planner_output, turn_end_positions, answers = self._assemble_planner_tensors(
                traces, gen_batch)
            return planner_output, turn_end_positions, info_gain_rewards, answers

        def _assemble_planner_tensors(self, traces, gen_batch):
            """Build the Planner training DataProto from traces (model-independent assembly).

            Per sample: re-tokenize each planner turn + each findings observation, interleave via
            assemble_planner_sequence (findings masked out of the loss), then pad to a batch.
            Emits the run_llm_loop output contract (prompts/responses/input_ids/attention_mask/
            position_ids + agent_grpo_idx) PLUS a `loss_mask` (response-frame, 1 on planner tokens):
            downstream IGPO uses loss_mask as the response_mask so only planner tokens are credited.

            Returns (planner_output: DataProto, turn_end_positions: List[List[int]], answers: List[str]).
            ⚠ SERVER-ITERATED: re-tokenizing decoded text (vs reusing generated ids) and the exact
            downstream mask field name are verified against the live runtime.
            """
            import torch
            import numpy as np
            from verl import DataProto

            def _ids(text):
                return self.tokenizer(text, add_special_tokens=False)["input_ids"]

            resp_rows, mask_rows, tep_rows, prompt_strs, answers = [], [], [], [], []
            for tr in traces:
                # planner turns = subtasks (in order) + the answer turn last; one findings per subtask.
                planner_turn_ids = [_ids(s) for s in tr.planner_texts]
                findings_ids = [_ids(f) for f in tr.findings]
                # guard: assemble requires findings == planner_turns - 1; a capped (no-answer) trace
                # has len(planner_texts) == len(findings); drop the trailing dangling findings.
                if len(findings_ids) >= len(planner_turn_ids):
                    findings_ids = findings_ids[: len(planner_turn_ids) - 1]
                resp, mask, tep = assemble_planner_sequence(planner_turn_ids, findings_ids)
                resp_rows.append(resp); mask_rows.append(mask); tep_rows.append(tep)
                prompt_strs.append(self.tokenizer.apply_chat_template(
                    list(get_planner_prompt(tr.question)), add_generation_prompt=True, tokenize=False))
                answers.append(tr.answer or "")

            pad_id = self.tokenizer.pad_token_id
            # prompts: left-pad
            ptok = self.tokenizer(prompt_strs, return_tensors="pt", padding=True)
            pmask = ptok["input_ids"] != pad_id
            porder = pmask.to(torch.int64).argsort(dim=1, stable=True)
            prompts = ptok["input_ids"].gather(1, porder)
            prompts_attn = ptok["attention_mask"].gather(1, porder)
            # responses: right-pad ids + masks to batch max
            R = max((len(r) for r in resp_rows), default=1)
            responses = torch.full((len(resp_rows), R), pad_id, dtype=torch.long)
            resp_attn = torch.zeros((len(resp_rows), R), dtype=prompts_attn.dtype)
            loss_mask = torch.zeros((len(resp_rows), R), dtype=torch.long)
            for r, (resp, mask) in enumerate(zip(resp_rows, mask_rows)):
                L = len(resp)
                responses[r, :L] = torch.tensor(resp, dtype=torch.long)
                resp_attn[r, :L] = 1
                loss_mask[r, :L] = torch.tensor(mask, dtype=torch.long)

            attention_mask = torch.cat((prompts_attn, resp_attn), dim=-1)
            position_ids = self.tensor_fn.create_position_ids(attention_mask)
            planner_output = DataProto.from_dict({
                "prompts": prompts,
                "responses": responses,
                "input_ids": torch.cat((prompts, responses), dim=-1),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,            # response-frame planner-token mask (IGPO response_mask)
            })
            return planner_output, tep_rows, answers

    return InterleavedRolloutManager()
