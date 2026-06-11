# Ported from IGPO (verl/utils/reward_score/info_gain.py) for Hi-IGPO.
# Computes token-level rewards: per-turn information gain (IG) at each turn-end token,
# and the final F1 (outcome) reward at the last turn's last token. Turn boundaries are
# located from the rendered solution string via the assistant separator + offset_mapping.
#
# Adaptations for this repo:
#   - dropped the unused `from openai import OpenAI` import.
#   - thinking is OFF here (no <think> tags), so the 'think'/'code' entries in
#     check_tags_balance are simply never present (0==0 → balanced); kept for parity.
import re
import string
import json


def check_tags_balance(solution_str: str) -> bool:
    """Check if tags are properly paired and correctly nested."""
    tags_to_check = ['code', 'tool_call', 'think', 'answer']

    for tag in tags_to_check:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)

        if start_count != end_count:
            return False

        last_pos = -1
        while True:
            start_pos = solution_str.find(start_tag, last_pos + 1)
            if start_pos == -1:
                break
            end_pos = solution_str.find(end_tag, start_pos)
            if end_pos == -1:
                return False
            last_pos = end_pos

    return True


def preprocess_text(text: str) -> str:
    """Lowercase-agnostic: strip punctuation and collapse whitespace."""
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def deal_multi_labels(ground_truth):
    for item in ground_truth:
        if item['label'].lower() == 'false':
            return 'false'
    return 'true'


def compute_f1(solution_str, ground_truth, data_source, val_type='f1') -> float:
    if data_source in ['Factbench', 'politifact', 'liar2']:
        ground_truth = json.loads(ground_truth)
        ground_truth = deal_multi_labels(ground_truth)
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    ground_truths = ground_truth.split("<|answer_split|>")
    # Format check first.
    if not check_tags_balance(solution_str):
        if val_type == 'noformatf1':
            return 0
        else:
            return -2.0

    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            answer_content = preprocess_text(answer_content)
        else:
            if val_type == 'noformatf1':
                return 0
            else:
                return -2.0
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        if val_type == 'noformatf1':
            return 0
        else:
            return -2.0

    max_score = 0.0

    for gt in ground_truths:
        gt = preprocess_text(gt)

        if val_type == 'em':
            if gt == answer_content:
                return 1.0
        else:
            pred_tokens = set(answer_content.split())
            gt_tokens = set(gt.split())

            if not gt_tokens:
                continue
            if not pred_tokens:
                continue

            common_tokens = pred_tokens & gt_tokens

            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                max_score = max(max_score, f1)

    return max_score


def _char_pos_to_token_idx(char_pos, offset_mapping):
    """Map a character position to its token index via offset_mapping."""
    for i, (start, end) in enumerate(offset_mapping):
        if start <= char_pos < end:
            return i
        if char_pos < start:
            return max(0, i - 1)
    return len(offset_mapping) - 1


def _assistant_text(solution_str: str, start: int, end: int) -> str:
    """The assistant-generated portion of a turn segment [start:end): from start up to
    the first <|im_end|>. Everything after (the env tool/user response) is excluded, so
    tool-response text (search snippets with stray angle brackets/tags) does not pollute
    the per-turn format check."""
    seg = solution_str[start:end]
    cut = seg.find("<|im_end|>")
    return seg if cut == -1 else seg[:cut]


def _turn_format_ok(asst_text: str, is_final: bool) -> bool:
    """Per-turn (DR-Venus eq.4) format validity on this turn's assistant output only:
    balanced tags AND the expected action tag (final turn → <answer>, else → <tool_call>)."""
    if not check_tags_balance(asst_text):
        return False
    if is_final:
        return "<answer>" in asst_text and "</answer>" in asst_text
    return "<tool_call>" in asst_text and "</tool_call>" in asst_text


def _answer_f1_from_text(asst_text: str, ground_truth, data_source, val_type='f1') -> float:
    """Pure F1/EM computed from the <answer> content of the final turn's assistant text,
    WITHOUT the whole-trajectory format gate (the -2 penalty). Format validity is handled
    separately and per-turn by the caller. Returns 0.0 when no <answer> is present."""
    if data_source in ['Factbench', 'politifact', 'liar2']:
        ground_truth = json.loads(ground_truth)
        ground_truth = deal_multi_labels(ground_truth)
    s = asst_text.lower()
    gt = ground_truth.lower()
    ground_truths = gt.split("<|answer_split|>")
    m = re.search(r'<answer>(.*?)</answer>', s, re.DOTALL)
    if not m:
        return 0.0
    answer_content = preprocess_text(m.group(1).strip())
    max_score = 0.0
    for g in ground_truths:
        g = preprocess_text(g)
        if val_type == 'em':
            if g == answer_content:
                return 1.0
        else:
            pred_tokens = set(answer_content.split())
            gt_tokens = set(g.split())
            if not gt_tokens or not pred_tokens:
                continue
            common_tokens = pred_tokens & gt_tokens
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
            if precision + recall > 0:
                max_score = max(max_score, 2 * (precision * recall) / (precision + recall))
    return max_score


def compute_score(solution_str, ground_truth, data_source, val_type='f1', info_gain_reward=[], tokenizer=None, is_validation=False, format_penalty=1.0):
    """Token-level reward: IG at each non-final turn-end token, F1 at the final turn.

    Turn-level format penalty (DR-Venus eq.4): each turn is graded independently — a
    well-formed turn keeps its reward (IG for non-final, pure F1 for final), a malformed
    turn's reward slot is set to -format_penalty. This replaces the previous whole-trajectory
    gate (one slip → final outcome=-2 → backward-accumulated over the whole rollout), which
    biased cold small models toward collapsing to single-turn direct answers.

    Returns a list of per-token scores aligned to tokenizer(solution_str) (no special
    tokens). With is_validation=True returns a dict with f1/em/noformatf1 + scores
    (validation keeps the legacy whole-trajectory metric for comparability).
    """
    if tokenizer is None:
        raise ValueError("tokenizer cannot be None")

    alpha = 1.0

    if is_validation:
        f1_score = compute_f1(solution_str, ground_truth, data_source, val_type='f1')
        em_score = compute_f1(solution_str, ground_truth, data_source, val_type='em')
        noformatf1_score = compute_f1(solution_str, ground_truth, data_source, val_type='noformatf1')
    else:
        f1_score = compute_f1(solution_str, ground_truth, data_source, val_type)

    encoding = tokenizer(solution_str, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = encoding['input_ids']
    offset_mapping = encoding['offset_mapping']

    tokens_size = len(token_ids)
    scores = [0.0] * tokens_size

    if tokens_size == 0:
        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
        return scores

    # Assistant-turn separator in the rendered chat string.
    separator = "\n<|im_start|>assistant\n"

    turn_start_positions = []
    turn_end_positions = []

    sep_positions = []
    search_pos = 0
    while True:
        sep_pos = solution_str.find(separator, search_pos)
        if sep_pos == -1:
            break
        sep_positions.append(sep_pos)
        search_pos = sep_pos + 1

    if len(sep_positions) == 0:
        turn_start_positions = [0]
        turn_end_positions = [len(solution_str)]
    else:
        if sep_positions[0] > 0:
            turn_start_positions.append(0)
            turn_end_positions.append(sep_positions[0])

        for i, sep_pos in enumerate(sep_positions):
            turn_start = sep_pos + len(separator)
            turn_start_positions.append(turn_start)

            if i + 1 < len(sep_positions):
                turn_end = sep_positions[i + 1]
            else:
                turn_end = len(solution_str)
            turn_end_positions.append(turn_end)

    chats_size = len(turn_start_positions)

    def _end_tok(turn_end_char):
        idx = _char_pos_to_token_idx(turn_end_char - 1, offset_mapping) if turn_end_char > 0 else 0
        return min(idx, tokens_size - 1)

    # Validation keeps the legacy whole-trajectory metric (comparability); val path does not
    # flow through this function in training (NaiveRewardManager -> format_and_f1).
    if is_validation:
        scores[-1] = alpha * f1_score
        return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}

    # Final turn's assistant output → pure F1 (no whole-traj -2 gate) + per-turn format check.
    final_asst = _assistant_text(solution_str, turn_start_positions[-1], turn_end_positions[-1])
    train_f1 = _answer_f1_from_text(final_asst, ground_truth, data_source, val_type)
    fmt_pen = -float(format_penalty)

    # Single turn / no IG / turn-count mismatch: only the final turn reward (format-gated).
    if info_gain_reward == [] or chats_size == 1 or len(info_gain_reward) != chats_size - 1:
        if len(info_gain_reward) != chats_size - 1 and info_gain_reward != [] and chats_size != 1:
            print(f"info_gain.py: turn mismatch - chats_size={chats_size}, info_gain_len={len(info_gain_reward)}")
        final_ok = _turn_format_ok(final_asst, is_final=True)
        scores[_end_tok(turn_end_positions[-1])] = (alpha * train_f1) if final_ok else fmt_pen
        return scores

    # Multi-turn: per-turn format-gated reward (DR-Venus eq.4) — keep IG/F1 on well-formed
    # turns, replace ONLY a malformed turn's slot with -format_penalty.
    for i in range(chats_size):
        is_final = (i == chats_size - 1)
        asst_i = _assistant_text(solution_str, turn_start_positions[i], turn_end_positions[i])
        end_tok = _end_tok(turn_end_positions[i])
        if not _turn_format_ok(asst_i, is_final):
            scores[end_tok] = fmt_pen
            continue
        if is_final:
            scores[end_tok] = alpha * train_f1
        else:
            ig_value = info_gain_reward[i]
            if ig_value == 0.0:
                ig_value = 1e-10  # keep non-zero so reward!=0 turn detection doesn't skip it
            scores[end_tok] = ig_value

    return scores
