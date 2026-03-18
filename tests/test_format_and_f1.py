import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from verl.utils.reward_score.format_and_f1 import compute_score, NO_TOOL_USE_PENALTY_FACTOR


class TestFormatErrors:
    """Format errors should always return -1.0 regardless of tool use."""

    def test_missing_answer_tag(self):
        assert compute_score("I think the answer is Paris", "paris") == -1.0

    def test_unbalanced_tags(self):
        response = "<tool_call>something<answer>Paris</answer>"
        assert compute_score(response, "paris") == -1.0

    def test_unbalanced_tool_call_tag(self):
        response = "<tool_call>search<answer>Paris</answer>"
        assert compute_score(response, "paris") == -1.0

    def test_empty_answer_tag(self):
        response = "<answer></answer>"
        assert compute_score(response, "paris") == 0.0


class TestWithToolCall:
    """Responses with <tool_call> should get full F1 score."""

    def test_exact_match_with_tool(self):
        response = (
            '<tool_call>{"name":"web_search","arguments":{"query":["Paris capital"]}}</tool_call>'
            "Search results: Paris is the capital of France. "
            "<answer>Paris</answer>"
        )
        score = compute_score(response, "paris")
        assert score == 1.0

    def test_partial_match_with_tool(self):
        response = (
            '<tool_call>{"name":"web_search","arguments":{"query":["mountain actor"]}}</tool_call>'
            "Search results: ... "
            "<answer>Hafthor Bjornsson actor</answer>"
        )
        score = compute_score(response, "hafthor bjornsson")
        assert score > 0.5
        assert score <= 1.0

    def test_wrong_answer_with_tool(self):
        response = (
            '<tool_call>{"name":"web_search","arguments":{"query":["capital france"]}}</tool_call>'
            "Search results: ... "
            "<answer>London</answer>"
        )
        score = compute_score(response, "paris")
        assert score == 0.0

    def test_multiple_tool_calls(self):
        response = (
            '<tool_call>{"name":"web_search","arguments":{"query":["q1"]}}</tool_call>'
            "Results: ... "
            '<tool_call>{"name":"web_search","arguments":{"query":["q2"]}}</tool_call>'
            "Results: ... "
            "<answer>Paris</answer>"
        )
        score = compute_score(response, "paris")
        assert score == 1.0


class TestWithoutToolCall:
    """Responses without <tool_call> should get penalized (score * 0.5)."""

    def test_exact_match_no_tool(self):
        response = "I know this. <answer>Paris</answer>"
        score = compute_score(response, "paris")
        assert score == 1.0 * NO_TOOL_USE_PENALTY_FACTOR

    def test_partial_match_no_tool(self):
        response = "<answer>Hafthor Bjornsson actor</answer>"
        score_no_tool = compute_score(response, "hafthor bjornsson")

        response_with_tool = (
            '<tool_call>{"name":"web_search","arguments":{"query":["x"]}}</tool_call>'
            "<answer>Hafthor Bjornsson actor</answer>"
        )
        score_with_tool = compute_score(response_with_tool, "hafthor bjornsson")

        assert score_no_tool == score_with_tool * NO_TOOL_USE_PENALTY_FACTOR

    def test_wrong_answer_no_tool_no_extra_penalty(self):
        response = "<answer>London</answer>"
        score = compute_score(response, "paris")
        assert score == 0.0

    def test_penalty_factor_value(self):
        assert NO_TOOL_USE_PENALTY_FACTOR == 0.5


class TestExactMatch:
    """EM mode should also respect the tool-use penalty."""

    def test_em_with_tool(self):
        response = (
            '<tool_call>{"name":"web_search","arguments":{"query":["x"]}}</tool_call>'
            "<answer>Paris</answer>"
        )
        score = compute_score(response, "paris", val_type='em')
        assert score == 1.0

    def test_em_without_tool(self):
        response = "<answer>Paris</answer>"
        score = compute_score(response, "paris", val_type='em')
        assert score == 1.0 * NO_TOOL_USE_PENALTY_FACTOR

    def test_em_wrong_answer(self):
        response = "<answer>London</answer>"
        score = compute_score(response, "paris", val_type='em')
        assert score == 0.0


class TestMultipleGroundTruths:
    """answer_split scenarios."""

    def test_multi_gt_with_tool(self):
        response = (
            '<tool_call>{"name":"web_search","arguments":{"query":["x"]}}</tool_call>'
            "<answer>New York City</answer>"
        )
        gt = "New York City<|answer_split|>NYC<|answer_split|>New York"
        score = compute_score(response, gt)
        assert score == 1.0

    def test_multi_gt_without_tool(self):
        response = "<answer>New York City</answer>"
        gt = "New York City<|answer_split|>NYC<|answer_split|>New York"
        score = compute_score(response, gt)
        assert score == 1.0 * NO_TOOL_USE_PENALTY_FACTOR


class TestEdgeCases:
    """Edge cases for robustness."""

    def test_tool_call_in_uppercase(self):
        response = "<TOOL_CALL>search</TOOL_CALL><answer>Paris</answer>"
        score = compute_score(response, "paris")
        assert score == 1.0

    def test_case_insensitive_detection(self):
        response = "<Tool_Call>search</Tool_Call><answer>Paris</answer>"
        score = compute_score(response, "paris")
        assert score == 1.0

    def test_tool_call_tag_inside_content(self):
        response = (
            "The model used <tool_call> to search. <answer>Paris</answer>"
        )
        score = compute_score(response, "paris")
        assert score == -1.0  # unbalanced: has <tool_call> but no </tool_call>
