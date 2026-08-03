from steering_poc.qualcomm.profile_qwen3_patch_links import _graph_name


def test_qwen3_profile_graph_names_from_summary():
    summary = {"sequence_lengths": "1,128", "context_lengths": "512"}

    assert _graph_name(summary, "prompt", 3, 4) == "prompt_ar128_cl512_3_of_4"
    assert _graph_name(summary, "token", 3, 4) == "token_ar1_cl512_3_of_4"
