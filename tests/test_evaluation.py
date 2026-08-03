import torch

from steering_poc.generate import _censoring_record, generate_batch
from steering_poc.metrics import bootstrap_mean_ci, spearman


class _FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{t}" for t in ids)


def test_censoring_record_eos():
    rec = _censoring_record([5, 6, 99, 0, 0], eos_ids={99}, pad_id=0, max_new=10,
                            tokenizer=_FakeTok())
    assert rec["reached_eos"] is True
    assert rec["hit_max_new_tokens"] is False
    assert rec["generated_tokens"] == 2
    assert rec["token_ids"] == [5, 6]


def test_censoring_record_truncated():
    rec = _censoring_record([5, 6, 7, 8], eos_ids={99}, pad_id=0, max_new=4,
                            tokenizer=_FakeTok())
    assert rec["reached_eos"] is False
    assert rec["hit_max_new_tokens"] is True
    assert rec["generated_tokens"] == 4


def test_generate_batch_records(tiny_model):
    """Batched generation returns full censoring records for every prompt.
    The tiny random model never emits EOS meaningfully, so with a small cap
    every record must be flagged censored, never reported as complete."""
    from transformers import AutoTokenizer  # noqa: F401  (not used: fake path)

    class Tok:
        pad_token_id = 0
        eos_token_id = 1
        chat_template = None

        def __call__(self, text, return_tensors=None):
            class R:
                input_ids = torch.randint(2, 128, (1, 6))
            return R()

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(int(t)) for t in ids)

    gen_cfg = {"max_new_tokens": 5, "do_sample": False, "chat_template": False}
    recs = generate_batch(tiny_model, Tok(), ["a", "bb"], gen_cfg, "cpu")
    assert len(recs) == 2
    for rec in recs:
        assert set(rec) >= {"token_ids", "text", "generated_tokens",
                            "reached_eos", "hit_max_new_tokens"}
        assert rec["reached_eos"] or rec["hit_max_new_tokens"] or \
            rec["generated_tokens"] < 5


def test_bootstrap_ci_contains_mean():
    mean, lo, hi = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=2000)
    assert lo <= mean <= hi
    assert mean == 3.0


def test_spearman_monotone():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9
    ties = spearman([1, 1, 2, 2], [3, 3, 5, 5])
    assert 0.99 < ties <= 1.0
