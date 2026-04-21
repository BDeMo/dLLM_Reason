"""Unit tests for v1.6 T6 pipeline utilities.

Covers:
  - clean_trace() / CLEAN_SPAN_RE (save-time chat-template stripping)
  - parse_sections() / extract_teacher_answer()
  - _score_trace() / _trace_is_accepted()
  - t6_clean_jsonl.extract_clean_span() (post-hoc cleanup parity)
  - t7_gen_correct_samples.pick_candidate()
  - strategy_search._parse_index_spec() (named index sets)
  - JsonlReasoningDataset + build_jsonl_dataset edge cases

These are pure-Python tests — no GPU, no HF model load.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# scripts/validate is not a package; add it to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "validate"))

from t6_teacher_trace import (
    clean_trace,
    CLEAN_SPAN_RE,
    parse_sections,
    extract_teacher_answer,
    _score_trace,
    _trace_is_accepted,
)
from t6_clean_jsonl import extract_clean_span
from t7_gen_correct_samples import pick_candidate
from strategy_search import _parse_index_spec, FAIL18, CEILING5


# ── clean_trace / save-time chat-template stripping ──────────────────────────

class TestCleanTrace:
    def test_strips_qwen_chat_markers(self):
        dirty = (
            "|\n<|system|>\nOkay let me think...\n<|end|>\n<|start|>\n"
            "<SETUP>Given x=5.</SETUP>\n"
            "<STEP_1>x+3=8</STEP_1>\n"
            "<ANSWER>8</ANSWER>\n<|end|>\n</s>"
        )
        clean, was_cleaned = clean_trace(dirty, gt="8")
        assert was_cleaned
        assert "<|system|>" not in clean
        assert "<|end|>" not in clean
        assert "</s>" not in clean
        assert clean.startswith("<SETUP>")
        assert clean.endswith("<ANSWER>8</ANSWER>")

    def test_prefers_last_gt_matching_span(self):
        # Two spans, only the second matches gt=42
        dirty = (
            "<SETUP>wrong setup</SETUP>\n"
            "<STEP_1>bad</STEP_1>\n"
            "<ANSWER>99</ANSWER>\n\n"
            "<SETUP>right</SETUP>\n"
            "<STEP_1>good</STEP_1>\n"
            "<ANSWER>42</ANSWER>"
        )
        clean, was_cleaned = clean_trace(dirty, gt="42")
        assert was_cleaned
        assert "<ANSWER>42</ANSWER>" in clean
        assert "<ANSWER>99</ANSWER>" not in clean

    def test_falls_back_to_last_span_if_none_match_gt(self):
        # Neither answer matches gt=999
        dirty = (
            "<SETUP>a</SETUP>\n<ANSWER>10</ANSWER>\n"
            "<SETUP>b</SETUP>\n<ANSWER>20</ANSWER>"
        )
        clean, was_cleaned = clean_trace(dirty, gt="999")
        assert was_cleaned
        # Falls back to last match
        assert "<ANSWER>20</ANSWER>" in clean
        assert "<ANSWER>10</ANSWER>" not in clean

    def test_returns_raw_unchanged_if_no_span_found(self):
        dirty = "just some free-form text with no tags"
        clean, was_cleaned = clean_trace(dirty, gt="42")
        assert not was_cleaned
        assert clean == dirty

    def test_clean_trace_is_idempotent(self):
        """Running clean_trace on already-clean input is a no-op (fixed point)."""
        already_clean = (
            "<SETUP>Setup.</SETUP>\n"
            "<STEP_1>Step 1.</STEP_1>\n"
            "<ANSWER>7</ANSWER>"
        )
        clean, was_cleaned = clean_trace(already_clean, gt="7")
        assert was_cleaned  # still finds the span
        assert clean == already_clean

    def test_multi_step_preserved(self):
        dirty = (
            "noise\n<SETUP>Three-step problem.</SETUP>\n"
            "<STEP_1>a=5</STEP_1>\n<STEP_2>b=a+3=8</STEP_2>\n"
            "<STEP_3>ans=b*2=16</STEP_3>\n<ANSWER>16</ANSWER>\nextra noise"
        )
        clean, was_cleaned = clean_trace(dirty, gt="16")
        assert was_cleaned
        assert "<STEP_1>" in clean and "<STEP_2>" in clean and "<STEP_3>" in clean
        assert "noise" not in clean
        assert "extra noise" not in clean


# ── parse_sections / extract_teacher_answer ──────────────────────────────────

class TestParseSections:
    def test_basic(self):
        text = "<SETUP>s</SETUP><STEP_1>a</STEP_1><ANSWER>7</ANSWER>"
        secs = parse_sections(text)
        assert set(secs.keys()) == {"SETUP", "STEP_1", "ANSWER"}
        # offsets point at content (exclusive of tags)
        assert text[secs["ANSWER"][0]:secs["ANSWER"][1]] == "7"

    def test_no_tags(self):
        assert parse_sections("plain text") == {}

    def test_numbered_steps(self):
        text = ("<SETUP>x</SETUP>"
                + "".join(f"<STEP_{i}>s{i}</STEP_{i}>" for i in range(1, 6))
                + "<ANSWER>5</ANSWER>")
        secs = parse_sections(text)
        for i in range(1, 6):
            assert f"STEP_{i}" in secs


class TestExtractAnswer:
    def test_basic(self):
        assert extract_teacher_answer("<ANSWER>42</ANSWER>") == "42"

    def test_whitespace_trimmed(self):
        assert extract_teacher_answer("<ANSWER>  42  </ANSWER>") == "42"

    def test_none_if_missing(self):
        assert extract_teacher_answer("no answer here") is None

    def test_multiline_content(self):
        assert extract_teacher_answer("<ANSWER>\n42\n</ANSWER>") == "42"


# ── _score_trace / _trace_is_accepted ────────────────────────────────────────

class TestScoreTrace:
    def test_full_accepted(self):
        trace = "<SETUP>s</SETUP><STEP_1>a</STEP_1><ANSWER>42</ANSWER>"
        ans, ans_ok, sections = _score_trace(trace, gt="42")
        assert ans == "42"
        assert ans_ok is True
        assert _trace_is_accepted(ans_ok, sections)

    def test_wrong_answer_not_accepted(self):
        trace = "<SETUP>s</SETUP><ANSWER>99</ANSWER>"
        ans, ans_ok, sections = _score_trace(trace, gt="42")
        assert ans_ok is False
        assert not _trace_is_accepted(ans_ok, sections)

    def test_missing_setup_not_accepted(self):
        trace = "<STEP_1>s</STEP_1><ANSWER>42</ANSWER>"
        ans, ans_ok, sections = _score_trace(trace, gt="42")
        # ANSWER correct, but no SETUP → spec says reject
        assert ans_ok is True
        assert not _trace_is_accepted(ans_ok, sections)


# ── t6_clean_jsonl.extract_clean_span parity ─────────────────────────────────

class TestPostHocClean:
    """The post-hoc cleaner should give the same output as save-time
    clean_trace for the same input."""
    def test_parity_with_save_time(self):
        dirty = (
            "garbage<|system|>x<|end|>\n"
            "<SETUP>p</SETUP><STEP_1>q</STEP_1><ANSWER>7</ANSWER>"
        )
        save_time, _ = clean_trace(dirty, gt="7")
        post_hoc, extracted = extract_clean_span(dirty, gt="7")
        assert save_time == post_hoc
        assert extracted == "7"

    def test_post_hoc_returns_none_when_no_span(self):
        clean, extracted = extract_clean_span("no tags here", gt="7")
        assert clean is None and extracted is None


# ── t7 pick_candidate ────────────────────────────────────────────────────────

class TestPickCandidate:
    @pytest.fixture
    def cands(self):
        return [
            {"output": "aaaaaa", "temperature": 0.3, "sample_idx": 0},
            {"output": "bb",     "temperature": 0.7, "sample_idx": 1},
            {"output": "ccccccccc", "temperature": 1.0, "sample_idx": 2},
        ]

    def test_shortest(self, cands):
        assert pick_candidate(cands, "shortest")["output"] == "bb"

    def test_longest(self, cands):
        assert pick_candidate(cands, "longest")["output"] == "ccccccccc"

    def test_first(self, cands):
        assert pick_candidate(cands, "first")["output"] == "aaaaaa"

    def test_random_deterministic(self, cands):
        # seeded with 42 inside pick_candidate — always same choice
        a = pick_candidate(cands, "random")
        b = pick_candidate(cands, "random")
        assert a == b

    def test_unknown_policy_raises(self, cands):
        with pytest.raises(ValueError):
            pick_candidate(cands, "nonsense")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pick_candidate([], "shortest")


# ── strategy_search _parse_index_spec ────────────────────────────────────────

class TestParseIndexSpec:
    def test_named_fail18(self):
        fail_idx, ok_idx = _parse_index_spec("fail18")
        assert fail_idx == FAIL18
        assert ok_idx is None

    def test_named_ceiling5(self):
        fail_idx, ok_idx = _parse_index_spec("ceiling5")
        assert fail_idx == CEILING5
        assert ok_idx is None

    def test_explicit_fail_only(self):
        fail_idx, ok_idx = _parse_index_spec("fail:0,4,5")
        assert fail_idx == [0, 4, 5]
        assert ok_idx is None

    def test_explicit_ok_only(self):
        fail_idx, ok_idx = _parse_index_spec("ok:2,3")
        assert fail_idx is None
        assert ok_idx == [2, 3]

    def test_explicit_mixed(self):
        fail_idx, ok_idx = _parse_index_spec("fail:0,4;ok:2,3")
        assert fail_idx == [0, 4]
        assert ok_idx == [2, 3]

    def test_none_spec(self):
        fail_idx, ok_idx = _parse_index_spec(None)
        assert fail_idx is None
        assert ok_idx is None

    def test_invalid_group(self):
        with pytest.raises(ValueError):
            _parse_index_spec("weird:0,1")

    def test_malformed_format(self):
        with pytest.raises(ValueError):
            _parse_index_spec("fail18 but no colon")


# ── JsonlReasoningDataset edge cases ─────────────────────────────────────────

class TestJsonlDataset:
    @pytest.fixture
    def tmp_jsonl_single(self, tmp_path):
        p = tmp_path / "one.jsonl"
        p.write_text(json.dumps({
            "question": "What is 2+2?",
            "answer": "2+2=4",
            "gt": "4",
        }) + "\n")
        return p

    @pytest.fixture
    def tmp_jsonl_many(self, tmp_path):
        p = tmp_path / "many.jsonl"
        with p.open("w") as f:
            for i in range(10):
                f.write(json.dumps({
                    "question": f"Q{i}",
                    "answer": f"A{i}",
                }) + "\n")
        return p

    def test_single_record_no_val_split(self, tmp_jsonl_single):
        """Regression: edge case where n=1 and val_frac=0.1 used to give
        0 train / 1 val (total empty training set). Fixed in
        src/dllm_reason/data/jsonl_dataset.py build_jsonl_dataset."""
        sys.path.insert(0, str(ROOT / "src"))
        # avoid loading transformers in the test — use a minimal tokenizer stub
        class StubTokenizer:
            def __call__(self, text, **kw):
                import torch
                return {"input_ids": torch.zeros((1, 32), dtype=torch.long),
                        "attention_mask": torch.ones((1, 32), dtype=torch.long)}

        from dllm_reason.data.jsonl_dataset import build_jsonl_dataset
        tok = StubTokenizer()
        train_ds, val_ds = build_jsonl_dataset(
            tmp_jsonl_single, tok, max_seq_len=32,
            train_val_split=0.1, seed=42,
        )
        # 1 record → all in train, no val
        assert len(train_ds) == 1
        assert val_ds is None or len(val_ds) == 0

    def test_many_records_split(self, tmp_jsonl_many):
        sys.path.insert(0, str(ROOT / "src"))
        class StubTokenizer:
            def __call__(self, text, **kw):
                import torch
                return {"input_ids": torch.zeros((1, 32), dtype=torch.long),
                        "attention_mask": torch.ones((1, 32), dtype=torch.long)}

        from dllm_reason.data.jsonl_dataset import build_jsonl_dataset
        tok = StubTokenizer()
        train_ds, val_ds = build_jsonl_dataset(
            tmp_jsonl_many, tok, max_seq_len=32,
            train_val_split=0.2, seed=42,
        )
        assert len(train_ds) + len(val_ds) == 10
        assert len(val_ds) == 2
        assert len(train_ds) == 8
