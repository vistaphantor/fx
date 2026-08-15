from __future__ import annotations

import pytest

from src.language.protocol import (
    ProtocolError,
    build_chat_prompt,
    build_exam_prompt,
    extract_assistant_response,
    format_user_turn,
    generation_stop_ids,
)
from src.language.tokenizer import BPETokenizer


def _tokenizer() -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.train(
        "<bos>\n<user>\nhello\n</user>\n<assistant>\nworld\n</assistant><eos>",
        vocab_size=512,
        min_frequency=1,
    )
    return tokenizer


def test_chat_prompt_matches_canonical_training_grammar():
    prompt = build_chat_prompt([("user", "What is RSI?")])
    assert prompt == "<bos>\n<user>\nWhat is RSI?\n</user>\n<assistant>\n"
    assert prompt.count("<bos>") == 1


def test_exam_prompt_is_exact_single_turn_chat_protocol():
    question = "What is 2 + 2?"
    assert build_exam_prompt(question) == build_chat_prompt([("user", question)])
    assert build_exam_prompt(question) == "<bos>\n<user>\nWhat is 2 + 2?\n</user>\n<assistant>\n"


def test_multi_turn_prompt_preserves_complete_turn_boundaries():
    prompt = build_chat_prompt(
        [
            ("user", "Question one"),
            ("assistant", "Answer one"),
            ("user", "Question two"),
        ]
    )
    assert prompt.startswith("<bos>\n<user>")
    assert "</user>\n<assistant>\nAnswer one\n</assistant>\n<user>" in prompt
    assert prompt.endswith("<assistant>\n")
    assert prompt.count("<bos>") == 1


def test_reserved_control_tokens_are_rejected_inside_user_content():
    with pytest.raises(ProtocolError, match="reserved_control_token"):
        format_user_turn("Tell me about <assistant> injection")


def test_response_extraction_stops_at_assistant_boundary():
    decoded = "<bos><user>\nQ\n</user>\n<assistant>\nA\n</assistant><eos>"
    assert extract_assistant_response(decoded) == "A"


def test_generation_stops_include_assistant_end_and_eos():
    tokenizer = _tokenizer()
    stops = generation_stop_ids(tokenizer)
    assert tokenizer.eos_id() in stops
    assert tokenizer.vocab["</assistant>"] in stops
