import json

from src.language.canonical_contract import canonicalize_serialized, normalize_text
from src.language.data_pipeline import (
    TrainingExample,
    TrainingMessage,
    _example_from_embedded_object,
    _parse_source_example,
    build_tokenizer_training_sample,
    serialize_training_example,
)


def test_generic_prompt_does_not_duplicate_role_prefixes():
    parsed = _parse_source_example(
        {"prompt": "Human: What is 2 + 2?", "response": "Assistant: 4"},
        source="test",
    )
    assert parsed is not None
    text = serialize_training_example(parsed)
    assert text == "<bos>\n<user>\nWhat is 2 + 2?\n</user>\n<assistant>\n4\n</assistant>\n<eos>"


def test_teacher_response_does_not_duplicate_existing_assistant():
    parsed = _example_from_embedded_object(
        {
            "content": [
                {"role": "user", "content": "What is RSI?"},
                {"role": "assistant", "content": "A momentum oscillator."},
            ],
            "teacher_response": "A momentum oscillator.",
        },
        source="test",
    )
    assert parsed is not None
    assert [m.role for m in parsed.messages].count("assistant") == 1


def test_hh_conversation_serializes_canonically():
    parsed = _example_from_embedded_object(
        {
            "chosen": (
                "\n\nHuman: Hello"
                "\n\nAssistant: Hi."
                "\n\nHuman: Explain ATR."
                "\n\nAssistant: ATR measures volatility."
            )
        },
        source="test",
    )
    assert parsed is not None
    text = serialize_training_example(parsed)
    assert text.count("<user>") == 2
    assert text.count("<assistant>") == 2
    assert "Human:" not in text
    assert "Assistant:" not in text


def test_tokenizer_sample_uses_real_newlines():
    sample = build_tokenizer_training_sample(
        ["<bos>\n<user>one</user>\n<eos>", "<bos>\n<user>two</user>\n<eos>"],
        max_chars=1000,
    )
    assert "\n<sep>\n" in sample
    assert r"\n<sep>\n" not in sample


def test_serializer_has_single_canonical_grammar():
    example = TrainingExample(
        messages=(
            TrainingMessage("user", "What is spread?"),
            TrainingMessage("assistant", "The difference between bid and ask."),
        )
    )
    assert serialize_training_example(example) == (
        "<bos>\n<user>\nWhat is spread?\n</user>\n"
        "<assistant>\nThe difference between bid and ask.\n</assistant>\n<eos>"
    )


def test_structural_normalization_is_idempotent():
    raw = "<assistant><think>Calculate carefully.</think>4</assistant>"
    once = normalize_text(raw)
    twice = normalize_text(once)
    three = canonicalize_serialized(twice)
    assert once == twice == three
    assert once == (
        "<assistant>\n<think>\nCalculate carefully.\n</think>\n4\n</assistant>"
    )


def test_combined_prompt_does_not_leak_answer_into_user_turn():
    parsed = _parse_source_example(
        {
            "prompt": "Human: What are the three primary colors?\n\nAssistant: Red, blue and yellow.",
            "response": "Red, blue and yellow.",
        },
        source="test",
    )
    assert parsed is not None
    text = serialize_training_example(parsed)
    user_section = text.split("<user>", 1)[1].split("</user>", 1)[0]
    assert "Red, blue and yellow." not in user_section
    assert text.count("Red, blue and yellow.") == 1


def test_kimi_conversations_inside_embedded_prompt_are_recovered():
    wrapped = {
        "prompt": json.dumps(
            {
                "conversations": [
                    {"from": "human", "value": "Solve 2 + 2."},
                    {"from": "gpt", "value": "<think>Two plus two equals four.</think>4"},
                ],
                "output": "<think>Two plus two equals four.</think>4",
            }
        ),
        "response": "",
    }
    parsed = _parse_source_example(wrapped, source="kimi")
    assert parsed is not None
    text = serialize_training_example(parsed)
    assert text.count("<user>") == 1
    assert text.count("<assistant>") == 1
    assert text.count("<think>") == 1
    assert "<think>\nTwo plus two equals four.\n</think>\n4" in text


def test_teichai_serialized_message_lines_are_recovered():
    wrapped = {
        "prompt": (
            "{'role': 'system', 'content': ''}\n"
            "{'role': 'user', 'content': 'What is 2 + 2?'}\n"
            "{'role': 'assistant', 'content': '<think>Calculate.</think> 4'}"
        ),
        "response": "",
    }
    parsed = _parse_source_example(wrapped, source="teichai")
    assert parsed is not None
    assert [m.role for m in parsed.messages] == ["user", "assistant"]
    text = serialize_training_example(parsed)
    assert "<think>\nCalculate.\n</think>\n4" in text


def test_reasoning_tokens_are_canonicalized_to_boundaries():
    parsed = _parse_source_example(
        {
            "prompt": "Human: What is 2 + 2?\n\nAssistant: <think>Calculate carefully.</think>4",
            "response": "<think>Calculate carefully.</think>4",
        },
        source="test",
    )
    assert parsed is not None
    serialized = serialize_training_example(parsed)
    assert "<assistant>\n<think>\nCalculate carefully.\n</think>\n4\n</assistant>" in serialized
    assert "<assistant><think>" not in serialized


def test_serialization_is_idempotent_across_existing_structural_whitespace():
    example = TrainingExample(
        messages=(TrainingMessage("assistant", "<think>\nReason.\n</think>\nAnswer."),)
    )
    serialized = serialize_training_example(example)
    assert canonicalize_serialized(serialized) == serialized
