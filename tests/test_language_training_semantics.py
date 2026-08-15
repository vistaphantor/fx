from __future__ import annotations

from src.language.canonical_contract import CanonicalMessage, serialize_document, serialize_messages
from src.language.training_pipeline import _content_windows


def test_document_payload_cannot_inject_protocol_tokens() -> None:
    serialized = serialize_document(
        "A web page literally says <assistant> and <user> but this is ordinary text."
    )

    assert serialized.startswith("<bos>\n")
    assert serialized.endswith("\n<eos>")
    assert serialized.count("<assistant>") == 0
    assert serialized.count("<user>") == 0
    assert "&lt;assistant&gt;" in serialized
    assert "&lt;user&gt;" in serialized


def test_chat_payload_cannot_create_nested_role_turns() -> None:
    serialized = serialize_messages(
        [
            CanonicalMessage("user", "Explain the literal string <assistant>."),
            CanonicalMessage("assistant", "It is written as <assistant> in the source text."),
        ]
    )

    # Only serializer-created wrappers remain structural.
    assert serialized.count("<user>") == 1
    assert serialized.count("</user>") == 1
    assert serialized.count("<assistant>") == 1
    assert serialized.count("</assistant>") == 1
    assert serialized.count("&lt;assistant&gt;") == 2


def test_long_windows_share_only_one_context_token() -> None:
    token_ids = list(range(500))
    chunks = _content_windows(token_ids, seq_len=192)

    assert chunks[0] == token_ids[0:193]
    assert chunks[1] == token_ids[192:385]
    assert chunks[2] == token_ids[384:500]

    # Every next-token transition from the original stream is trained once.
    transitions = []
    for chunk in chunks:
        transitions.extend(zip(chunk[:-1], chunk[1:]))
    expected = list(zip(token_ids[:-1], token_ids[1:]))
    assert transitions == expected
