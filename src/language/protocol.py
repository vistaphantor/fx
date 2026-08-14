from __future__ import annotations

from collections.abc import Iterable

from src.language.tokenizer import BPETokenizer, SPECIAL_TOKENS


class ProtocolError(ValueError):
    pass


def _validate_content(text: str) -> str:
    value = str(text).strip()
    if not value:
        raise ProtocolError("message_content_empty")
    embedded = [token for token in SPECIAL_TOKENS if token in value]
    if embedded:
        raise ProtocolError(f"message_contains_reserved_control_token:{embedded[0]}")
    return value


def format_user_turn(text: str) -> str:
    return f"<user>\n{_validate_content(text)}\n</user>"


def format_assistant_turn(text: str) -> str:
    return f"<assistant>\n{_validate_content(text)}\n</assistant>"


def assistant_prefix() -> str:
    return "<assistant>\n"


def build_chat_prompt(turns: Iterable[tuple[str, str]]) -> str:
    serialized: list[str] = []
    for role, text in turns:
        if role == "user":
            serialized.append(format_user_turn(text))
        elif role == "assistant":
            serialized.append(format_assistant_turn(text))
        else:
            raise ProtocolError(f"unsupported_chat_role:{role}")
    serialized.append(assistant_prefix().rstrip("\n"))
    return "\n".join(serialized) + "\n"


def build_exam_prompt(question: str) -> str:
    """Build the deterministic single-turn prompt used by epoch exams.

    Exams must exercise the exact same user/assistant grammar as interactive
    inference. Keeping this here prevents the evaluator from inventing a
    second prompt protocol that the model never saw during training.
    """
    return build_chat_prompt((("user", question),))


def generation_stop_ids(tokenizer: BPETokenizer) -> set[int]:
    return {
        tokenizer.eos_id(),
        tokenizer.vocab["</assistant>"],
    }


def extract_assistant_response(decoded: str) -> str:
    marker = "<assistant>"
    if marker not in decoded:
        raise ProtocolError("assistant_marker_missing_from_generation")
    response = decoded.rsplit(marker, 1)[1]
    for terminator in ("</assistant>", "<eos>", "<user>"):
        if terminator in response:
            response = response.split(terminator, 1)[0]
    return response.strip()
