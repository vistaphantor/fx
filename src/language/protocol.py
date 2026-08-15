from __future__ import annotations

from collections.abc import Iterable

from src.language.tokenizer import BOS, BPETokenizer, SPECIAL_TOKENS


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
    """Serialize inference with the exact canonical grammar used in training.

    Canonical training examples always begin with <bos>. Inference previously
    omitted it while also tokenizing with add_bos=False, so a tiny model was
    evaluated under a prefix distribution it never saw during training. Keep a
    single explicit BOS here and never ask the tokenizer to add another one.
    """
    serialized: list[str] = [BOS]
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

    Exams exercise the exact same BOS/user/assistant grammar as interactive
    inference and canonical training. Keeping this here prevents evaluator and
    chat code from inventing a second prompt protocol.
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
