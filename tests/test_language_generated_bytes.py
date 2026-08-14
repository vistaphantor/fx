from __future__ import annotations

import pytest
import torch

from src.language.exam import run_epoch_exam
from src.language.tokenizer import BPETokenizer


def _tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    tok.train("plain ascii training text", vocab_size=512, min_frequency=1)
    return tok


def test_strict_decode_still_rejects_invalid_utf8() -> None:
    tok = _tokenizer()
    invalid_id = tok.vocab["<byte:ff>"]
    with pytest.raises(UnicodeDecodeError):
        tok.decode([invalid_id], skip_special=False)


def test_generation_decode_replaces_invalid_utf8() -> None:
    tok = _tokenizer()
    invalid_id = tok.vocab["<byte:ff>"]
    assert tok.decode([invalid_id], skip_special=False, errors="replace") == "\ufffd"


class _MalformedByteModel:
    max_seq_len = 256

    def __init__(self, invalid_id: int):
        self.invalid_id = invalid_id

    def eval(self):
        return self

    def generate(self, idx: torch.Tensor, **_: object) -> torch.Tensor:
        continuation = torch.tensor([[self.invalid_id]], dtype=torch.long)
        return torch.cat((idx, continuation), dim=1)


def test_epoch_zero_exam_observes_malformed_bytes_without_crashing() -> None:
    tok = _tokenizer()
    model = _MalformedByteModel(tok.vocab["<byte:ff>"])
    result = run_epoch_exam(
        model=model,  # type: ignore[arg-type]
        tokenizer=tok,
        epoch=0,
        training_stage="foundation",
        train_loss=None,
        validation_loss=None,
        max_new_tokens=1,
    )
    assert result.total_questions > 0
    assert all("invalid_utf8_bytes" in answer.gibberish_flags for answer in result.answers)
