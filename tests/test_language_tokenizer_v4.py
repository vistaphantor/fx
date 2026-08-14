from __future__ import annotations

from pathlib import Path

import pytest

from src.language.tokenizer import BPETokenizer, TOKENIZER_ALGORITHM_VERSION


def _tokenizer() -> BPETokenizer:
    corpus = (
        "<user>What is RSI?</user>\n"
        "<assistant><think>Calculate carefully.</think>"
        "RSI is a momentum oscillator.</assistant>\n"
        "XAUUSD EURUSD KES market bullish bearish risk ATR RSI"
    )
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=512, min_frequency=1)
    return tok


@pytest.mark.parametrize(
    "text",
    [
        "The market is bullish.",
        "one  two   three",
        "one\ntwo\n\nthree",
        "<user>What is RSI?</user>",
        "<assistant><think>Calculate carefully.</think>RSI is a momentum oscillator.</assistant>",
        "<user>\nWhat is RSI?\n</user>",
        "Walae Mkuu Mtaji",
        "KES 50,000",
        "XAUUSD @ 2,431.75",
        "€ £ ¥ KSh",
        "α β Σ ∂",
        "你好",
        "مرحبا",
        r"\mathrm{ATR} \frac{1}{2}",
    ],
)
def test_any_utf8_text_round_trips_exactly(text: str) -> None:
    tok = _tokenizer()
    ids = tok.encode(text, add_bos=False, add_eos=False)
    assert tok.decode(ids, skip_special=False) == text
    assert tok.unk_id() not in ids


def test_control_tokens_are_atomic() -> None:
    tok = _tokenizer()
    ids = tok.encode("<user>x</user>", add_bos=False, add_eos=False)
    assert ids[0] == tok.vocab["<user>"]
    assert ids[-1] == tok.vocab["</user>"]


def test_saved_tokenizer_is_version_bound(tmp_path: Path) -> None:
    tok = _tokenizer()
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.algorithm_version == TOKENIZER_ALGORITHM_VERSION == 4
    assert loaded.fingerprint() == tok.fingerprint()
    text = "unseen UTF-8: Mkuu € α 你好 مرحبا"
    assert loaded.decode(loaded.encode(text, False, False), False) == text


def test_v3_and_legacy_tokenizers_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        '{"algorithm_version":3,"vocab":{"<pad>":0},"merges":[]}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="unsupported_tokenizer_algorithm_version:3"):
        BPETokenizer.load(path)


def test_base_vocabulary_requires_all_bytes() -> None:
    tok = BPETokenizer()
    tok.train("tiny", vocab_size=512, min_frequency=1)
    for value in range(256):
        assert f"<byte:{value:02x}>" in tok.vocab
