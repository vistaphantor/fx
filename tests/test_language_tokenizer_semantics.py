import pytest

from src.language.tokenizer import BPETokenizer, TOKENIZER_ALGORITHM_VERSION


def _tokenizer():
    tok = BPETokenizer()
    tok.train(
        (
            "<user>Hello world</user> "
            "<assistant>Hello trader</assistant> "
            "bullish bearish market price risk"
        ),
        vocab_size=512,
        min_frequency=1,
    )
    return tok


def test_encode_decode_preserves_exact_spacing():
    tok = _tokenizer()
    text = "Hello  world\ntrader"
    ids = tok.encode(text, add_bos=False, add_eos=False)
    assert tok.decode(ids, skip_special=False) == text


def test_unseen_utf8_never_requires_unknown_token():
    tok = _tokenizer()
    text = "Walae Mkuu Mtaji — KES 50,000 — α β Σ ∂ — 你好 — مرحبا"
    ids = tok.encode(text, add_bos=False, add_eos=False)
    assert tok.unk_id() not in ids
    assert tok.decode(ids, skip_special=False) == text


def test_tokenizer_algorithm_version_is_saved(tmp_path):
    tok = _tokenizer()
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.algorithm_version == tok.algorithm_version == TOKENIZER_ALGORITHM_VERSION == 4
    assert loaded.fingerprint() == tok.fingerprint()


def test_legacy_tokenizer_is_rejected(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(
        '{"algorithm_version":3,"vocab":{"<pad>":0},"merges":[]}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="unsupported_tokenizer_algorithm_version:3"):
        BPETokenizer.load(path)
