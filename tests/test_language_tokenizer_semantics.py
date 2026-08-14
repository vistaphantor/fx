import pytest

from src.language.tokenizer import BPETokenizer


def _tokenizer():
    tok = BPETokenizer()

    tok.train(
        (
            "<user>Hello world</user> "
            "<assistant>Hello trader</assistant> "
            "bullish bearish market price risk"
        ),
        vocab_size=256,
        min_frequency=1,
    )

    return tok


def test_word_boundary_is_not_encoded_twice():
    tok = _tokenizer()

    ids = tok.encode(
        "Hello world",
        add_bos=False,
        add_eos=False,
    )

    decoded = tok.decode(
        ids,
        skip_special=False,
    )

    assert "Hello  world" not in decoded
    assert "Hello world" in decoded


def test_encode_decode_preserves_readable_word_spacing():
    tok = _tokenizer()

    text = "bullish market risk"

    ids = tok.encode(
        text,
        add_bos=False,
        add_eos=False,
    )

    decoded = tok.decode(ids)

    assert decoded == text


def test_tokenizer_algorithm_version_is_saved(tmp_path):
    tok = _tokenizer()

    path = tmp_path / "tokenizer.json"
    tok.save(path)

    loaded = BPETokenizer.load(path)

    assert loaded.algorithm_version == tok.algorithm_version
    assert loaded.fingerprint() == tok.fingerprint()


def test_legacy_tokenizer_is_rejected(tmp_path):
    path = tmp_path / "tokenizer.json"

    path.write_text(
        '{"vocab":{"<pad>":0},"merges":[]}',
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="unsupported_tokenizer_algorithm_version",
    ):
        BPETokenizer.load(path)
