from pathlib import Path

import pytest

from src.language.model_bundle import BUNDLE_VERSION, load_model_bundle, save_model_bundle
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer, TOKENIZER_ALGORITHM_VERSION


def _tiny_tokenizer():
    text = (
        "<user>What is RSI?</user> "
        "<assistant>RSI is a momentum oscillator.</assistant> "
        "<market>XAUUSD M15 bullish</market> "
        "<decision>WAIT</decision>"
    )
    tok = BPETokenizer()
    tok.train(text, vocab_size=512, min_frequency=1)
    return tok


def test_special_tokens_are_atomic():
    tok = _tiny_tokenizer()
    ids = tok.encode(
        "<user>Hello</user><decision>WAIT</decision>",
        add_bos=False,
        add_eos=False,
    )
    assert tok.vocab["<user>"] in ids
    assert tok.vocab["</user>"] in ids
    assert tok.vocab["<decision>"] in ids
    assert tok.vocab["</decision>"] in ids


def test_model_bundle_round_trip(tmp_path: Path):
    tok = _tiny_tokenizer()
    cfg = {
        "vocab_size": tok.vocab_size,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "ffn_dim": 128,
        "max_seq_len": 64,
        "dropout": 0.0,
    }
    model = VistaReasoningGPT(**cfg)
    bundle = tmp_path / "bundle"
    save_model_bundle(
        bundle_dir=bundle,
        model=model,
        tokenizer=tok,
        model_config=cfg,
        training_stage="smoke",
        corpus_fingerprint="test-corpus",
        metrics={"loss": 1.0},
    )

    loaded_model, loaded_tok, manifest = load_model_bundle(bundle)
    assert manifest.bundle_version == BUNDLE_VERSION == 2
    assert manifest.tokenizer_algorithm_version == TOKENIZER_ALGORITHM_VERSION == 4
    assert loaded_tok.fingerprint() == tok.fingerprint()
    assert loaded_model.get_num_params() == model.get_num_params()
    assert manifest.training_stage == "smoke"


def test_bundle_refuses_tokenizer_tamper(tmp_path: Path):
    tok = _tiny_tokenizer()
    cfg = {
        "vocab_size": tok.vocab_size,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "ffn_dim": 128,
        "max_seq_len": 64,
        "dropout": 0.0,
    }
    model = VistaReasoningGPT(**cfg)
    bundle = tmp_path / "bundle"
    save_model_bundle(
        bundle_dir=bundle,
        model=model,
        tokenizer=tok,
        model_config=cfg,
        training_stage="smoke",
        corpus_fingerprint="test-corpus",
    )

    tokenizer_path = bundle / "tokenizer.json"
    tokenizer_path.write_text(
        tokenizer_path.read_text(encoding="utf-8") + "\n ",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="tokenizer_sha256_mismatch"):
        load_model_bundle(bundle)
