import pytest
import torch

from src.language.fx_sentiment_probe import (
    FXProbeAnswer,
    FX_SENTIMENT_PROMPTS,
    _resolve_device,
    render_probe,
)


def test_fx_probe_cases_cover_direction_causality_invalidation_and_ambiguity():
    prompts = dict(FX_SENTIMENT_PROMPTS)
    assert set(prompts) == {
        "eurusd_fed_hawkish",
        "eurusd_ecb_hawkish",
        "gbpusd_boe_dovish",
        "usdjpy_risk_off_ambiguity",
    }
    assert all("invalidat" in prompt.casefold() for prompt in prompts.values())
    assert all("do not invent live prices" in prompt.casefold() for prompt in prompts.values())
    assert "without pretending the direction is certain" in prompts["usdjpy_risk_off_ambiguity"]


def test_fx_probe_cpu_resolution_and_cuda_requirement(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_device("auto", require_cuda=False) == torch.device("cpu")
    with pytest.raises(RuntimeError, match="fx_probe_cuda_required"):
        _resolve_device("auto", require_cuda=True)
    with pytest.raises(RuntimeError, match="fx_probe_cuda_unavailable"):
        _resolve_device("cuda", require_cuda=False)


def test_render_probe_includes_checkpoint_and_exact_response():
    answer = FXProbeAnswer(
        case_id="case",
        prompt="prompt",
        response="bearish because rates rose",
        generated_tokens=5,
        replacement_characters=0,
    )
    rendered = render_probe(
        (answer,),
        {
            "epoch": 9,
            "cumulative_prediction_tokens": 123,
            "exam_correctness_percent": 42.0,
            "exam_gibberish_answers": 0,
        },
    )
    assert "checkpoint_epoch=9" in rendered
    assert "checkpoint_tokens=123" in rendered
    assert "RESPONSE: bearish because rates rose" in rendered
