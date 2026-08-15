from __future__ import annotations

from src.language.source_mix_audit import SourceTokenStat, format_supervised_source_mix


def test_gradient_mix_reports_supervised_tokens_not_keyword_counts() -> None:
    rows = (
        SourceTokenStat("TinyStories", 10, 700, 70.0),
        SourceTokenStat("PrimitiveArithmetic", 20, 300, 30.0),
    )
    rendered = format_supervised_source_mix(rows)
    assert rendered.startswith("[GradientMix]")
    assert "TinyStories=70.0%(700t/10e)" in rendered
    assert "PrimitiveArithmetic=30.0%(300t/20e)" in rendered
    assert "math=" not in rendered
