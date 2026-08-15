from __future__ import annotations

import pytest

from corpus.source import HFSource


def test_language_trainer_imports_after_stream_only_cutover() -> None:
    import tools.train_language_reasoner as trainer

    assert trainer.TRAINER_VERSION >= 8


def test_hf_stream_reopens_after_transient_transport_failure() -> None:
    source = HFSource(
        path="example/dataset",
        revision="deadbeef",
        text_fields=["text"],
        shuffle_buffer_size=0,
        stream_retry_attempts=2,
        stream_retry_base_seconds=0.0,
    )
    calls: list[int] = []

    class BrokenOnce:
        def __iter__(self):
            raise TimeoutError("read operation timed out")
            yield  # pragma: no cover

    class Healthy:
        def __iter__(self):
            yield {"text": "The first healthy streamed English sentence."}
            yield {"text": "The second healthy streamed English sentence."}

    def fake_load_dataset(*, shuffle: bool, retry_generation: int = 0):
        assert shuffle is True
        calls.append(retry_generation)
        return BrokenOnce() if retry_generation == 0 else Healthy()

    source._load_dataset = fake_load_dataset  # type: ignore[method-assign]

    rows = list(source.stream())

    assert len(rows) == 2
    assert calls == [0, 1]
    assert "first healthy" in rows[0]
    assert "second healthy" in rows[1]


def test_hf_stream_does_not_hide_non_transport_failures() -> None:
    source = HFSource(
        path="example/dataset",
        revision="deadbeef",
        text_fields=["text"],
        shuffle_buffer_size=0,
        stream_retry_attempts=5,
        stream_retry_base_seconds=0.0,
    )

    class InvalidSchemaStream:
        def __iter__(self):
            raise ValueError("invalid dataset schema")
            yield  # pragma: no cover

    def fake_load_dataset(*, shuffle: bool, retry_generation: int = 0):
        return InvalidSchemaStream()

    source._load_dataset = fake_load_dataset  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="invalid dataset schema"):
        list(source.stream())


def test_hf_source_rejects_invalid_retry_configuration() -> None:
    with pytest.raises(ValueError, match="stream_retry_attempts"):
        HFSource(
            path="example/dataset",
            revision="deadbeef",
            stream_retry_attempts=-1,
        )
