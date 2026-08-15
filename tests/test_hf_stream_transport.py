from __future__ import annotations

import sys
import types

from corpus.source import HFSource


class _FakeDataset:
    def __init__(self, rows, *, fail_after: int | None = None):
        self._rows = list(rows)
        self._fail_after = fail_after
        self.shuffle_called = False

    def shuffle(self, *args, **kwargs):  # pragma: no cover - must never be used
        self.shuffle_called = True
        raise AssertionError("remote IterableDataset.shuffle must not be called")

    def __iter__(self):
        for index, row in enumerate(self._rows):
            if self._fail_after is not None and index == self._fail_after:
                raise TimeoutError("synthetic transient timeout")
            yield row


def _rows(count: int) -> list[dict[str, str]]:
    return [
        {"text": f"document {index:02d} contains enough words to pass the minimum serialization length"}
        for index in range(count)
    ]


def test_hf_load_dataset_never_delegates_remote_shuffle(monkeypatch) -> None:
    dataset = _FakeDataset(_rows(5))
    calls: list[dict] = []

    def fake_load_dataset(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return dataset

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    source = HFSource(
        "example/corpus",
        revision="deadbeef",
        text_fields=["text"],
        shuffle_buffer_size=1000,
    )

    loaded = source._load_dataset(shuffle=True, retry_generation=99)

    assert loaded is dataset
    assert dataset.shuffle_called is False
    assert calls == [{
        "path": "example/corpus",
        "split": "train",
        "streaming": True,
        "revision": "deadbeef",
    }]


def test_local_reservoir_shuffle_is_deterministic(monkeypatch) -> None:
    rows = _rows(12)

    def fake_load_dataset(path, **kwargs):
        return _FakeDataset(rows)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    def collect() -> list[str]:
        return list(HFSource(
            "example/corpus",
            revision="deadbeef",
            text_fields=["text"],
            shuffle_buffer_size=4,
            seed=1234,
        ).stream())

    first = collect()
    second = collect()

    assert first == second
    assert len(first) == len(rows)
    assert set(first) == set(HFSource(
        "example/corpus",
        revision="deadbeef",
        text_fields=["text"],
        shuffle_buffer_size=0,
        seed=1234,
    ).stream())
    assert first != list(HFSource(
        "example/corpus",
        revision="deadbeef",
        text_fields=["text"],
        shuffle_buffer_size=0,
        seed=1234,
    ).stream())


def test_retry_reopens_same_sequential_stream_without_duplicate_outputs(monkeypatch) -> None:
    rows = _rows(10)
    loads = 0

    def fake_load_dataset(path, **kwargs):
        nonlocal loads
        loads += 1
        # First connection dies after raw row 4. The retry exposes the identical
        # sequential source and the HFSource resumes after its consumed boundary.
        return _FakeDataset(rows, fail_after=4 if loads == 1 else None)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    source = HFSource(
        "example/corpus",
        revision="deadbeef",
        text_fields=["text"],
        shuffle_buffer_size=3,
        seed=77,
        stream_retry_attempts=2,
        stream_retry_base_seconds=0,
    )

    output = list(source.stream())

    assert loads == 2
    assert len(output) == len(rows)
    assert len(set(output)) == len(rows)
    expected = {
        HFSource(
            "example/corpus",
            revision="deadbeef",
            text_fields=["text"],
        )._row_to_text(row)
        for row in rows
    }
    assert set(output) == expected


def test_transport_metadata_declares_sequential_remote_local_shuffle() -> None:
    source = HFSource(
        "example/corpus",
        revision="deadbeef",
        text_fields=["text"],
        shuffle_buffer_size=1000,
    )

    metadata = source.metadata()
    assert metadata["remote_shard_order"] == "sequential"
    assert metadata["shuffle_location"] == "local_reservoir"
    assert metadata["shuffle_buffer_size"] == 1000
