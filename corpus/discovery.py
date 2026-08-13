from __future__ import annotations

from pathlib import Path
from typing import Iterator

from corpus.source import LocalSource


class DatasetDiscovery:
    SKIP_NAMES = {
        "master_index.json", "registry.json", "__pycache__",
        ".DS_Store", "Thumbs.db",
    }
    VALID_EXTS = {".json", ".jsonl", ".txt", ".gz"}

    def __init__(self, roots: list[str | Path]):
        self._roots = [Path(r) for r in roots]

    def discover(self) -> list[LocalSource]:
        sources: list[LocalSource] = []
        for root in self._roots:
            if not root.exists():
                print(f"[Discovery] Root not found, skipping: {root}")
                continue
            if root.is_file():
                sources.append(LocalSource(root))
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if path.name in self.SKIP_NAMES:
                    continue
                if path.suffix.lower() not in self.VALID_EXTS:
                    continue
                if any(part.startswith(".") or part == "__pycache__"
                       for part in path.parts):
                    continue
                sources.append(LocalSource(path))
        return sources

    def iter_texts(self, sources: list[LocalSource]) -> Iterator[tuple[str, str]]:
        for src in sources:
            for text in src.stream():
                yield str(src._path), text
