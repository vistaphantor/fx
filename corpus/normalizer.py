from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator


_THINK_OPEN  = "___THINK_OPEN___"
_THINK_CLOSE = "___THINK_CLOSE___"


def _clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace("<think>", _THINK_OPEN).replace("</think>", _THINK_CLOSE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace(_THINK_OPEN, "<think>").replace(_THINK_CLOSE, "</think>")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class TextNormalizer:
    def parse_file(self, path: Path) -> Iterator[str]:
        suffix = path.suffix.lower()
        try:
            if suffix == ".jsonl":
                yield from self._parse_jsonl(path)
            elif suffix == ".json":
                yield from self._parse_json(path)
            elif suffix == ".txt":
                yield from self._parse_txt(path)
        except Exception:
            return

    def _parse_jsonl(self, path: Path) -> Iterator[str]:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    text = self._row_to_text(row)
                    if text:
                        yield text
                except Exception:
                    continue

    def _parse_json(self, path: Path) -> Iterator[str]:
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return

        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = (
                data.get("examples")
                or data.get("data")
                or data.get("train")
                or (data if isinstance(data, list) else [])
            )
            if isinstance(rows, dict):
                rows = [rows]
        else:
            return

        for row in rows:
            if not isinstance(row, dict):
                continue
            text = self._row_to_text(row)
            if text:
                yield text

    def _parse_txt(self, path: Path) -> Iterator[str]:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) > 25:
            yield _clean(text)

    def _row_to_text(self, row: dict) -> str | None:
        prompt_str = row.get("prompt", "")

        if isinstance(prompt_str, str) and prompt_str.startswith("{"):
            try:
                obj = json.loads(prompt_str)
                return self._parse_nested(obj)
            except Exception:
                pass

        for fmt_fn in (
            self._try_hh_rlhf,
            self._try_teacher_response,
            self._try_alpaca,
            self._try_prompt_response,
            self._try_conversations,
            self._try_raw_text,
        ):
            result = fmt_fn(row)
            if result:
                return result

        return None

    def _parse_nested(self, obj: dict) -> str | None:
        if "chosen" in obj:
            raw = obj["chosen"]
            turns = re.split(r"\n\n(Human:|Assistant:)", raw)
            parts, i = [], 0
            while i < len(turns):
                chunk = turns[i].strip()
                if chunk in ("Human:", "Assistant:"):
                    role = "Human" if "Human" in chunk else "Assistant"
                    content = turns[i + 1].strip() if i + 1 < len(turns) else ""
                    parts.append(f"{role}: {content}")
                    i += 2
                elif chunk:
                    parts.append(chunk)
                    i += 1
                else:
                    i += 1
            text = "\n\n".join(parts)
            return _clean(text) if text else None

        if "content" in obj or "teacher_response" in obj:
            parts = []
            for item in obj.get("content") or []:
                role = item.get("role", "")
                val  = item.get("content", "").strip()
                if role == "user":
                    parts.append(f"Human: {val}")
                elif role == "assistant":
                    parts.append(f"Assistant: {val}")
            tr = obj.get("teacher_response", "").strip()
            if tr:
                parts.append(f"Assistant: {tr}")
            text = "\n\n".join(parts)
            return _clean(text) if text else None

        return None

    def _try_hh_rlhf(self, row: dict) -> str | None:
        chosen = row.get("chosen", "")
        if not chosen:
            return None
        turns = re.split(r"\n\n(Human:|Assistant:)", str(chosen))
        parts, i = [], 0
        while i < len(turns):
            chunk = turns[i].strip()
            if chunk in ("Human:", "Assistant:"):
                role = "Human" if "Human" in chunk else "Assistant"
                content = turns[i + 1].strip() if i + 1 < len(turns) else ""
                parts.append(f"{role}: {content}")
                i += 2
            elif chunk:
                parts.append(chunk)
                i += 1
            else:
                i += 1
        text = "\n\n".join(parts)
        return _clean(text) if len(text) > 25 else None

    def _try_teacher_response(self, row: dict) -> str | None:
        if "teacher_response" not in row and "content" not in row:
            return None
        parts = []
        for item in row.get("content") or []:
            role = item.get("role", "")
            val  = item.get("content", "").strip()
            if role == "user":
                parts.append(f"Human: {val}")
            elif role == "assistant":
                parts.append(f"Assistant: {val}")
        tr = row.get("teacher_response", "").strip()
        if tr:
            parts.append(f"Assistant: <think>\n{tr}\n</think>")
        text = "\n\n".join(parts)
        return _clean(text) if len(text) > 25 else None

    def _try_alpaca(self, row: dict) -> str | None:
        inst = (row.get("instruction") or "").strip()
        inp  = (row.get("input") or "").strip()
        out  = (row.get("output") or row.get("response") or "").strip()
        if not inst or not out:
            return None
        p = f"{inst}\n{inp}".strip() if inp else inst
        return _clean(f"Human: {p}\n\nAssistant: <think>\n{out}\n</think>")

    def _try_prompt_response(self, row: dict) -> str | None:
        p = (row.get("prompt") or row.get("text") or "").strip()
        r = (row.get("response") or row.get("answer") or row.get("completion") or "").strip()
        if p and r:
            return _clean(f"Human: {p}\n\nAssistant: <think>\n{r}\n</think>")
        if p and len(p) > 25:
            return _clean(p)
        return None

    def _try_conversations(self, row: dict) -> str | None:
        convs = row.get("conversations") or row.get("messages")
        if not isinstance(convs, list):
            return None
        parts = []
        for t in convs:
            role = t.get("from", t.get("role", "")).lower()
            val  = t.get("value", t.get("content", "")).strip()
            if role in ("human", "user"):
                parts.append(f"Human: {val}")
            elif role in ("gpt", "assistant"):
                parts.append(f"Assistant: {val}")
        text = "\n\n".join(parts)
        return _clean(text) if len(text) > 25 else None

    def _try_raw_text(self, row: dict) -> str | None:
        text = (row.get("text") or row.get("content") or "").strip()
        return _clean(text) if len(text) > 25 else None
