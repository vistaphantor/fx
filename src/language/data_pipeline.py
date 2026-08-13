"""
Data pipeline — loads all datasets from data/data/trainingdata and
converts them into a flat stream of text suitable for language model training.

Handles all discovered formats (including truncated JSON strings):
  - Anthropic hh-rlhf  (chosen/rejected conversations)
  - TeichAI reasoning  (<think>...</think> + response)
  - Turing Open Reasoning
  - LMSYS Chat (JSON string with content list + teacher_response)
  - Sample training data
  - Plain .txt files
"""
from __future__ import annotations

import json
import re
import random
from pathlib import Path
from typing import Iterator


DATA_ROOT = Path("data/data/trainingdata")

# ── Format parsers ────────────────────────────────────────────────────────────

def _parse_prompt_json(prompt_str: str) -> str | None:
    """Attempt to parse a prompt string that contains JSON (or truncated JSON)."""
    obj = None
    try:
        obj = json.loads(prompt_str)
    except Exception:
        # Try fixing truncated json string
        try:
            # Append closing quotes and braces
            fixed = prompt_str.strip()
            if not fixed.endswith("}"):
                if not fixed.endswith('"'):
                    fixed += '"'
                fixed += "}"
            obj = json.loads(fixed)
        except Exception:
            pass

    if isinstance(obj, dict):
        # Format 1: Anthropic hh-rlhf chosen / rejected
        if "chosen" in obj or "rejected" in obj:
            chosen = obj.get("chosen", "") or obj.get("rejected", "")
            turns = re.split(r"\n\n(Human:|Assistant:)", chosen)
            parts = []
            i = 0
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
            return "\n\n".join(parts) if parts else None

        # Format 2: LMSYS / OpenAI message list + teacher_response
        if "content" in obj or "teacher_response" in obj:
            parts = []
            contents = obj.get("content", [])
            if isinstance(contents, list):
                for item in contents:
                    if isinstance(item, dict):
                        role = item.get("role", "")
                        text = item.get("content", "").strip()
                        if role == "user":
                            text = re.sub(r"^Below is an instruction that describes a task\.\s*Write a response that appropriately completes the request\.\s*### Instruction:\s*", "", text, flags=re.IGNORECASE)
                            text = re.sub(r"\s*### Response:\s*$", "", text, flags=re.IGNORECASE)
                            parts.append(f"Human: {text}")
                        elif role == "assistant":
                            parts.append(f"Assistant: {text}")
            
            teacher_resp = obj.get("teacher_response", "").strip()
            if teacher_resp:
                parts.append(f"Assistant: {teacher_resp}")

            if parts:
                return "\n\n".join(parts)

    # Regex extraction fallback if JSON parsing completely failed due to truncation
    user_match = re.search(r'"role":\s*"user",?\s*"content":\s*"(.*?)"(?:\}|,|\s*"role")', prompt_str, re.DOTALL)
    teacher_match = re.search(r'"teacher_response":\s*"(.*)', prompt_str, re.DOTALL)

    parts = []
    if user_match:
        u_text = user_match.group(1).replace("\\n", "\n").replace('\\"', '"').strip()
        u_text = re.sub(r"^Below is an instruction that describes a task\.\s*Write a response that appropriately completes the request\.\s*### Instruction:\s*", "", u_text, flags=re.IGNORECASE)
        u_text = re.sub(r"\s*### Response:\s*$", "", u_text, flags=re.IGNORECASE)
        parts.append(f"Human: {u_text}")

    if teacher_match:
        t_text = teacher_match.group(1).replace("\\n", "\n").replace('\\"', '"').strip()
        t_text = re.sub(r'"\s*\}?$', '', t_text).strip()
        parts.append(f"Assistant: {t_text}")

    if parts:
        return "\n\n".join(parts)

    return None


def _parse_teichai(prompt_str: str) -> str | None:
    """Parse TeichAI format: {'role':..., 'content':...} lines."""
    try:
        match = re.search(r"'role':\s*'assistant',\s*'content':\s*'(.*)'", prompt_str, re.DOTALL)
        if not match:
            return None
        content = match.group(1)
        content = content.encode("utf-8").decode("unicode_escape", errors="replace")
        content = content.replace("\\n", "\n").replace("\\'", "'")

        user_match = re.search(r"'role':\s*'user',\s*'content':\s*'([^']*)'", prompt_str)
        user_q = user_match.group(1).strip() if user_match else ""

        result = ""
        if user_q:
            result += f"Human: {user_q}\n\n"
        result += f"Assistant: {content}"
        return result.strip()
    except Exception:
        return None


def _parse_generic(obj: dict) -> str | None:
    """Fallback parser for any {prompt, response} or {text} format."""
    if "text" in obj:
        return str(obj["text"]).strip()
    if "prompt" in obj and "response" in obj:
        p = str(obj.get("prompt", "")).strip()
        r = str(obj.get("response", "")).strip()
        if p and r:
            return f"Human: {p}\n\nAssistant: {r}"
        return p or r or None
    return None


# ── Dataset loader ─────────────────────────────────────────────────────────────

def _load_json_file(path: Path) -> Iterator[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        print(f"  [DataLoader] Skip {path.name}: {e}")
        return

    examples = data.get("examples") or data.get("data") or []
    if not examples and isinstance(data, list):
        examples = data

    for ex in examples:
        if not isinstance(ex, dict):
            continue
        prompt_str = ex.get("prompt", "")

        text = None
        if isinstance(prompt_str, str) and prompt_str.startswith("{"):
            text = _parse_prompt_json(prompt_str)
        
        if not text and isinstance(prompt_str, str) and "'role':" in prompt_str:
            text = _parse_teichai(prompt_str)

        if not text:
            text = _parse_generic(ex)

        if text and len(text.strip()) > 20:
            yield text.strip()


def _load_txt_file(path: Path) -> Iterator[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        for para in re.split(r"\n{2,}", text):
            para = para.strip()
            if len(para) > 30:
                yield para
    except Exception:
        return


def load_all_training_text(
    data_root: Path = DATA_ROOT,
    max_examples: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[str]:
    """
    Load and return all training texts from data/data/trainingdata.
    Returns a list of strings, one per conversation/document.
    """
    all_texts: list[str] = []
    rng = random.Random(seed)

    json_files = sorted(data_root.glob("*.json"))
    txt_files  = sorted(data_root.glob("*.txt"))

    print(f"[DataLoader] Found {len(json_files)} root JSON files and {len(txt_files)} TXT files in {data_root}")

    for path in json_files:
        if "master_index" in path.name:
            continue
        count_before = len(all_texts)
        for text in _load_json_file(path):
            all_texts.append(text)
        print(f"  {path.name}: +{len(all_texts) - count_before} examples")

    for subdir in sorted(data_root.iterdir()):
        if not subdir.is_dir():
            continue
        for path in sorted(subdir.glob("*.json")):
            count_before = len(all_texts)
            for text in _load_json_file(path):
                all_texts.append(text)
            n = len(all_texts) - count_before
            if n > 0:
                print(f"  {subdir.name}/{path.name}: +{n} examples")

    for path in txt_files:
        for text in _load_txt_file(path):
            all_texts.append(text)

    print(f"[DataLoader] Total raw parsed examples: {len(all_texts):,}")

    if shuffle:
        rng.shuffle(all_texts)

    if max_examples:
        all_texts = all_texts[:max_examples]

    return all_texts


def build_corpus_string(texts: list[str], sep: str = "\n\n<sep>\n\n") -> str:
    return sep.join(texts)


def make_batches(
    token_ids: list[int],
    seq_len: int = 256,
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[list[list[int]]]:
    rng = random.Random(seed)
    stride = seq_len

    windows = []
    for start in range(0, len(token_ids) - seq_len - 1, stride):
        windows.append(token_ids[start : start + seq_len + 1])

    if shuffle:
        rng.shuffle(windows)

    batch = []
    for window in windows:
        batch.append(window)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
