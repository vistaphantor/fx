from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"
_CANONICAL_LABELS = ("positive", "negative", "neutral")


@dataclass(frozen=True, slots=True)
class FinBertSentimentResult:
    """Structured financial sentiment prediction from FinBERT."""

    text: str
    label: str
    sentiment_score: float
    confidence: float
    scores: dict[str, float]


def resolve_torch_device(value: str = "auto", *, require_cuda: bool = False) -> torch.device:
    requested = value.strip().casefold()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("finbert_cuda_required")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("finbert_cuda_unavailable")
    return device


class FinBertSentimentAnalyzer:
    """Wrapper around ProsusAI/finbert for FX news and headline sentiment scoring."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        model_name: str = DEFAULT_FINBERT_MODEL,
        max_length: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.max_length = max_length
        self._id2label = _resolve_id2label(getattr(getattr(model, "config", None), "id2label", None))

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_FINBERT_MODEL,
        *,
        device: str = "auto",
        require_cuda: bool = False,
        local_files_only: bool = False,
        max_length: int = 512,
    ) -> FinBertSentimentAnalyzer:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "FinBERT requires the 'transformers' package declared in requirements.txt"
            ) from exc

        resolved_device = resolve_torch_device(device, require_cuda=require_cuda)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(resolved_device)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            model_name=model_name,
            max_length=max_length,
        )

    @torch.no_grad()
    def score_batch(
        self, texts: Sequence[str], *, batch_size: int = 16
    ) -> tuple[FinBertSentimentResult, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        cleaned = [str(text).strip() for text in texts]
        if any(not item for item in cleaned):
            raise ValueError("texts must not contain empty items")
        if not cleaned:
            return ()

        results: list[FinBertSentimentResult] = []
        for start in range(0, len(cleaned), batch_size):
            chunk = cleaned[start : start + batch_size]
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in encoded.items()
            }
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            for text, row_probs in zip(chunk, probs.tolist()):
                scores = {label: 0.0 for label in _CANONICAL_LABELS}
                for idx, prob in enumerate(row_probs):
                    label = self._id2label.get(idx, f"label_{idx}")
                    scores[label] = float(prob)
                top_label = max(scores.items(), key=lambda item: item[1])[0]
                sentiment_score = float(scores.get("positive", 0.0) - scores.get("negative", 0.0))
                results.append(
                    FinBertSentimentResult(
                        text=text,
                        label=top_label,
                        sentiment_score=sentiment_score,
                        confidence=float(scores[top_label]),
                        scores=scores,
                    )
                )
        return tuple(results)

    def score_text(self, text: str) -> FinBertSentimentResult:
        return self.score_batch([text], batch_size=1)[0]


def _resolve_id2label(raw: dict[int | str, str] | None) -> dict[int, str]:
    if not raw:
        return {idx: label for idx, label in enumerate(_CANONICAL_LABELS)}
    resolved: dict[int, str] = {}
    for key, value in raw.items():
        resolved[int(key)] = str(value).strip().casefold()
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score FX news/text sentiment with FinBERT.")
    parser.add_argument("--model-name", default=DEFAULT_FINBERT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--text", action="append", default=[])
    args = parser.parse_args(argv)

    texts = args.text or [
        "The Federal Reserve signals interest rates may stay high for longer after stronger US payrolls.",
        "The Bank of England cuts rates and lowers its UK growth forecast.",
    ]
    analyzer = FinBertSentimentAnalyzer.from_pretrained(
        args.model_name,
        device=args.device,
        require_cuda=bool(args.require_cuda),
        local_files_only=bool(args.local_files_only),
    )
    results = analyzer.score_batch(texts)
    print(
        json.dumps(
            {
                "model": args.model_name,
                "device": str(analyzer.device),
                "results": [asdict(item) for item in results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
