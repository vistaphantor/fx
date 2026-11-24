from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.language.finbert_sentiment import DEFAULT_FINBERT_MODEL, FinBertSentimentAnalyzer
from src.language.protocol import (
    build_chat_prompt,
    extract_assistant_response,
    generation_stop_ids,
)
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer


FX_SENTIMENT_PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "eurusd_fed_hawkish",
        "A news headline says: The Federal Reserve signals interest rates may stay high "
        "for longer after stronger US payrolls. Analyze the likely near-term EUR/USD "
        "sentiment. State the direction, the causal reason, and what would invalidate it. "
        "Do not invent live prices or claim certainty.",
    ),
    (
        "eurusd_ecb_hawkish",
        "A news headline says: The ECB unexpectedly raises interest rates because euro-area "
        "inflation accelerated. Analyze the likely near-term EUR/USD sentiment. State the "
        "direction, the causal reason, and what would invalidate it. Do not invent live prices.",
    ),
    (
        "gbpusd_boe_dovish",
        "A news headline says: The Bank of England cuts rates and lowers its UK growth forecast. "
        "Analyze the likely near-term GBP/USD sentiment. State the direction, the causal reason, "
        "and what would invalidate it. Do not invent live prices.",
    ),
    (
        "usdjpy_risk_off_ambiguity",
        "A news headline says: A geopolitical shock causes broad risk aversion and safe-haven "
        "demand. Analyze USD/JPY sentiment without pretending the direction is certain. Explain "
        "why US yields and the relative demand for USD versus JPY are needed, and give an "
        "invalidation condition. Do not invent live prices.",
    ),
)


@dataclass(frozen=True, slots=True)
class FXProbeAnswer:
    case_id: str
    prompt: str
    response: str
    generated_tokens: int
    replacement_characters: int


def _resolve_device(value: str, *, require_cuda: bool) -> torch.device:
    requested = value.strip().casefold()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("fx_probe_cuda_required")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("fx_probe_cuda_unavailable")
    return device


def load_training_checkpoint(
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device: torch.device,
) -> tuple[VistaReasoningGPT, BPETokenizer, dict]:
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("fx_probe_checkpoint_payload_invalid")
    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise RuntimeError("fx_probe_checkpoint_missing_model")

    tokenizer = BPETokenizer.load(tokenizer_path)
    if int(config.get("vocab_size", -1)) != tokenizer.vocab_size:
        raise RuntimeError("fx_probe_checkpoint_tokenizer_mismatch")
    model = VistaReasoningGPT(**config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, payload


@torch.no_grad()
def run_fx_sentiment_probe(
    *,
    model: VistaReasoningGPT,
    tokenizer: BPETokenizer,
    max_new_tokens: int = 96,
) -> tuple[FXProbeAnswer, ...]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    answers: list[FXProbeAnswer] = []
    for case_id, prompt in FX_SENTIMENT_PROMPTS:
        prompt_ids = tokenizer.encode(
            build_chat_prompt((("user", prompt),)),
            add_bos=False,
            add_eos=False,
        )
        available = model.max_seq_len - len(prompt_ids)
        if available <= 0:
            raise RuntimeError(f"fx_probe_prompt_exceeds_model_context:{case_id}")
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            inputs,
            max_new_tokens=min(max_new_tokens, available),
            stop_ids=generation_stop_ids(tokenizer),
            do_sample=False,
        )
        continuation = generated[0, len(prompt_ids) :].tolist()
        decoded = tokenizer.decode(generated[0].tolist(), skip_special=False, errors="replace")
        response = extract_assistant_response(decoded)
        answers.append(
            FXProbeAnswer(
                case_id=case_id,
                prompt=prompt,
                response=response,
                generated_tokens=len(continuation),
                replacement_characters=response.count("\ufffd"),
            )
        )
    return tuple(answers)


def render_probe(answers: tuple[FXProbeAnswer, ...], checkpoint: dict) -> str:
    lines = [
        "Vista FX sentiment language probe",
        f"checkpoint_epoch={checkpoint.get('epoch')}",
        f"checkpoint_tokens={checkpoint.get('cumulative_prediction_tokens')}",
        f"exam_correctness_percent={checkpoint.get('exam_correctness_percent')}",
        f"exam_gibberish_answers={checkpoint.get('exam_gibberish_answers')}",
        "",
    ]
    for answer in answers:
        lines.extend(
            (
                "=" * 80,
                f"CASE {answer.case_id}",
                f"PROMPT: {answer.prompt}",
                f"RESPONSE: {answer.response}",
                f"GENERATED_TOKENS: {answer.generated_tokens}",
                f"REPLACEMENT_CHARACTERS: {answer.replacement_characters}",
                "",
            )
        )
    return "\n".join(lines)


def run_finbert_fx_probe(
    *,
    analyzer: FinBertSentimentAnalyzer,
) -> tuple[FXProbeAnswer, ...]:
    prompts = [prompt for _, prompt in FX_SENTIMENT_PROMPTS]
    scored = analyzer.score_batch(prompts)
    answers: list[FXProbeAnswer] = []
    for (case_id, prompt), item in zip(FX_SENTIMENT_PROMPTS, scored):
        summary = (
            f"label={item.label} sentiment_score={item.sentiment_score:+.4f} "
            f"confidence={item.confidence:.4f} scores={json.dumps(item.scores, sort_keys=True)}"
        )
        answers.append(
            FXProbeAnswer(
                case_id=case_id,
                prompt=prompt,
                response=summary,
                generated_tokens=0,
                replacement_characters=0,
            )
        )
    return tuple(answers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic FX-sentiment probes using FinBERT or a retained training checkpoint."
    )
    parser.add_argument("--backend", choices=("finbert", "checkpoint"), default="finbert")
    parser.add_argument("--model-name", default=DEFAULT_FINBERT_MODEL)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    if args.backend == "finbert":
        analyzer = FinBertSentimentAnalyzer.from_pretrained(
            args.model_name,
            device=args.device,
            require_cuda=bool(args.require_cuda),
            local_files_only=bool(args.local_files_only),
        )
        answers = run_finbert_fx_probe(analyzer=analyzer)
        checkpoint = {
            "epoch": None,
            "cumulative_prediction_tokens": None,
            "validation_loss": None,
            "exam_correctness_percent": None,
            "exam_gibberish_answers": 0,
            "backend": "finbert",
            "model_name": args.model_name,
        }
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else (Path(args.bundle_dir) / "fx_sentiment_probe" if args.bundle_dir else Path("logs") / "fx_sentiment_probe")
        )
    else:
        if not args.bundle_dir:
            parser.error("--bundle-dir is required when --backend checkpoint is used")
        bundle = Path(args.bundle_dir)
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else bundle / ".training" / "best_model.pt"
        tokenizer_path = bundle / ".training" / "tokenizer.json"
        output_dir = Path(args.output_dir) if args.output_dir else bundle / "fx_sentiment_probe"
        device = _resolve_device(args.device, require_cuda=bool(args.require_cuda))
        model, tokenizer, checkpoint = load_training_checkpoint(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            device=device,
        )
        answers = run_fx_sentiment_probe(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "fx_sentiment_probe.txt"
    json_path = output_dir / "fx_sentiment_probe.json"
    text_path.write_text(render_probe(answers, checkpoint), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "checkpoint": {
                    key: checkpoint.get(key)
                    for key in (
                        "epoch",
                        "cumulative_prediction_tokens",
                        "validation_loss",
                        "exam_correctness_percent",
                        "exam_gibberish_answers",
                    )
                },
                "answers": [asdict(answer) for answer in answers],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(render_probe(answers, checkpoint), flush=True)
    print(f"[FXProbe] text={text_path} json={json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
