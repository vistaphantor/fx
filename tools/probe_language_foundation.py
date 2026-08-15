from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer


DEFAULT_PROMPTS = (
    "Once upon a time",
    "The little boy went to",
    "The weather was cold, so",
    "The woman opened the door and",
    "A market is a place where",
    "Gold prices rose because",
    "The cat sat on the",
    "When people are hungry, they",
)


def _load_checkpoint(bundle: Path, checkpoint: str):
    work = bundle / ".training"
    tokenizer_path = work / "tokenizer.json"
    if not tokenizer_path.exists():
        raise RuntimeError(f"tokenizer_missing:{tokenizer_path}")

    if checkpoint == "current":
        checkpoint_path = work / "training_state.pt"
    elif checkpoint == "best":
        checkpoint_path = work / "best_model.pt"
    else:
        checkpoint_path = Path(checkpoint)

    if not checkpoint_path.exists():
        raise RuntimeError(f"checkpoint_missing:{checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = payload.get("model_config")
    model_state = payload.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(model_state, dict):
        raise RuntimeError(f"invalid_language_checkpoint:{checkpoint_path}")

    tokenizer = BPETokenizer.load(tokenizer_path)
    if int(model_config.get("vocab_size", -1)) != tokenizer.vocab_size:
        raise RuntimeError("checkpoint_tokenizer_vocab_mismatch")

    model = VistaReasoningGPT(**model_config)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model, tokenizer, payload, checkpoint_path


def _decode_token(tokenizer: BPETokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special=False, errors="replace")


@torch.no_grad()
def _next_token_report(
    model: VistaReasoningGPT,
    tokenizer: BPETokenizer,
    prompt_ids: list[int],
    *,
    top_n: int,
) -> list[tuple[int, float, str]]:
    ids = torch.tensor([prompt_ids[-model.max_seq_len:]], dtype=torch.long)
    logits, _ = model(ids)
    probs = torch.softmax(logits[0, -1], dim=-1)
    values, indices = torch.topk(probs, min(top_n, probs.numel()))
    return [
        (int(token_id), float(prob), _decode_token(tokenizer, int(token_id)))
        for prob, token_id in zip(values.tolist(), indices.tolist())
    ]


@torch.no_grad()
def _continue(
    model: VistaReasoningGPT,
    tokenizer: BPETokenizer,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    sampled: bool,
    seed: int,
) -> str:
    torch.manual_seed(seed)
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    output = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=40,
        top_p=0.92,
        stop_ids={tokenizer.eos_id()},
        do_sample=sampled,
    )
    continuation = output[0, len(prompt_ids):].tolist()
    return tokenizer.decode(continuation, skip_special=False, errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe a foundation checkpoint as a plain causal language model. "
            "This deliberately bypasses <user>/<assistant> chat formatting."
        )
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument(
        "--checkpoint",
        default="current",
        help="current, best, or an explicit .pt checkpoint path",
    )
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--top-next", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.max_new_tokens <= 0 or args.top_next <= 0:
        raise ValueError("probe token budgets must be positive")

    bundle = Path(args.bundle)
    model, tokenizer, payload, checkpoint_path = _load_checkpoint(bundle, args.checkpoint)
    prompts = tuple(args.prompt) if args.prompt else DEFAULT_PROMPTS

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Parameters: {model.get_num_params()/1e6:.3f}M")
    print(f"Context: {model.max_seq_len}")
    if "epoch" in payload:
        print(f"Checkpoint epoch: {payload['epoch']}")
    if "step" in payload:
        print(f"Checkpoint step: {payload['step']}")
    if "validation_loss" in payload:
        print(f"Validation loss: {payload['validation_loss']}")

    for index, prompt in enumerate(prompts, start=1):
        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        if len(prompt_ids) >= model.max_seq_len:
            raise RuntimeError(f"probe_prompt_exceeds_context:{index}")

        print("\n" + "=" * 88)
        print(f"PROMPT {index}: {prompt!r}")
        print(f"Prompt tokens: {len(prompt_ids)}")
        print("NEXT TOKEN TOP CANDIDATES")
        for rank, (token_id, probability, decoded) in enumerate(
            _next_token_report(
                model,
                tokenizer,
                prompt_ids,
                top_n=args.top_next,
            ),
            start=1,
        ):
            print(
                f"  {rank:>2}. p={probability:7.3%} "
                f"id={token_id:<5} token={decoded!r}"
            )

        greedy = _continue(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            sampled=False,
            seed=args.seed,
        )
        sampled = _continue(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            sampled=True,
            seed=args.seed + index,
        )

        print("GREEDY CONTINUATION")
        print(repr(greedy))
        print("SAMPLED CONTINUATION")
        print(repr(sampled))


if __name__ == "__main__":
    main()
