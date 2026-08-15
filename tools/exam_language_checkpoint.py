from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.exam import render_exam_text, run_epoch_exam
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer


def _resolve_checkpoint(bundle: Path, checkpoint: str) -> Path:
    work = bundle / ".training"
    if checkpoint == "current":
        return work / "training_state.pt"
    if checkpoint == "best":
        return work / "best_model.pt"
    return Path(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the authoritative deterministic Vista exam against an existing checkpoint without training or mutating the bundle."
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--checkpoint", default="current", help="current, best, or explicit .pt path")
    parser.add_argument("--training-stage", default="foundation")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")

    bundle = Path(args.bundle)
    tokenizer_path = bundle / ".training" / "tokenizer.json"
    checkpoint_path = _resolve_checkpoint(bundle, args.checkpoint)
    if not tokenizer_path.exists():
        raise RuntimeError(f"tokenizer_missing:{tokenizer_path}")
    if not checkpoint_path.exists():
        raise RuntimeError(f"checkpoint_missing:{checkpoint_path}")

    tokenizer = BPETokenizer.load(tokenizer_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = payload.get("model_config")
    model_state = payload.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(model_state, dict):
        raise RuntimeError(f"invalid_language_checkpoint:{checkpoint_path}")
    if int(model_config.get("vocab_size", -1)) != tokenizer.vocab_size:
        raise RuntimeError("checkpoint_tokenizer_vocab_mismatch")

    model = VistaReasoningGPT(**model_config)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    epoch = int(payload.get("epoch", -1))
    train_loss = payload.get("train_loss")
    validation_loss = payload.get("validation_loss")
    result = run_epoch_exam(
        model=model,
        tokenizer=tokenizer,
        epoch=max(epoch, 0),
        training_stage=args.training_stage,
        train_loss=float(train_loss) if train_loss is not None else None,
        validation_loss=float(validation_loss) if validation_loss is not None else None,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Parameters: {model.get_num_params()/1e6:.3f}M")
    print(f"Context: {model.max_seq_len}")
    print(render_exam_text(result))


if __name__ == "__main__":
    main()
