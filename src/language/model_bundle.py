from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer


BUNDLE_VERSION = 1


@dataclass(frozen=True)
class ModelManifest:
    bundle_version: int
    architecture: str
    training_stage: str
    tokenizer_fingerprint: str
    tokenizer_sha256: str
    vocab_size: int
    parameter_count: int
    corpus_fingerprint: str
    model_config: dict[str, Any]
    metrics: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model_bundle(
    *,
    bundle_dir: str | Path,
    model: VistaReasoningGPT,
    tokenizer: BPETokenizer,
    model_config: dict[str, Any],
    training_stage: str,
    corpus_fingerprint: str,
    metrics: dict[str, Any] | None = None,
) -> Path:
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)

    model_path = bundle / "model.pt"
    tokenizer_path = bundle / "tokenizer.json"
    manifest_path = bundle / "manifest.json"

    tokenizer.save(tokenizer_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dict(model_config),
        },
        model_path,
    )

    manifest = ModelManifest(
        bundle_version=BUNDLE_VERSION,
        architecture="VistaReasoningGPT",
        training_stage=str(training_stage),
        tokenizer_fingerprint=tokenizer.fingerprint(),
        tokenizer_sha256=sha256_file(tokenizer_path),
        vocab_size=tokenizer.vocab_size,
        parameter_count=model.get_num_params(),
        corpus_fingerprint=str(corpus_fingerprint),
        model_config=dict(model_config),
        metrics=dict(metrics or {}),
    )
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return bundle


def load_model_bundle(
    bundle_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[VistaReasoningGPT, BPETokenizer, ModelManifest]:
    bundle = Path(bundle_dir)
    model_path = bundle / "model.pt"
    tokenizer_path = bundle / "tokenizer.json"
    manifest_path = bundle / "manifest.json"

    missing = [path.name for path in (model_path, tokenizer_path, manifest_path) if not path.exists()]
    if missing:
        raise RuntimeError(f"incomplete_model_bundle:{','.join(missing)}")

    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ModelManifest(**manifest_raw)

    if manifest.bundle_version != BUNDLE_VERSION:
        raise RuntimeError("unsupported_model_bundle_version")
    if manifest.architecture != "VistaReasoningGPT":
        raise RuntimeError("unsupported_model_architecture")

    actual_tokenizer_sha = sha256_file(tokenizer_path)
    if actual_tokenizer_sha != manifest.tokenizer_sha256:
        raise RuntimeError("tokenizer_sha256_mismatch")

    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.fingerprint() != manifest.tokenizer_fingerprint:
        raise RuntimeError("tokenizer_fingerprint_mismatch")
    if tokenizer.vocab_size != manifest.vocab_size:
        raise RuntimeError("tokenizer_vocab_size_mismatch")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("model_config_missing")
    if config != manifest.model_config:
        raise RuntimeError("model_config_mismatch")

    model = VistaReasoningGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        ffn_dim=config["ffn_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if model.get_num_params() != manifest.parameter_count:
        raise RuntimeError("parameter_count_mismatch")

    return model, tokenizer, manifest
