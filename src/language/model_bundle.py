from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.language.loss_objective import LOSS_OBJECTIVE_VERSION
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer, TOKENIZER_ALGORITHM_VERSION


BUNDLE_VERSION = 4


@dataclass(frozen=True)
class ModelManifest:
    bundle_version: int
    architecture: str
    training_stage: str
    tokenizer_algorithm_version: int
    tokenizer_fingerprint: str
    tokenizer_sha256: str
    loss_objective_version: int
    vocab_size: int
    parameter_count: int
    active_parameter_count: int
    activation_ratio: float
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
    loss_objective_version: int = LOSS_OBJECTIVE_VERSION,
) -> Path:
    if tokenizer.algorithm_version != TOKENIZER_ALGORITHM_VERSION:
        raise RuntimeError("refusing_to_save_non_authoritative_tokenizer")
    if int(loss_objective_version) != LOSS_OBJECTIVE_VERSION:
        raise RuntimeError("refusing_to_save_non_authoritative_loss_objective")
    if model_config.get("vocab_size") != tokenizer.vocab_size:
        raise RuntimeError("model_tokenizer_vocab_size_mismatch")

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
            "loss_objective_version": LOSS_OBJECTIVE_VERSION,
        },
        model_path,
    )

    total = model.get_num_params()
    active = model.get_active_params_per_token()
    manifest = ModelManifest(
        bundle_version=BUNDLE_VERSION,
        architecture="VistaReasoningGPT",
        training_stage=str(training_stage),
        tokenizer_algorithm_version=tokenizer.algorithm_version,
        tokenizer_fingerprint=tokenizer.fingerprint(),
        tokenizer_sha256=sha256_file(tokenizer_path),
        loss_objective_version=LOSS_OBJECTIVE_VERSION,
        vocab_size=tokenizer.vocab_size,
        parameter_count=total,
        active_parameter_count=active,
        activation_ratio=active / max(total, 1),
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
    if int(manifest_raw.get("bundle_version", 0)) != BUNDLE_VERSION:
        raise RuntimeError("unsupported_model_bundle_version")

    manifest = ModelManifest(**manifest_raw)
    if manifest.architecture != "VistaReasoningGPT":
        raise RuntimeError("unsupported_model_architecture")
    if manifest.tokenizer_algorithm_version != TOKENIZER_ALGORITHM_VERSION:
        raise RuntimeError("unsupported_bundle_tokenizer_algorithm_version")
    if manifest.loss_objective_version != LOSS_OBJECTIVE_VERSION:
        raise RuntimeError("unsupported_bundle_loss_objective_version")

    if sha256_file(tokenizer_path) != manifest.tokenizer_sha256:
        raise RuntimeError("tokenizer_sha256_mismatch")
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.algorithm_version != manifest.tokenizer_algorithm_version:
        raise RuntimeError("tokenizer_algorithm_version_mismatch")
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
    if config.get("vocab_size") != tokenizer.vocab_size:
        raise RuntimeError("checkpoint_tokenizer_vocab_size_mismatch")
    if int(checkpoint.get("loss_objective_version", 0)) != LOSS_OBJECTIVE_VERSION:
        raise RuntimeError("checkpoint_loss_objective_version_mismatch")

    model = VistaReasoningGPT(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if model.get_num_params() != manifest.parameter_count:
        raise RuntimeError("parameter_count_mismatch")
    if model.get_active_params_per_token() != manifest.active_parameter_count:
        raise RuntimeError("active_parameter_count_mismatch")
    return model, tokenizer, manifest
