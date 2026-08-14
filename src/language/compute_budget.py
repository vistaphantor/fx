from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch

from src.language.loss_objective import build_loss_targets
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer

REFERENCE_TOKENS_PER_PARAMETER = 20.0
DEFAULT_WALL_CLOCK_HOURS = 4.0


@dataclass(frozen=True, slots=True)
class TrainingThroughputReport:
    parameter_count: int
    active_parameter_count: int
    activation_ratio: float
    benchmark_steps: int
    benchmark_prediction_tokens: int
    elapsed_seconds: float
    useful_tokens_per_second: float
    wall_clock_hours: float
    projected_useful_tokens: int
    projected_tokens_per_parameter: float
    projected_tokens_per_active_parameter: float
    reference_tokens_per_parameter: float
    reference_target_tokens: int
    projected_hours_to_reference_target: float
    required_tokens_per_second_for_reference_in_window: float

    def to_dict(self) -> dict:
        return asdict(self)


def reference_token_target(
    parameter_count: int,
    *,
    tokens_per_parameter: float = REFERENCE_TOKENS_PER_PARAMETER,
) -> int:
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    if tokens_per_parameter <= 0:
        raise ValueError("tokens_per_parameter must be positive")
    return int(math.ceil(parameter_count * tokens_per_parameter))


def required_tokens_per_second(target_tokens: int, wall_clock_hours: float) -> float:
    if target_tokens <= 0 or wall_clock_hours <= 0:
        raise ValueError("target_tokens and wall_clock_hours must be positive")
    return target_tokens / (wall_clock_hours * 3600.0)


def _build_probe_batch(
    tokenizer: BPETokenizer,
    sequences: list[list[int]],
    *,
    seq_len: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build a representative probe from windows that actually carry loss.

    Assistant-supervised training can legitimately create prompt-only windows
    after context chunking. Those windows are context, not optimizer work, and
    must not make hardware preflight fail or inflate throughput measurements.
    """
    if not sequences:
        raise ValueError("sequences must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    trainable: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for sequence in sequences:
        x_ids, y_ids, stats = build_loss_targets(
            sequence,
            seq_len=seq_len,
            pad_id=tokenizer.pad_id(),
        )
        if stats.prediction_tokens <= 0:
            continue
        trainable.append(
            (
                torch.tensor(x_ids, dtype=torch.long),
                torch.tensor(y_ids, dtype=torch.long),
                stats.prediction_tokens,
            )
        )
        if len(trainable) >= batch_size:
            break

    if not trainable:
        raise RuntimeError("compute_probe_has_no_prediction_tokens")

    while len(trainable) < batch_size:
        trainable.append(trainable[len(trainable) % len(trainable)])

    x = torch.stack([pair[0] for pair in trainable[:batch_size]])
    y = torch.stack([pair[1] for pair in trainable[:batch_size]])
    valid_tokens = int((y != tokenizer.pad_id()).sum().item())
    if valid_tokens <= 0:
        raise RuntimeError("compute_probe_has_no_prediction_tokens")
    return x, y, valid_tokens


def benchmark_training_throughput(
    *,
    model_config: dict,
    tokenizer: BPETokenizer,
    sequences: list[list[int]],
    batch_size: int,
    steps: int = 3,
    wall_clock_hours: float = DEFAULT_WALL_CLOCK_HOURS,
    reference_tokens_per_parameter: float = REFERENCE_TOKENS_PER_PARAMETER,
) -> TrainingThroughputReport:
    """Measure real CPU forward/backward throughput on supervised targets."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if wall_clock_hours <= 0:
        raise ValueError("wall_clock_hours must be positive")

    seq_len = int(model_config["max_seq_len"])
    x, y, useful_per_step = _build_probe_batch(
        tokenizer, sequences, seq_len=seq_len, batch_size=batch_size,
    )

    torch.manual_seed(20260814)
    model = VistaReasoningGPT(**model_config).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    model.train()

    def update() -> None:
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y, pad_id=tokenizer.pad_id())
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("compute_probe_non_finite_loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    update()
    started = time.perf_counter()
    for _ in range(steps):
        update()
    elapsed = time.perf_counter() - started
    benchmark_tokens = useful_per_step * steps
    tokens_per_second = benchmark_tokens / max(elapsed, 1e-9)

    params = model.get_num_params()
    active = model.get_active_params_per_token()
    projected = int(tokens_per_second * wall_clock_hours * 3600.0)
    reference_target = reference_token_target(
        params, tokens_per_parameter=reference_tokens_per_parameter,
    )
    hours_to_target = reference_target / max(tokens_per_second, 1e-9) / 3600.0

    return TrainingThroughputReport(
        parameter_count=params,
        active_parameter_count=active,
        activation_ratio=active / max(params, 1),
        benchmark_steps=steps,
        benchmark_prediction_tokens=benchmark_tokens,
        elapsed_seconds=elapsed,
        useful_tokens_per_second=tokens_per_second,
        wall_clock_hours=wall_clock_hours,
        projected_useful_tokens=projected,
        projected_tokens_per_parameter=projected / max(params, 1),
        projected_tokens_per_active_parameter=projected / max(active, 1),
        reference_tokens_per_parameter=reference_tokens_per_parameter,
        reference_target_tokens=reference_target,
        projected_hours_to_reference_target=hours_to_target,
        required_tokens_per_second_for_reference_in_window=required_tokens_per_second(
            reference_target, wall_clock_hours,
        ),
    )
