from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExamQuestion:
    question_id: str
    category: str
    prompt: str
    expected_all: tuple[str, ...] = ()
    expected_any: tuple[str, ...] = ()
    numeric_answer: str | None = None
    expected_expression: str | None = None
    numeric_units: tuple[str, ...] = ()
    diagnostic_target: str | None = None
    skill: str | None = None
    conceptual_gate: bool = False


@dataclass(frozen=True)
class TokenCandidate:
    rank: int
    token_id: int
    token: str
    logit: float
    probability: float


@dataclass(frozen=True)
class DecisionTrace:
    step: int
    chosen_token_id: int
    chosen_token: str
    chosen_logit: float
    chosen_probability: float
    entropy_bits: float
    winner_margin_probability: float
    candidates_evaluated: int
    top_candidates: tuple[TokenCandidate, ...]


@dataclass(frozen=True)
class TargetTokenTrace:
    step: int
    target_token_id: int
    target_token: str
    target_rank: int
    target_logit: float
    target_probability: float
    winning_token_id: int
    winning_token: str
    winning_probability: float
    entropy_bits: float
    candidates_evaluated: int
    top_candidates: tuple[TokenCandidate, ...]


@dataclass(frozen=True)
class ParameterHealth:
    group: str
    parameters: int
    rms: float
    l2_norm: float
    max_abs: float


@dataclass(frozen=True)
class ExamAnswer:
    question_id: str
    category: str
    prompt: str
    raw_output: str
    normalized_output: str
    generated_tokens: int
    correct: bool
    quality_score: float
    repetition_ratio: float
    gibberish_flags: tuple[str, ...]
    skill: str | None = None
    conceptual_gate: bool = False
    decision_trace: tuple[DecisionTrace, ...] = ()
    target_trace: tuple[TargetTokenTrace, ...] = ()


@dataclass(frozen=True)
class EpochExamResult:
    exam_version: int
    exam_contract_fingerprint: str
    decoding_mode: str
    epoch: int
    training_stage: str
    train_loss: float | None
    validation_loss: float | None
    total_questions: int
    correct_questions: int
    correctness_percent: float
    mean_quality_percent: float
    gibberish_answers: int
    mean_generated_tokens: float
    training_signal: str
    answer_diversity_percent: float
    max_answer_collision: int
    max_prefix_collision: int
    mode_collapse: bool
    answers: tuple[ExamAnswer, ...]
    mean_prompt_js_bits: float = 0.0
    min_prompt_js_bits: float = 0.0
    max_prompt_js_bits: float = 0.0
    parameter_health: tuple[ParameterHealth, ...] = ()
