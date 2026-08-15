from __future__ import annotations

import torch

from src.language.exam import exam_questions, render_exam_text, run_epoch_exam
from src.language.protocol import build_exam_prompt
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer


def _tokenizer() -> BPETokenizer:
    corpus = "\n".join(build_exam_prompt(q.prompt) for q in exam_questions("foundation"))
    corpus += "\n4. 7. -3. 12. 5. 13 shillings. A price is the amount of money asked or paid for a good or service. Purchasing power falls."
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=512, min_frequency=1)
    return tokenizer


def _model(tokenizer: BPETokenizer) -> VistaReasoningGPT:
    torch.manual_seed(404)
    return VistaReasoningGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        max_seq_len=96,
        dropout=0.0,
        ffn_type="dense",
        num_experts=1,
        experts_per_token=1,
        moe_ffn_dim=128,
        shared_expert_ffn_dim=0,
        router_aux_loss_coef=0.0,
        router_jitter=0.0,
    )


def test_binoculars_trace_scores_full_vocab_and_correct_target_rank() -> None:
    tokenizer = _tokenizer()
    model = _model(tokenizer)
    result = run_epoch_exam(
        model=model,
        tokenizer=tokenizer,
        epoch=0,
        training_stage="foundation",
        train_loss=None,
        validation_loss=None,
        max_new_tokens=6,
    )

    answer = result.answers[0]
    assert answer.decision_trace
    assert answer.target_trace
    assert all(step.candidates_evaluated == tokenizer.vocab_size for step in answer.decision_trace)
    assert all(1 <= step.target_rank <= tokenizer.vocab_size for step in answer.target_trace)
    assert all(len(step.top_candidates) <= 8 for step in answer.target_trace)
    assert result.mean_prompt_js_bits >= 0.0
    assert result.min_prompt_js_bits >= 0.0
    assert result.max_prompt_js_bits >= result.min_prompt_js_bits
    assert result.parameter_health
    assert {item.group for item in result.parameter_health} >= {
        "token_embedding_tied_head",
        "attention_q_proj",
        "attention_k_proj",
        "attention_v_proj",
        "attention_out_proj",
        "normalization",
    }

    text = render_exam_text(result)
    assert "BINOCULARS — PROMPT SENSITIVITY" in text
    assert "BINOCULARS — PARAMETER HEALTH" in text
    assert "MODEL DECISION TRACE" in text
    assert "CORRECT TARGET PATH" in text
    assert "rank=" in text
    assert "logit=" in text
    assert "entropy=" in text


def test_binoculars_are_deterministic_and_rng_neutral() -> None:
    tokenizer = _tokenizer()
    model = _model(tokenizer)
    model.train()
    torch.manual_seed(991)
    before = torch.get_rng_state().clone()
    first = run_epoch_exam(
        model=model,
        tokenizer=tokenizer,
        epoch=2,
        training_stage="foundation",
        train_loss=3.0,
        validation_loss=3.1,
        max_new_tokens=5,
    )
    after = torch.get_rng_state().clone()
    second = run_epoch_exam(
        model=model,
        tokenizer=tokenizer,
        epoch=2,
        training_stage="foundation",
        train_loss=3.0,
        validation_loss=3.1,
        max_new_tokens=5,
    )
    assert first == second
    assert torch.equal(before, after)
    assert model.training is True
