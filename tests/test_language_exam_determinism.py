from __future__ import annotations

import torch

from src.language.exam import (
    EXAM_DECODING_MODE,
    exam_contract_fingerprint,
    exam_contract_payload,
    exam_questions,
    run_epoch_exam,
)
from src.language.protocol import build_exam_prompt
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer


def _tokenizer() -> BPETokenizer:
    corpus = "\n".join(
        build_exam_prompt(question.prompt)
        for question in exam_questions("foundation")
    )
    corpus += "\n4 210 3 bullish rising risk loss Charlie bid ask spread volatility range."
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=512, min_frequency=1)
    return tokenizer


def _model(tokenizer: BPETokenizer) -> VistaReasoningGPT:
    torch.manual_seed(1234)
    return VistaReasoningGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        max_seq_len=96,
        dropout=0.1,
        ffn_type="dense",
        num_experts=1,
        experts_per_token=1,
        moe_ffn_dim=128,
        shared_expert_ffn_dim=0,
        router_aux_loss_coef=0.0,
        router_jitter=0.0,
    )


def test_exam_contract_is_stable_and_contains_serialized_prompts() -> None:
    first = exam_contract_fingerprint("foundation")
    second = exam_contract_fingerprint("foundation")
    payload = exam_contract_payload("foundation")

    assert first == second
    assert len(first) == 64
    assert payload["decoding_mode"] == EXAM_DECODING_MODE
    assert [item["question_id"] for item in payload["questions"]] == [
        question.question_id for question in exam_questions("foundation")
    ]
    assert all(
        item["serialized_prompt"] == build_exam_prompt(item["prompt"])
        for item in payload["questions"]
    )


def test_greedy_generation_does_not_consume_torch_rng() -> None:
    tokenizer = _tokenizer()
    model = _model(tokenizer)
    prompt = tokenizer.encode(
        build_exam_prompt("What is 2 + 2?"),
        add_bos=False,
        add_eos=False,
    )
    ids = torch.tensor([prompt], dtype=torch.long)

    torch.manual_seed(777)
    before = torch.get_rng_state().clone()
    first = model.generate(ids, max_new_tokens=8, do_sample=False)
    after = torch.get_rng_state().clone()
    second = model.generate(ids, max_new_tokens=8, do_sample=False)

    assert torch.equal(before, after)
    assert torch.equal(first, second)


def test_epoch_exam_is_repeatable_and_observer_neutral() -> None:
    tokenizer = _tokenizer()
    model = _model(tokenizer)
    model.train()

    torch.manual_seed(999)
    before = torch.get_rng_state().clone()
    first = run_epoch_exam(
        model=model,
        tokenizer=tokenizer,
        epoch=3,
        training_stage="foundation",
        train_loss=2.5,
        validation_loss=2.8,
        max_new_tokens=8,
    )
    after_first = torch.get_rng_state().clone()
    second = run_epoch_exam(
        model=model,
        tokenizer=tokenizer,
        epoch=3,
        training_stage="foundation",
        train_loss=2.5,
        validation_loss=2.8,
        max_new_tokens=8,
    )
    after_second = torch.get_rng_state().clone()

    assert first == second
    assert torch.equal(before, after_first)
    assert torch.equal(before, after_second)
    assert model.training is True
    assert first.decoding_mode == "greedy_argmax_v1"
    assert first.exam_contract_fingerprint == exam_contract_fingerprint("foundation")
