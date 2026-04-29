# Continuation-Aware Execution And Quant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bot’s blunt continuation rejection math with continuation-aware execution and quant logic that allows lower immediate `R:R` when persistence and post-cost expected value are strong enough.

**Architecture:** Keep the top-down decision tree intact, but enrich its trade plans with explicit continuation-state metrics. Move the continuation decision burden into `src/strategy/execution_cost.py` and `src/strategy/quant_engine.py`, where dynamic `R:R` floors, continuation probability, continuation EV, and continuation-aware quant states can be evaluated explicitly and logged cleanly.

**Tech Stack:** Python 3.11/3.12, `pytest`, MetaTrader5, existing quant/strategy modules

---

## File Map

**Existing files to modify**

- `src/strategy/decision_tree.py`
  Emit continuation-state metadata required by execution and quant.
- `src/strategy/execution_cost.py`
  Add continuation probability, dynamic `R:R` floor, continuation EV, and richer rejection reasons.
- `src/strategy/quant_engine.py`
  Add continuation-aware approval/reduction/block behavior and continuation-specific reasons.
- `src/live_trade_loop.py`
  Route continuation metadata into execution/quant and improve logging reasons.
- `tests/strategy/test_execution_cost.py`
  Cover continuation-aware execution approvals and disaster-floor rejections.
- `tests/strategy/test_quant_engine.py`
  Cover continuation-aware quant states and retained fresh-entry strictness.
- `tests/strategy/test_decision_tree.py`
  Cover continuation metadata emission.
- `tests/test_live_trade_loop.py`
  Cover runtime logging and routing of continuation-aware decisions.

**New files to create**

- None required unless helper extraction becomes necessary during implementation.

**Workspace note**

This workspace is not currently a git repository, so commit steps below are included as hygiene but will be skipped locally unless the repo is initialized later.

### Task 1: Add Continuation Metadata To Trade Plans

**Files:**
- Modify: `src/strategy/decision_tree.py`
- Modify: `tests/strategy/test_decision_tree.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_trade_plan_includes_continuation_metadata_for_execution_layers():
    result = evaluate_top_down_decision_tree(...)
    assert result.metadata["is_continuation_setup"] is True
    assert "m15_quality" in result.metadata
    assert "m10_quality" in result.metadata
    assert "m5_quality" in result.metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_decision_tree.py -q`
Expected: FAIL because continuation metadata is not fully exposed.

- [ ] **Step 3: Write minimal implementation**

```python
metadata={
    ...
    "is_continuation_setup": True,
    "m15_quality": trigger_decision.quality_score,
    "m10_quality": refinement_decision.quality_score,
    "m5_quality": execution_decision.quality_score,
    "range_expansion_ratio": volatility_state.range_expansion_ratio,
    "body_efficiency": volatility_state.body_efficiency,
    "regime_confidence": regime_state.confidence,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_decision_tree.py -q`
Expected: PASS with continuation metadata present.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/decision_tree.py tests/strategy/test_decision_tree.py
git commit -m "feat: expose continuation metadata in trade plans"
```

### Task 2: Add Continuation Probability And Dynamic RR Floor

**Files:**
- Modify: `src/strategy/execution_cost.py`
- Modify: `tests/strategy/test_execution_cost.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_execution_allows_strong_continuation_below_legacy_rr_floor():
    assessment = assess_market_order_execution(..., continuation_inputs=...)
    assert assessment.is_tradeable is True

def test_execution_rejects_below_disaster_floor_even_with_good_momentum():
    assessment = assess_market_order_execution(..., continuation_inputs=...)
    assert assessment.reason == "continuation_rr_below_disaster_floor"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_execution_cost.py -q`
Expected: FAIL because execution still uses the legacy hard `1.5R` continuation veto.

- [ ] **Step 3: Write minimal implementation**

```python
def _continuation_probability(...):
    ...

def _dynamic_rr_floor(...):
    ...
```

```python
if effective_rr < 0.60:
    return _reject(reason="continuation_rr_below_disaster_floor", ...)

if effective_rr < dynamic_rr_floor and continuation_ev <= 0.0:
    return _reject(reason="continuation_ev_negative", ...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_execution_cost.py -q`
Expected: PASS with disaster-floor and dynamic-floor behavior.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/execution_cost.py tests/strategy/test_execution_cost.py
git commit -m "feat: add continuation-aware execution gating"
```

### Task 3: Add Continuation EV And Richer Execution Reasons

**Files:**
- Modify: `src/strategy/execution_cost.py`
- Modify: `tests/strategy/test_execution_cost.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_execution_returns_positive_continuation_ev_for_strong_momentum_case():
    assessment = assess_market_order_execution(..., continuation_inputs=...)
    assert assessment.reason == "execution_approved"
    assert assessment.continuation_ev > 0

def test_execution_penalty_can_still_block_continuation_when_costs_are_extreme():
    assessment = assess_market_order_execution(..., continuation_inputs=...)
    assert assessment.reason in {"execution_penalty_too_high", "continuation_ev_negative"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_execution_cost.py -q`
Expected: FAIL because continuation EV and richer reasons are not returned yet.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class ExecutionAssessment:
    ...
    continuation_probability: float = 0.0
    continuation_ev: float = 0.0
    dynamic_rr_floor: float = 1.5
```

```python
continuation_ev = (
    continuation_probability * effective_rr
    - (1.0 - continuation_probability)
    - execution_penalty
    - tail_penalty
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_execution_cost.py -q`
Expected: PASS with continuation EV and richer assessment fields.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/execution_cost.py tests/strategy/test_execution_cost.py
git commit -m "feat: add continuation expected-value execution math"
```

### Task 4: Add Continuation-Aware Quant States

**Files:**
- Modify: `src/strategy/quant_engine.py`
- Modify: `tests/strategy/test_quant_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_continuation_quant_can_approve_trade_with_positive_continuation_ev():
    decision = evaluate_master_equation(..., continuation_context=...)
    assert decision.reason == "continuation_quant_approved"
    assert decision.is_trade is True

def test_continuation_quant_can_reduce_size_in_borderline_case():
    decision = evaluate_master_equation(..., continuation_context=...)
    assert decision.reason == "continuation_quant_reduced"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_quant_engine.py -q`
Expected: FAIL because the quant engine only emits the current flat/approved states.

- [ ] **Step 3: Write minimal implementation**

```python
def evaluate_master_equation(..., continuation_context=None):
    ...
```

```python
if continuation_context is not None:
    return _evaluate_continuation_quant(...)
```

The continuation path should support:

- `continuation_quant_approved`
- `continuation_quant_reduced`
- `continuation_quant_blocked`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_quant_engine.py -q`
Expected: PASS with continuation-aware quant states while preserving legacy fresh-entry behavior.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/quant_engine.py tests/strategy/test_quant_engine.py
git commit -m "feat: add continuation-aware quant decisions"
```

### Task 5: Route Continuation Context Through Live Runtime

**Files:**
- Modify: `src/live_trade_loop.py`
- Modify: `tests/test_live_trade_loop.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_live_loop_logs_continuation_execution_reason_when_rr_relaxed():
    ...

def test_live_loop_logs_continuation_quant_reason_instead_of_master_flat():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_live_trade_loop.py -q`
Expected: FAIL because the live loop does not yet pass or log continuation-specific states.

- [ ] **Step 3: Write minimal implementation**

```python
continuation_context = {
    "is_continuation_setup": metadata.get("is_continuation_setup", False),
    "m15_quality": metadata.get("m15_quality", 0.0),
    "m10_quality": metadata.get("m10_quality", 0.0),
    "m5_quality": metadata.get("m5_quality", 0.0),
    ...
}
```

Pass this context into:

- `assess_market_order_execution(...)`
- `evaluate_master_equation(...)`

and log their continuation-specific reasons directly.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_live_trade_loop.py -q`
Expected: PASS with continuation-aware runtime reasons.

- [ ] **Step 5: Commit**

```bash
git add src/live_trade_loop.py tests/test_live_trade_loop.py
git commit -m "feat: route continuation context through live runtime"
```

### Task 6: Protect Fresh-Entry Strictness

**Files:**
- Modify: `tests/strategy/test_quant_engine.py`
- Modify: `tests/strategy/test_execution_cost.py`

- [ ] **Step 1: Write the failing regression tests**

```python
def test_fresh_entry_still_rejects_weak_rr_trade_without_continuation_context():
    ...

def test_fresh_entry_quant_remains_strict_when_no_continuation_context_present():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_execution_cost.py tests/strategy/test_quant_engine.py -q`
Expected: FAIL until explicit regression coverage is added or implementation reveals leakage.

- [ ] **Step 3: Write minimal implementation or guards if needed**

```python
if not continuation_context:
    # preserve existing fresh-entry rules
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_execution_cost.py tests/strategy/test_quant_engine.py -q`
Expected: PASS with strict fresh-entry behavior preserved.

- [ ] **Step 5: Commit**

```bash
git add tests/strategy/test_execution_cost.py tests/strategy/test_quant_engine.py src/strategy/execution_cost.py src/strategy/quant_engine.py
git commit -m "test: preserve fresh-entry strictness under continuation upgrade"
```

### Task 7: Full Verification And Live-Safe Review

**Files:**
- Modify: any touched files if verification reveals issues

- [ ] **Step 1: Run targeted strategy tests**

Run: `python -m pytest tests/strategy/test_decision_tree.py tests/strategy/test_execution_cost.py tests/strategy/test_quant_engine.py tests/test_live_trade_loop.py -q`
Expected: PASS across continuation-aware execution and quant surfaces.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS with no regressions to the broader bot behavior.

- [ ] **Step 3: Perform a safe read-only diagnostic if needed**

Run a non-order-placing local diagnostic path against current MT5 data.
Expected: continuation-aware reasons appear in diagnostics without placing a trade.

- [ ] **Step 4: Review live-risk implications**

Confirm:

- disaster floor still exists
- quant can still hard-block weak continuation
- logs now distinguish continuation approval vs block vs reduction

- [ ] **Step 5: Commit**

```bash
git add src/strategy/decision_tree.py src/strategy/execution_cost.py src/strategy/quant_engine.py src/live_trade_loop.py tests
git commit -m "feat: add continuation-aware execution and quant math"
```
