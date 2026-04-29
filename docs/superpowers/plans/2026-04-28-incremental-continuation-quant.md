# Incremental Continuation Quant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the bot’s continuation quant math so action selection is driven by incremental certainty-equivalent trade value, continuation-conditioned expectancy, directional tail risk, and truthful runtime logging rather than absolute-wealth utility compression.

**Architecture:** Keep the existing top-down strategy tree and continuation-aware execution overlay, but upgrade the signal-to-action path in five controlled layers: stronger `M10` continuation state, richer execution-side continuation economics, incremental quant action math, runtime reason preservation, and end-to-end regression/diagnostic verification. The implementation should preserve fresh-entry strictness while giving continuation trades a mathematically defensible path to `approve`, `reduce`, or `block`.

**Tech Stack:** Python 3.11/3.12, `pytest`, MetaTrader5, existing strategy/quant modules

---

## File Map

**Existing files to modify**

- `src/strategy/setup.py`
  Strengthen `M10` continuation scoring and expose richer continuation metadata.
- `src/strategy/execution_cost.py`
  Compute continuation-conditioned expectancy inputs, directional tail proxy, and quant-facing economics.
- `src/strategy/quant_engine.py`
  Replace wealth-level action selection with incremental certainty-equivalent selection and expose continuation-specific quant outcomes.
- `src/live_trade_loop.py`
  Preserve the true no-trade reason and route the new continuation economics into quant.
- `tests/strategy/test_setup.py`
  Cover stronger `M10` continuation scoring and metadata.
- `tests/strategy/test_execution_cost.py`
  Cover continuation-conditioned expectancy and directional tail behavior.
- `tests/strategy/test_quant_engine.py`
  Cover certainty-equivalent action selection and continuation quant states.
- `tests/test_live_trade_loop.py`
  Cover runtime reason preservation and continuation routing.

**New files to create**

- None required unless helper extraction becomes necessary during implementation.

**Workspace note**

This workspace is not currently a git repository, so commit steps below are included as hygiene but will be skipped locally unless the repo is initialized later.

### Task 1: Strengthen `M10` Continuation Scoring

**Files:**
- Modify: `src/strategy/setup.py`
- Modify: `tests/strategy/test_setup.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_m10_continuation_score_rewards_orderly_shallow_pullback():
    decision = evaluate_m10_setup(...)
    assert decision.is_ready is True
    assert decision.metadata["continuation_score"] >= 0.55
    assert decision.metadata["structure_intact"] is True


def test_m10_continuation_metadata_exposes_decay_terms_when_not_ready():
    decision = evaluate_m10_setup(...)
    assert decision.is_ready is False
    assert "slope_persistence" in decision.metadata
    assert "expansion_decay" in decision.metadata
    assert "retrace_damage" in decision.metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_setup.py -q`
Expected: FAIL because `M10` metadata does not yet expose all new continuation state and the current scoring is too limited.

- [ ] **Step 3: Write minimal implementation**

```python
continuation_score = (
    retrace_quality * 0.24
    + structure_integrity * 0.22
    + reclaim_quality * 0.18
    + slope_persistence * 0.16
    + expansion_persistence * 0.12
    + entry_distance_score * 0.08
    - retrace_damage * 0.18
)
```

Expose at least:

```python
metadata = {
    ...,
    "slope_persistence": slope_persistence,
    "expansion_decay": expansion_decay,
    "retrace_damage": retrace_damage,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_setup.py -q`
Expected: PASS with richer `M10` continuation scoring and metadata.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/setup.py tests/strategy/test_setup.py
git commit -m "feat: strengthen m10 continuation scoring"
```

### Task 2: Add Continuation-Conditioned Execution Economics

**Files:**
- Modify: `src/strategy/execution_cost.py`
- Modify: `tests/strategy/test_execution_cost.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_execution_assessment_exposes_mu_cont_for_continuation_setup():
    assessment = assess_market_order_execution(..., continuation_context=...)
    assert assessment.continuation_probability > 0
    assert assessment.metadata["mu_cont"] > -1


def test_directional_tail_proxy_differs_for_long_and_short():
    long_assessment = assess_market_order_execution(..., direction=BreakoutDirection.BULLISH, continuation_context=...)
    short_assessment = assess_market_order_execution(..., direction=BreakoutDirection.BEARISH, continuation_context=...)
    assert long_assessment.metadata["directional_tail_proxy"] != short_assessment.metadata["directional_tail_proxy"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_execution_cost.py -q`
Expected: FAIL because the execution assessment does not yet expose `mu_cont` or a directional tail proxy.

- [ ] **Step 3: Write minimal implementation**

Extend `ExecutionAssessment` with quant-facing economics, for example:

```python
@dataclass(frozen=True, slots=True)
class ExecutionAssessment:
    ...
    effective_gain_remaining: float = 0.0
    effective_loss_if_failed: float = 0.0
    directional_tail_proxy: float = 0.0
    continuation_mu: float = 0.0
```

Then compute:

```python
effective_gain_remaining = effective_reward_distance / max(abs(market_price), 1e-9)
effective_loss_if_failed = effective_stop_distance / max(abs(market_price), 1e-9)
directional_tail_proxy = max(
    float(volatility_state.range_expansion_ratio) - 1.0,
    0.0,
)
continuation_mu = (
    continuation_probability * effective_gain_remaining
    - (1.0 - continuation_probability) * effective_loss_if_failed
    - normalized_transaction_cost
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_execution_cost.py -q`
Expected: PASS with continuation-conditioned execution economics exposed.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/execution_cost.py tests/strategy/test_execution_cost.py
git commit -m "feat: expose continuation execution economics"
```

### Task 3: Rebase Quant To Incremental Certainty-Equivalent Math

**Files:**
- Modify: `src/strategy/quant_engine.py`
- Modify: `tests/strategy/test_quant_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_certainty_equivalent_prefers_trade_when_mu_cont_is_positive():
    decision = evaluate_master_equation(..., continuation_context=...)
    assert decision.is_trade is True
    assert decision.reason in {"continuation_quant_approved", "continuation_quant_reduced"}
    assert decision.metadata["ce_scores"][decision.action] > decision.metadata["ce_scores"][0]


def test_certainty_equivalent_prefers_flat_when_directional_tail_is_too_large():
    decision = evaluate_master_equation(..., continuation_context=...)
    assert decision.is_trade is False
    assert decision.reason == "continuation_quant_blocked"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_quant_engine.py -q`
Expected: FAIL because the current quant engine still uses wealth-level CARA utility for action selection.

- [ ] **Step 3: Write minimal implementation**

Add an incremental certainty-equivalent helper:

```python
def compute_certainty_equivalent(
    *,
    action: int,
    position_fraction: float,
    continuation_mu: float,
    transaction_cost: float,
    cvar_dir: float,
    cvar_eta: float,
    drawdown_ratio: float,
    dd_rho: float,
) -> float:
    return (
        action * position_fraction * continuation_mu
        - abs(action) * transaction_cost
        - abs(action) * cvar_eta * cvar_dir
        - dd_rho * drawdown_ratio
    )
```

Then choose:

```python
best_action = max(ce_scores, key=ce_scores.get)
```

Retain optional CARA reporting only on the incremental trade value if needed for logs.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_quant_engine.py -q`
Expected: PASS with continuation trades now judged by incremental certainty-equivalent value.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/quant_engine.py tests/strategy/test_quant_engine.py
git commit -m "feat: use certainty equivalent quant action math"
```

### Task 4: Route `mu_cont` And Directional Tail Risk Through Quant

**Files:**
- Modify: `src/strategy/quant_engine.py`
- Modify: `src/live_trade_loop.py`
- Modify: `tests/strategy/test_quant_engine.py`
- Modify: `tests/test_live_trade_loop.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_live_loop_passes_execution_continuation_mu_into_quant():
    ...
    assert captured_quant_kwargs["continuation_context"]["mu_cont"] == pytest.approx(expected_mu)


def test_quant_decision_metadata_exposes_directional_tail_and_ce_scores():
    decision = evaluate_master_equation(..., continuation_context=...)
    assert "mu_cont" in decision.metadata
    assert "cvar_dir" in decision.metadata
    assert "ce_scores" in decision.metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_quant_engine.py tests/test_live_trade_loop.py -q`
Expected: FAIL because the live loop and quant metadata do not yet route the new economics explicitly.

- [ ] **Step 3: Write minimal implementation**

In the live loop, enrich the continuation context before quant evaluation:

```python
continuation_context.update(
    {
        "mu_cont": execution_assessment.continuation_mu,
        "cvar_dir": execution_assessment.directional_tail_proxy,
    }
)
```

In quant output metadata, expose:

```python
metadata = {
    ...,
    "mu_cont": continuation_mu,
    "cvar_dir": cvar_dir,
    "ce_scores": ce_scores,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_quant_engine.py tests/test_live_trade_loop.py -q`
Expected: PASS with `mu_cont` and directional tail risk routed through quant.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/quant_engine.py src/live_trade_loop.py tests/strategy/test_quant_engine.py tests/test_live_trade_loop.py
git commit -m "feat: route continuation expectancy into quant"
```

### Task 5: Preserve The True Blocking Reason In The Runtime

**Files:**
- Modify: `src/live_trade_loop.py`
- Modify: `tests/test_live_trade_loop.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_runtime_preserves_strategy_reason_when_tree_is_not_trade():
    ...
    assert result.reason == "m10_setup_not_ready"
    assert result.metadata["node"] == "m10_setup"


def test_quant_override_only_logs_when_strategy_wanted_trade():
    ...
    assert "quant_engine" not in result.reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_live_trade_loop.py -q`
Expected: FAIL because the runtime can still collapse a strategy block into a quant block.

- [ ] **Step 3: Write minimal implementation**

Refactor the live loop branch so that:

```python
if not strategy_result.is_trade:
    return no_trade_from_strategy(...)

quant_decision = evaluate_master_equation(...)
if not quant_decision.is_trade:
    return no_trade_from_quant(...)
```

This preserves the strategy’s no-trade reason when no trade plan exists.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_live_trade_loop.py -q`
Expected: PASS with truthful no-trade reason routing.

- [ ] **Step 5: Commit**

```bash
git add src/live_trade_loop.py tests/test_live_trade_loop.py
git commit -m "fix: preserve strategy no-trade reasons before quant"
```

### Task 6: Re-Tune Continuation Quant Outcomes

**Files:**
- Modify: `src/strategy/quant_engine.py`
- Modify: `tests/strategy/test_quant_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_continuation_quant_reduces_size_for_positive_but_weak_ce():
    decision = evaluate_master_equation(..., continuation_context=...)
    assert decision.reason == "continuation_quant_reduced"
    assert 0.0 < decision.lot_multiplier < 1.0


def test_fresh_entry_logic_remains_strict_after_refactor():
    decision = evaluate_master_equation(..., continuation_context=None)
    assert decision.reason in {"omega_below_threshold", "master_equation_flat"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_quant_engine.py -q`
Expected: FAIL because the new certainty-equivalent path will still need explicit sizing-state tuning.

- [ ] **Step 3: Write minimal implementation**

Add continuation-specific thresholds based on certainty-equivalent score:

```python
if ce_trade <= 0:
    reason = "continuation_quant_blocked"
elif ce_trade < ce_reduce_threshold:
    reason = "continuation_quant_reduced"
    lot_multiplier = 0.5
else:
    reason = "continuation_quant_approved"
    lot_multiplier = 1.0
```

Keep the fresh-entry path unchanged unless a trade plan is explicitly marked as continuation.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_quant_engine.py -q`
Expected: PASS with stable continuation approve/reduce/block outcomes and preserved fresh-entry strictness.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/quant_engine.py tests/strategy/test_quant_engine.py
git commit -m "feat: tune continuation quant states"
```

### Task 7: Run Focused Verification

**Files:**
- Modify: none
- Test: `tests/strategy/test_setup.py`
- Test: `tests/strategy/test_execution_cost.py`
- Test: `tests/strategy/test_quant_engine.py`
- Test: `tests/test_live_trade_loop.py`

- [ ] **Step 1: Run the focused regression band**

Run:

```bash
python -m pytest tests/strategy/test_setup.py tests/strategy/test_execution_cost.py tests/strategy/test_quant_engine.py tests/test_live_trade_loop.py -q
```

Expected: PASS for the entire continuation quant band.

- [ ] **Step 2: Investigate and fix any failures**

If a test fails:
- inspect the exact failure
- apply the minimum code change necessary
- rerun the same focused band until clean

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "test: verify incremental continuation quant band"
```

### Task 8: Run Full-Suite And Read-Only Live Diagnostic

**Files:**
- Modify: none unless verification exposes a defect
- Test: full suite

- [ ] **Step 1: Run the full automated suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS with no regression outside the continuation stack.

- [ ] **Step 2: Run a read-only MT5 diagnostic**

Use an inline script pattern similar to the existing safe diagnostics to:
- load settings
- initialize MT5
- build the live strategy input for the configured symbol
- evaluate the tree
- if a trade plan exists, run execution assessment and quant evaluation
- print:
  - `TREE_IS_TRADE`
  - `TREE_REASON`
  - `MU_CONT`
  - `CVAR_DIR`
  - `CE_SCORES`
  - final blocker layer: strategy, execution, or quant

Expected:
- no orders placed
- no position modifications
- explicit report of the active bottleneck

- [ ] **Step 3: Document verification results in the final handoff**

Capture:
- focused test results
- full suite result
- read-only diagnostic result

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "chore: finalize incremental continuation quant verification"
```

## Notes For The Implementer

- Do not weaken fresh-entry logic while improving continuation logic.
- Keep hard disaster floors intact in the execution layer.
- Prefer exposing new metrics through metadata rather than inventing parallel hidden state.
- Preserve existing public signatures where possible; add optional fields rather than breaking callers.
- If helper extraction becomes necessary, keep files focused and small rather than growing `quant_engine.py` or `setup.py` further without boundaries.
