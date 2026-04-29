# Quant Scoring Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the live MT5 bot from mostly categorical decision gates to a volatility-normalized, regime-aware scoring engine with adaptive thresholds and stricter campaign add logic.

**Architecture:** Keep the existing top-down `D1 -> H4 -> H1 -> M30 -> M15` flow, but insert a quantitative layer that computes volatility state, regime, feature scores, uncertainty penalties, and net edge before any trade or add-on decision. The live loop remains a thin orchestrator while `src/strategy/decision_tree.py` becomes the coordinator for numeric scoring and `src/strategy/management.py` becomes stricter about campaign continuation quality.

**Tech Stack:** Python 3.11/3.12, MetaTrader5 Python package, `pytest`, `python-dotenv`

---

## File Map

**Existing files to modify**

- `src/config.py`
  Add symbol-aware strategy profile settings and adaptive threshold configuration.
- `.env.example`
  Document new quant strategy settings with safe defaults.
- `src/strategy/context.py`
  Expose normalized location inputs needed by the scoring layer.
- `src/strategy/direction.py`
  Reduce direct decision ownership and shift to quantified directional contributions.
- `src/strategy/setup.py`
  Replace mostly categorical `M30` readiness with numeric setup quality features.
- `src/strategy/trigger.py`
  Replace mostly binary `M15` trigger readiness with quantified trigger quality outputs.
- `src/strategy/decision_tree.py`
  Orchestrate volatility, regime, scoring, uncertainty, and adaptive thresholds.
- `src/strategy/management.py`
  Require stricter continuation math for add-ons once a campaign is active.
- `src/live_trade_loop.py`
  Pass any new scoring metadata needed by campaign decisions and log richer outputs.
- `tests/test_config.py`
  Cover new profile settings and validation rules.
- `tests/strategy/test_context.py`
  Cover normalized feature inputs.
- `tests/strategy/test_direction.py`
  Update from hard-gate expectations to directional contribution expectations.
- `tests/strategy/test_setup.py`
  Update from readiness-only expectations to quantified setup scoring.
- `tests/strategy/test_trigger.py`
  Update from readiness-only expectations to quantified trigger scoring.
- `tests/strategy/test_decision_tree.py`
  Extend to cover edge thresholds, uncertainty, and regime-aware outcomes.
- `tests/strategy/test_management.py`
  Extend to cover stricter add-on thresholds once campaigns are active.
- `tests/test_live_trade_loop.py`
  Verify richer decision metadata and campaign behavior still flow correctly through the live loop.

**New files to create**

- `src/strategy/volatility.py`
  ATR, range normalization, body-efficiency, and volatility state helpers.
- `src/strategy/regime.py`
  Regime classification helpers for trend, pullback, compression, expansion, gap-reversion, and gap-acceptance.
- `src/strategy/scoring.py`
  Central scoring engine for bullish/bearish side scores, uncertainty penalties, expected move, and adaptive thresholds.
- `tests/strategy/test_volatility.py`
- `tests/strategy/test_regime.py`
- `tests/strategy/test_scoring.py`

**Workspace note**

This workspace is not currently a git repository, so commit steps below should be treated as required hygiene when the project is moved into git. In this workspace, the engineer should complete the code and verification steps and skip the commit command itself.

### Task 1: Add Quant Strategy Settings And Profiles

**Files:**
- Modify: `src/config.py`
- Modify: `.env.example`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_load_settings_reads_symbol_specific_thresholds(tmp_path):
    ...

def test_load_settings_rejects_negative_uncertainty_threshold(tmp_path):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL because the quant profile settings do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class SymbolStrategyProfile:
    symbol: str
    min_edge_threshold: float
    max_uncertainty_threshold: float
    minimum_expected_move_multiple: float
    add_on_edge_multiplier: float
    trend_regime_weight: float
    compression_regime_weight: float
```

```python
@dataclass(frozen=True)
class Settings:
    ...
    strategy_profiles: dict[str, SymbolStrategyProfile]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS with symbol-aware profile parsing and validation.

- [ ] **Step 5: Commit**

```bash
git add src/config.py .env.example tests/test_config.py
git commit -m "feat: add quant strategy profile settings"
```

### Task 2: Add Volatility Normalization Helpers

**Files:**
- Create: `src/strategy/volatility.py`
- Create: `tests/strategy/test_volatility.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_compute_atr_returns_expected_value_for_known_candles():
    ...

def test_build_volatility_state_normalizes_body_efficiency_and_range():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_volatility.py -q`
Expected: FAIL because the volatility module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class VolatilityState:
    short_atr: float
    medium_atr: float
    realized_range: float
    body_efficiency: float
    range_expansion_ratio: float

def compute_atr(candles: list[Candle], *, period: int) -> float:
    ...

def build_volatility_state(*, candles: list[Candle]) -> VolatilityState:
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_volatility.py -q`
Expected: PASS for ATR, range normalization, and body-efficiency calculations.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/volatility.py tests/strategy/test_volatility.py
git commit -m "feat: add volatility normalization helpers"
```

### Task 3: Add Regime Classification

**Files:**
- Create: `src/strategy/regime.py`
- Create: `tests/strategy/test_regime.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_classify_regime_marks_expansion_when_range_and_impulse_expand():
    ...

def test_classify_regime_marks_compression_when_ranges_contract():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_regime.py -q`
Expected: FAIL because the regime classifier does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class RegimeState:
    name: str
    tradable: bool
    continuation_bias: float
    reversion_bias: float
    confidence: float

def classify_regime(*, h1_candles: list[Candle], m30_candles: list[Candle], m15_candles: list[Candle], volatility_state: VolatilityState, gap_decision: GapDecision | None) -> RegimeState:
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_regime.py -q`
Expected: PASS for expansion, compression, and gap-driven classifications.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/regime.py tests/strategy/test_regime.py
git commit -m "feat: add regime classification"
```

### Task 4: Add Scoring Engine And Adaptive Threshold Logic

**Files:**
- Create: `src/strategy/scoring.py`
- Create: `tests/strategy/test_scoring.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_score_sides_returns_bullish_edge_when_location_and_trigger_dominate():
    ...

def test_apply_thresholds_blocks_trade_when_uncertainty_exceeds_limit():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_scoring.py -q`
Expected: FAIL because the scoring engine does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class SideScore:
    location_score: float
    momentum_score: float
    setup_score: float
    trigger_score: float
    gap_score: float
    external_confluence_score: float
    expected_move_score: float
    total: float

@dataclass(frozen=True, slots=True)
class ScoreDecision:
    bullish: SideScore
    bearish: SideScore
    uncertainty_penalty: float
    edge: float
    threshold: float
    is_tradeable: bool

def score_market_sides(... ) -> ScoreDecision:
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_scoring.py -q`
Expected: PASS for side scoring, uncertainty penalties, and adaptive threshold behavior.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/scoring.py tests/strategy/test_scoring.py
git commit -m "feat: add quant scoring engine"
```

### Task 5: Expose Quant Features From Context And Direction

**Files:**
- Modify: `src/strategy/context.py`
- Modify: `src/strategy/direction.py`
- Modify: `tests/strategy/test_context.py`
- Modify: `tests/strategy/test_direction.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_h4_context_exposes_normalized_zone_distances():
    ...

def test_determine_h1_bias_returns_directional_contributions_not_hard_gate_only():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_context.py tests/strategy/test_direction.py -q`
Expected: FAIL because normalized context features and richer directional contribution metadata are not exposed yet.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class DirectionDecision:
    is_valid: bool
    direction: BreakoutDirection | None
    reason: str
    metadata: dict[str, object]
    bullish_contribution: float
    bearish_contribution: float
```

```python
common_metadata["normalized_demand_distance"] = ...
common_metadata["normalized_supply_distance"] = ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_context.py tests/strategy/test_direction.py -q`
Expected: PASS with normalized features and contribution-style outputs.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/context.py src/strategy/direction.py tests/strategy/test_context.py tests/strategy/test_direction.py
git commit -m "refactor: expose quant context and direction features"
```

### Task 6: Quantify M30 Setup And M15 Trigger Quality

**Files:**
- Modify: `src/strategy/setup.py`
- Modify: `src/strategy/trigger.py`
- Modify: `tests/strategy/test_setup.py`
- Modify: `tests/strategy/test_trigger.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_evaluate_m30_setup_returns_setup_quality_score_and_state():
    ...

def test_evaluate_m15_trigger_returns_trigger_quality_and_expected_move_inputs():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_setup.py tests/strategy/test_trigger.py -q`
Expected: FAIL because the setup and trigger modules currently expose mostly binary readiness information.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class SetupDecision:
    is_ready: bool
    reason: str
    setup_state: str
    metadata: dict[str, object]
    quality_score: float
```

```python
@dataclass(frozen=True, slots=True)
class TriggerDecision:
    is_ready: bool
    reason: str
    entry_price: float | None
    invalidation_price: float | None
    metadata: dict[str, object]
    quality_score: float
    expected_move_multiple: float
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_setup.py tests/strategy/test_trigger.py -q`
Expected: PASS with quantified setup and trigger outputs while preserving current readiness behavior.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/setup.py src/strategy/trigger.py tests/strategy/test_setup.py tests/strategy/test_trigger.py
git commit -m "feat: quantify setup and trigger quality"
```

### Task 7: Upgrade The Decision Tree To Use Quant Scoring

**Files:**
- Modify: `src/strategy/decision_tree.py`
- Modify: `tests/strategy/test_decision_tree.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_evaluate_top_down_decision_tree_trades_when_edge_clears_threshold():
    ...

def test_evaluate_top_down_decision_tree_returns_no_trade_when_scores_are_too_close():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_decision_tree.py -q`
Expected: FAIL because the decision tree does not yet use volatility, regime, scoring, or uncertainty thresholds.

- [ ] **Step 3: Write minimal implementation**

```python
volatility_state = build_volatility_state(candles=m15_candles[-20:])
regime_state = classify_regime(...)
score_decision = score_market_sides(...)

if not score_decision.is_tradeable:
    return _no_trade("quant_edge_insufficient", "scoring", edge=score_decision.edge, threshold=score_decision.threshold)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_decision_tree.py -q`
Expected: PASS for tradable edge, blocked edge, and regime-aware no-trade cases.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/decision_tree.py tests/strategy/test_decision_tree.py
git commit -m "feat: wire quant scoring into decision tree"
```

### Task 8: Tighten Campaign Add Logic With Quant Thresholds

**Files:**
- Modify: `src/strategy/management.py`
- Modify: `src/live_trade_loop.py`
- Modify: `tests/strategy/test_management.py`
- Modify: `tests/test_live_trade_loop.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_evaluate_campaign_action_requires_stricter_edge_for_add_on():
    ...

def test_live_loop_logs_campaign_hold_when_add_on_edge_is_insufficient():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_management.py tests/test_live_trade_loop.py -q`
Expected: FAIL because campaign adds currently rely on simpler continuation checks.

- [ ] **Step 3: Write minimal implementation**

```python
def evaluate_campaign_action(..., continuation_edge: float, continuation_threshold: float, ...):
    if latest_trade_r_multiple >= 2.0 and continuation_edge >= continuation_threshold:
        ...
```

```python
action = evaluate_campaign_action(
    ...,
    continuation_edge=score_decision.edge,
    continuation_threshold=score_decision.threshold * settings.strategy_profiles[symbol].add_on_edge_multiplier,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_management.py tests/test_live_trade_loop.py -q`
Expected: PASS with stricter add-on behavior and unchanged safe hold/trail behavior.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/management.py src/live_trade_loop.py tests/strategy/test_management.py tests/test_live_trade_loop.py
git commit -m "feat: tighten campaign add logic with quant thresholds"
```

### Task 9: Full Verification And Config Refresh

**Files:**
- Modify: `.env.example`
- Modify: `tests/test_run.py`
- Verify: `src/config.py`
- Verify: `src/strategy/*.py`

- [ ] **Step 1: Add any remaining regression tests for startup/config wiring**

```python
def test_main_passes_strategy_profiles_into_live_loop():
    ...
```

- [ ] **Step 2: Run the focused regression tests**

Run: `python -m pytest tests/test_run.py tests/test_config.py tests/strategy/test_decision_tree.py tests/strategy/test_management.py tests/test_live_trade_loop.py -q`
Expected: PASS for the live runtime path and quant strategy wiring.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS across the full strategy, runtime, and executor test suite.

- [ ] **Step 4: Update documentation defaults**

```env
XAUUSD_MIN_EDGE_THRESHOLD=...
EURJPY_MIN_EDGE_THRESHOLD=...
XAUUSD_MAX_UNCERTAINTY_THRESHOLD=...
EURJPY_ADD_ON_EDGE_MULTIPLIER=...
```

- [ ] **Step 5: Commit**

```bash
git add .env.example tests/test_run.py
git commit -m "docs: document quant strategy configuration"
```
