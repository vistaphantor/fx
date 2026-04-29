# Break And Retest Scalping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic strategy engine for `XAUUSD` and `EURJPY` that detects `M5` break-and-retest scalping setups using `H1`/`M15` structure, first-retest-only logic, session filters, and `1:3` structure-based risk targets.

**Architecture:** The strategy layer sits above the existing MT5 execution scaffold and is split into focused modules for candle ingestion, zone construction, breakout detection, retest tracking, confirmation, risk calculation, and orchestration. Each module is verified with synthetic-candle tests so the full engine can be trusted without needing a live terminal.

**Tech Stack:** Python 3.12, `pytest`, existing MT5 integration code

---

### Task 1: Strategy Package Scaffolding

**Files:**
- Create: `src/strategy/__init__.py`
- Create: `tests/strategy/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
def test_strategy_package_imports():
    import src.strategy
    assert src.strategy is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy -q`
Expected: FAIL because `src.strategy` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create the strategy package markers.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy -q`
Expected: PASS for the package import test.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/__init__.py tests/strategy/__init__.py
git commit -m "chore: scaffold strategy package"
```

### Task 2: Session Filter Module

**Files:**
- Create: `src/strategy/session_filter.py`
- Create: `tests/strategy/test_session_filter.py`

- [ ] **Step 1: Write the failing test**

```python
def test_is_allowed_session_accepts_london_and_new_york():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_session_filter.py -q`
Expected: FAIL because the session filter helper does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement deterministic London/New York session checks using timezone-aware UTC datetimes and explicit session boundaries.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_session_filter.py -q`
Expected: PASS for allowed and rejected timestamps.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/session_filter.py tests/strategy/test_session_filter.py
git commit -m "feat: add strategy session filters"
```

### Task 3: Market Data Models

**Files:**
- Create: `src/market_data.py`
- Create: `tests/test_market_data.py`

- [ ] **Step 1: Write the failing test**

```python
def test_candle_from_mt5_rate_maps_required_fields():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_market_data.py -q`
Expected: FAIL because candle models and converters do not exist.

- [ ] **Step 3: Write minimal implementation**

Add candle dataclasses, timeframe labels, and a thin MT5-rate conversion helper that normalizes timestamps and OHLCV fields.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_market_data.py -q`
Expected: PASS for candle normalization behavior.

- [ ] **Step 5: Commit**

```bash
git add src/market_data.py tests/test_market_data.py
git commit -m "feat: add market data models"
```

### Task 4: Zone Construction

**Files:**
- Create: `src/strategy/zones.py`
- Create: `tests/strategy/test_zones.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_zones_creates_h1_zone_and_m15_refinement():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_zones.py -q`
Expected: FAIL because zone builders do not exist.

- [ ] **Step 3: Write minimal implementation**

Implement deterministic zone extraction from synthetic `H1` swing points and `M15` refinement bounds with small, test-friendly dataclasses.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_zones.py -q`
Expected: PASS for major-zone and refinement-zone construction.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/zones.py tests/strategy/test_zones.py
git commit -m "feat: add structure zone builder"
```

### Task 5: Breakout Detection

**Files:**
- Create: `src/strategy/breakout.py`
- Create: `tests/strategy/test_breakout.py`

- [ ] **Step 1: Write the failing test**

```python
def test_detect_breakout_requires_m5_close_beyond_zone():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_breakout.py -q`
Expected: FAIL because breakout helpers do not exist.

- [ ] **Step 3: Write minimal implementation**

Implement breakout detection that rejects wick-only probes and returns structured breakout metadata.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_breakout.py -q`
Expected: PASS for valid breakout and wick-only rejection cases.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/breakout.py tests/strategy/test_breakout.py
git commit -m "feat: add breakout detection"
```

### Task 6: First-Retest Tracking

**Files:**
- Create: `src/strategy/retest.py`
- Create: `tests/strategy/test_retest.py`

- [ ] **Step 1: Write the failing test**

```python
def test_retest_tracker_allows_only_first_touch_after_breakout():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_retest.py -q`
Expected: FAIL because the retest tracker does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement breakout-state tracking, first-touch recognition, and invalidation rules for late or degraded retests.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_retest.py -q`
Expected: PASS for first-touch acceptance and second-touch rejection.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/retest.py tests/strategy/test_retest.py
git commit -m "feat: add first-retest tracking"
```

### Task 7: Entry Confirmation

**Files:**
- Create: `src/strategy/confirmation.py`
- Create: `tests/strategy/test_confirmation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_confirm_entry_accepts_rejection_candle_or_micro_break():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_confirmation.py -q`
Expected: FAIL because confirmation helpers do not exist.

- [ ] **Step 3: Write minimal implementation**

Implement rejection-candle validation and micro-structure-break validation with explicit result objects describing the trigger type.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_confirmation.py -q`
Expected: PASS for both supported confirmation paths.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/confirmation.py tests/strategy/test_confirmation.py
git commit -m "feat: add retest confirmation rules"
```

### Task 8: Risk Model

**Files:**
- Create: `src/strategy/risk.py`
- Create: `tests/strategy/test_risk.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_trade_levels_sets_stop_from_structure_and_target_at_three_r():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_risk.py -q`
Expected: FAIL because the risk helpers do not exist.

- [ ] **Step 3: Write minimal implementation**

Implement trade-level calculations that place stop loss beyond the retest structure and set take profit at exactly `3R`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_risk.py -q`
Expected: PASS for long and short risk calculations.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/risk.py tests/strategy/test_risk.py
git commit -m "feat: add strategy risk model"
```

### Task 9: Strategy Engine Orchestration

**Files:**
- Create: `src/strategy/engine.py`
- Create: `tests/strategy/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_engine_returns_trade_plan_for_valid_break_and_retest_setup():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/strategy/test_engine.py -q`
Expected: FAIL because the strategy engine does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement a thin coordinator that consumes zones, breakout state, retest status, confirmation, session filter, and risk outputs to emit either a structured trade plan or a structured no-trade result.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/strategy/test_engine.py -q`
Expected: PASS for a valid trade path and representative no-trade cases.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/engine.py tests/strategy/test_engine.py
git commit -m "feat: add break and retest strategy engine"
```

### Task 10: End-To-End Local Verification

**Files:**
- Modify: `docs/superpowers/specs/2026-04-04-break-retest-scalping-design.md` (only if clarification notes are needed)

- [ ] **Step 1: Run the full strategy test suite**

Run: `python -m pytest tests/strategy tests/test_market_data.py -q`
Expected: All strategy-focused tests pass.

- [ ] **Step 2: Run the full project test suite**

Run: `python -m pytest -q`
Expected: All existing MT5 skeleton tests and new strategy tests pass together.

- [ ] **Step 3: Review engine outputs against the spec**

Confirm the engine still enforces:
- `H1`/`M15` structure
- `M5` breakout close rule
- first retest only
- London/New York sessions only
- rejection candle or micro-break confirmation
- `1:3` risk target

- [ ] **Step 4: Capture any known follow-up gaps**

Document future filters such as spread thresholds, symbol-specific tuning, and backtest harness work.

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "test: verify break and retest strategy modules"
```
