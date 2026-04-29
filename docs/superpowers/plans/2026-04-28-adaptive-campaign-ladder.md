# Adaptive Campaign Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade live campaign management to use per-position `R` ladders, adaptive `+1.5R` add triggers, and quant soft/hard block behavior that is auditable for large-account review.

**Architecture:** Keep `src/strategy/management.py` responsible for campaign eligibility, reversal, and stop progression, but replace its coarse trail/add rules with a per-position ladder model. Keep `src/strategy/campaign_add.py` responsible for continuation-quality add evaluation, and update `src/live_trade_loop.py` so the runtime distinguishes hard quant vetoes from reduced-size add permission. All behavior changes should be introduced through TDD and logged with explicit numeric metadata.

**Tech Stack:** Python 3.11/3.12, MetaTrader5 Python package, `pytest`, `python-dotenv`

---

## File Map

**Existing files to modify**

- `src/config.py`
  Add symbol-aware breakeven distance settings and campaign ladder parameters.
- `.env.example`
  Document new ladder and adaptive add settings.
- `src/strategy/management.py`
  Replace coarse `>=1R`/`>=2R` behavior with per-position breakeven and profit-lock ladder logic.
- `src/strategy/campaign_add.py`
  Add adaptive add-trigger math, acceleration bonuses, execution penalties, and quant soft/hard block inputs.
- `src/live_trade_loop.py`
  Distinguish campaign add hard block, soft reduction, and add-ready execution paths; route new ladder settings into management.
- `tests/test_config.py`
  Cover new settings and validation.
- `tests/strategy/test_management.py`
  Cover breakeven, profit-lock ladder, and oldest-vs-newest protection behavior.
- `tests/strategy/test_campaign_add.py`
  Cover adaptive add trigger, adaptive threshold, and quant soft/hard block math.
- `tests/test_live_trade_loop.py`
  Verify runtime logging and routing for reduced-size adds, hard-blocked adds, and ladder-driven trailing.

**New files to create**

- None required unless helper extraction from `management.py` becomes necessary during implementation.

**Workspace note**

This workspace is not currently a git repository, so commit steps below should be treated as required hygiene when the project is moved into git. In this workspace, complete the code and verification steps and skip the commit commands themselves.

### Task 1: Add Ladder Configuration Settings

**Files:**
- Modify: `src/config.py`
- Modify: `.env.example`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_load_settings_reads_symbol_breakeven_distance(tmp_path):
    ...

def test_load_settings_rejects_non_positive_campaign_add_trigger(tmp_path):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL because the ladder settings do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class SymbolStrategyProfile:
    ...
    breakeven_distance: float
    campaign_base_add_trigger_r: float
    campaign_add_trigger_floor_r: float
    campaign_add_trigger_ceiling_r: float
```

```python
_PROFILE_DEFAULTS = {
    "XAUUSD": {
        ...
        "breakeven_distance": 1.5,
        "campaign_base_add_trigger_r": 1.5,
        "campaign_add_trigger_floor_r": 1.25,
        "campaign_add_trigger_ceiling_r": 1.75,
    },
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS with parsed and validated ladder settings.

- [ ] **Step 5: Commit**

```bash
git add src/config.py .env.example tests/test_config.py
git commit -m "feat: add adaptive campaign ladder settings"
```

### Task 2: Add Per-Position Breakeven And Profit-Lock Ladder

**Files:**
- Modify: `src/strategy/management.py`
- Modify: `tests/strategy/test_management.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_campaign_moves_stop_to_entry_at_min_breakeven_trigger():
    ...

def test_campaign_locks_plus_1r_at_plus_2r():
    ...

def test_campaign_locks_more_profit_on_older_trade_than_newer_trade():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_management.py -q`
Expected: FAIL because the current campaign logic only has a coarse trail step.

- [ ] **Step 3: Write minimal implementation**

```python
def _breakeven_trigger_r(*, breakeven_distance: float, risk: float) -> float:
    return min(breakeven_distance, risk) / risk

def _locked_r_multiple(progress_r: float) -> float:
    return max(floor(progress_r) - 1, 0)
```

```python
if progress_r >= breakeven_trigger_r and current_stop_not_protected:
    return CampaignAction(action="trail_all", reason="campaign_breakeven_earned", ...)

if progress_r >= 2.0:
    locked_r = _locked_r_multiple(progress_r)
    return CampaignAction(action="trail_all", reason="campaign_profit_lock_progression", ...)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_management.py -q`
Expected: PASS for breakeven and ladder-based stop progression.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/management.py tests/strategy/test_management.py
git commit -m "feat: add per-position campaign profit ladder"
```

### Task 3: Add Adaptive Campaign Add Trigger Math

**Files:**
- Modify: `src/strategy/campaign_add.py`
- Modify: `tests/strategy/test_campaign_add.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_campaign_add_trigger_relaxes_when_momentum_accelerates():
    ...

def test_campaign_add_trigger_tightens_when_execution_penalty_rises():
    ...

def test_campaign_add_threshold_relaxes_in_protected_campaign():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_campaign_add.py -q`
Expected: FAIL because the add trigger is still effectively static and coarse.

- [ ] **Step 3: Write minimal implementation**

```python
def _adaptive_add_trigger_r(*, base_trigger_r, accel_bonus, execution_penalty, volatility_penalty, floor_r, ceiling_r):
    return clamp(base_trigger_r - accel_bonus + execution_penalty + volatility_penalty, floor_r, ceiling_r)
```

```python
def _momentum_acceleration_score(...):
    return (
        0.35 * m10_delta
        + 0.30 * m5_delta
        + 0.20 * range_expansion
        + 0.15 * body_efficiency
    )
```

```python
def _adaptive_add_threshold(...):
    return clamp(base_threshold - protected_bonus - accel_bonus + execution_penalty + quant_penalty, lower, upper)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_campaign_add.py -q`
Expected: PASS for relaxed/tightened trigger and threshold behavior.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/campaign_add.py tests/strategy/test_campaign_add.py
git commit -m "feat: add adaptive campaign add trigger math"
```

### Task 4: Add Quant Soft Block vs Hard Block Behavior

**Files:**
- Modify: `src/strategy/campaign_add.py`
- Modify: `src/live_trade_loop.py`
- Modify: `tests/strategy/test_campaign_add.py`
- Modify: `tests/test_live_trade_loop.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_campaign_add_is_hard_blocked_when_quant_flips_against_direction():
    ...

def test_campaign_add_is_reduced_not_blocked_when_quant_is_flat_but_campaign_is_protected():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_campaign_add.py tests/test_live_trade_loop.py -q`
Expected: FAIL because runtime only understands a hard campaign quant block today.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class CampaignAddDecision:
    ...
    lot_multiplier: float = 1.0
    quant_state: str = "neutral"
```

```python
if quant_direction_flip or cvar_too_high or drawdown_dampener_active:
    return CampaignAddDecision(is_ready=False, reason="campaign_add_quant_blocked", ...)

if quant_flat_but_protected and add_score_strong:
    return CampaignAddDecision(is_ready=True, reason="campaign_add_quant_reduced", lot_multiplier=0.5, ...)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_campaign_add.py tests/test_live_trade_loop.py -q`
Expected: PASS with explicit reduced-size and blocked behavior.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/campaign_add.py src/live_trade_loop.py tests/strategy/test_campaign_add.py tests/test_live_trade_loop.py
git commit -m "feat: split campaign quant soft and hard blocks"
```

### Task 5: Route Ladder Metadata Through Live Runtime

**Files:**
- Modify: `src/live_trade_loop.py`
- Modify: `tests/test_live_trade_loop.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_live_loop_logs_campaign_breakeven_earned():
    ...

def test_live_loop_logs_campaign_add_quant_reduced():
    ...

def test_live_loop_logs_campaign_profit_lock_progression():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_live_trade_loop.py -q`
Expected: FAIL because runtime logs do not yet distinguish these finer states.

- [ ] **Step 3: Write minimal implementation**

```python
log_fn(
    f"LIVE CAMPAIGN TRAIL {symbol} positions={len(positions)} "
    f"updated={updated_positions} reason={action.reason} new_sl={action.new_stop_loss}"
)
```

```python
log_fn(
    f"LIVE CAMPAIGN ADD {symbol} ... lot={adjusted_add_lot} "
    f"reason={add_decision.reason}"
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_live_trade_loop.py -q`
Expected: PASS with explicit ladder/add/quant runtime logs.

- [ ] **Step 5: Commit**

```bash
git add src/live_trade_loop.py tests/test_live_trade_loop.py
git commit -m "feat: improve campaign ladder runtime logging"
```

### Task 6: Final Verification

**Files:**
- Verify: `src/config.py`
- Verify: `src/strategy/management.py`
- Verify: `src/strategy/campaign_add.py`
- Verify: `src/live_trade_loop.py`
- Verify: `tests/test_config.py`
- Verify: `tests/strategy/test_management.py`
- Verify: `tests/strategy/test_campaign_add.py`
- Verify: `tests/test_live_trade_loop.py`

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest tests/test_config.py tests/strategy/test_management.py tests/strategy/test_campaign_add.py tests/test_live_trade_loop.py -q
```

Expected: PASS for ladder settings, profit locking, adaptive adds, and runtime routing.

- [ ] **Step 2: Run full test suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS with no regressions.

- [ ] **Step 3: Run a safe read-only MT5 diagnostic**

Run a one-off script that:

- logs into MT5
- inspects current bot-owned positions
- computes `R`, current locked `R`, and current add decision
- prints the adaptive trigger, add score, and quant state
- exits without placing or modifying any order

Expected: readable numeric evidence that the ladder and add engine are behaving as designed.

- [ ] **Step 4: Commit**

```bash
git add src/config.py src/strategy/management.py src/strategy/campaign_add.py src/live_trade_loop.py tests/test_config.py tests/strategy/test_management.py tests/strategy/test_campaign_add.py tests/test_live_trade_loop.py
git commit -m "feat: add adaptive campaign ladder management"
```
