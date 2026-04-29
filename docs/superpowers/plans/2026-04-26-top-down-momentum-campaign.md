# Top-Down Momentum Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bot's loose fallback logic with a live-only top-down `D1 -> H4 -> H1 -> M30 -> M15` decision tree that can open, add to, and manage synchronized momentum campaigns on MT5.

**Architecture:** The runtime becomes live-only and delegates all market reasoning to focused strategy modules: context, direction, setup, trigger, patterns, decision tree, and campaign management. `src/live_trade_loop.py` becomes a thin orchestrator that asks the strategy layer what to do next and asks `src/trade_executor.py` to execute or manage only one safe campaign action per cycle.

**Tech Stack:** Python 3.12, MetaTrader5 Python package, `pytest`, `python-dotenv`

---

## File Map

**Existing files to modify**

- `run.py`
  Remove `dry_run` and `test_trade` runtime branching so the entrypoint always launches the live loop.
- `src/config.py`
  Simplify settings to a live-only runtime and add campaign-management configuration.
- `src/market_data.py`
  Add reusable fetch helpers and any small candle utilities needed by the new decision tree.
- `src/live_trade_loop.py`
  Replace the current break-retest-plus-fallback logic with decision-tree evaluation and campaign management.
- `src/trade_executor.py`
  Add bot-position ownership helpers, lot normalization, and stop modification support.
- `tests/test_config.py`
  Update for live-only settings.
- `tests/test_run.py`
  Update for the single runtime path.
- `tests/test_live_trade_loop.py`
  Replace current fallback-based expectations with decision-tree/campaign expectations.
- `tests/test_trade_executor.py`
  Extend to cover stop updates, lot normalization, and ownership behavior.

**New files to create**

- `src/strategy/context.py`
  Build `D1` and `H4` context, including daily extremes, previous-session levels, demand/supply candidates, and MT5-based volume-profile reference levels.
- `src/strategy/direction.py`
  Derive `H1` directional bias from structure, reaction, and context alignment.
- `src/strategy/setup.py`
  Determine `M30` setup readiness.
- `src/strategy/trigger.py`
  Determine `M15` trigger readiness.
- `src/strategy/patterns.py`
  Detect `three-drives` and related confluence patterns.
- `src/strategy/decision_tree.py`
  Coordinate the top-down evaluation and return either a trade plan or a no-trade result with the failed node.
- `src/strategy/management.py`
  Coordinate campaign adds, synchronized trailing, exposure checks, and reversal exits.
- `tests/strategy/test_context.py`
- `tests/strategy/test_direction.py`
- `tests/strategy/test_setup.py`
- `tests/strategy/test_trigger.py`
- `tests/strategy/test_patterns.py`
- `tests/strategy/test_decision_tree.py`
- `tests/strategy/test_management.py`

**Files to remove after migration**

- `src/dry_run_loop.py`
- `tests/test_dry_run_loop.py`
- `src/strategy/m15_fallback.py`
- `tests/strategy/test_m15_fallback.py`

**Workspace note**

This workspace is not currently a git repository, so commit steps below should be treated as required hygiene when the project is moved into git. In this workspace, the engineer should complete the code and verification steps and skip the commit command itself.

### Task 1: Make The Runtime Live-Only

**Files:**
- Modify: `run.py`
- Modify: `src/config.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_run.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_load_settings_no_longer_requires_bot_mode_flags(tmp_path):
    ...

def test_main_always_starts_live_signal_loop(monkeypatch):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py tests/test_run.py -q`
Expected: FAIL because `Settings` still expects `BOT_MODE`, `ENABLE_TEST_TRADE`, and `LIVE_TRADING_ENABLED`, and `run.py` still branches by mode.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class Settings:
    mt5_terminal_path: str
    hfm_login: int
    hfm_password: str
    hfm_server: str
    trading_symbol: str
    mt5_startup_wait_seconds: int
    loop_poll_seconds: int
    max_live_loops: int | None
    default_trade_lot: float
    add_on_lot_increment: float
    campaign_max_exposure_pct: float
```

```python
def main() -> int:
    configure_logging()
    settings = load_settings()
    mt5 = create_mt5_module()
    ...
    run_live_signal_loop(...)
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py tests/test_run.py -q`
Expected: PASS with only the live runtime path remaining.

- [ ] **Step 5: Commit**

```bash
git add run.py src/config.py tests/test_config.py tests/test_run.py
git commit -m "refactor: make runtime live-only"
```

### Task 2: Add Shared Market Context Builders

**Files:**
- Modify: `src/market_data.py`
- Create: `src/strategy/context.py`
- Create: `tests/strategy/test_context.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_daily_context_returns_high_low_and_range_position():
    ...

def test_build_h4_context_returns_previous_session_levels_and_volume_profile_markers():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_context.py -q`
Expected: FAIL because the context builders do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class DailyContext:
    daily_high: float
    daily_low: float
    current_price: float
    range_position: float
    objective_high: float
    objective_low: float

@dataclass(frozen=True, slots=True)
class H4Context:
    previous_session_high: float
    previous_session_low: float
    demand_zones: tuple[tuple[float, float], ...]
    supply_zones: tuple[tuple[float, float], ...]
    volume_profile_levels: tuple[float, ...]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_context.py -q`
Expected: PASS for daily context, previous-session levels, and volume-profile references.

- [ ] **Step 5: Commit**

```bash
git add src/market_data.py src/strategy/context.py tests/strategy/test_context.py
git commit -m "feat: add top-down context builders"
```

### Task 3: Add H1 Directional Bias Logic

**Files:**
- Create: `src/strategy/direction.py`
- Create: `tests/strategy/test_direction.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_determine_h1_bias_returns_bullish_when_context_and_reaction_align():
    ...

def test_determine_h1_bias_returns_no_trade_when_context_conflicts():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_direction.py -q`
Expected: FAIL because the H1 bias module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class DirectionDecision:
    is_valid: bool
    direction: BreakoutDirection | None
    reason: str
    metadata: dict[str, object]

def determine_h1_bias(*, h1_candles, daily_context, h4_context) -> DirectionDecision:
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_direction.py -q`
Expected: PASS for aligned bullish/bearish cases and blocked-context cases.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/direction.py tests/strategy/test_direction.py
git commit -m "feat: add h1 direction bias logic"
```

### Task 4: Add M30 Setup Refinement

**Files:**
- Create: `src/strategy/setup.py`
- Create: `tests/strategy/test_setup.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_evaluate_m30_setup_marks_rejection_as_ready():
    ...

def test_evaluate_m30_setup_rejects_dirty_midrange_location():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_setup.py -q`
Expected: FAIL because the M30 setup module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class SetupDecision:
    is_ready: bool
    reason: str
    setup_state: str
    metadata: dict[str, object]

def evaluate_m30_setup(*, m30_candles, direction_decision, h4_context) -> SetupDecision:
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_setup.py -q`
Expected: PASS for approach, rejection, breakout-away, and dirty-location rejection cases.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/setup.py tests/strategy/test_setup.py
git commit -m "feat: add m30 setup refinement"
```

### Task 5: Add M15 Trigger Validation

**Files:**
- Create: `src/strategy/trigger.py`
- Create: `tests/strategy/test_trigger.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_evaluate_m15_trigger_accepts_velocity_and_candle_intent():
    ...

def test_evaluate_m15_trigger_blocks_when_velocity_is_missing():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_trigger.py -q`
Expected: FAIL because the M15 trigger module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class TriggerDecision:
    is_ready: bool
    reason: str
    entry_price: float | None
    invalidation_price: float | None
    metadata: dict[str, object]

def evaluate_m15_trigger(*, m15_candles, setup_decision, direction_decision) -> TriggerDecision:
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_trigger.py -q`
Expected: PASS for valid trigger and no-trigger conditions.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/trigger.py tests/strategy/test_trigger.py
git commit -m "feat: add m15 trigger validation"
```

### Task 6: Add Pattern Confluence Detection

**Files:**
- Create: `src/strategy/patterns.py`
- Create: `tests/strategy/test_patterns.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_detect_three_drives_returns_confluence_near_key_level():
    ...

def test_detect_three_drives_returns_absent_when_swings_are_not_clean():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_patterns.py -q`
Expected: FAIL because the patterns module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class PatternDecision:
    is_present: bool
    reason: str
    confluence_score: int
    metadata: dict[str, object]

def detect_three_drives(*, candles, reference_levels) -> PatternDecision:
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_patterns.py -q`
Expected: PASS for clean and unclean swing-pattern cases.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/patterns.py tests/strategy/test_patterns.py
git commit -m "feat: add three-drives confluence detection"
```

### Task 7: Build The Top-Down Decision Tree

**Files:**
- Create: `src/strategy/decision_tree.py`
- Modify: `src/strategy/__init__.py`
- Create: `tests/strategy/test_decision_tree.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_decision_tree_returns_trade_plan_when_all_nodes_align():
    ...

def test_decision_tree_returns_exact_failed_node_reason():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_decision_tree.py -q`
Expected: FAIL because the decision tree module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class TopDownTradePlan:
    is_trade: bool
    direction: BreakoutDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    objective_price: float
    reason: str
    metadata: dict[str, object]

@dataclass(frozen=True, slots=True)
class TopDownNoTrade:
    is_trade: bool
    reason: str
    failed_node: str
    metadata: dict[str, object]

def evaluate_top_down_decision_tree(...):
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_decision_tree.py -q`
Expected: PASS for valid trade plans and exact failed-node reporting.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/decision_tree.py src/strategy/__init__.py tests/strategy/test_decision_tree.py
git commit -m "feat: add top-down decision tree"
```

### Task 8: Add Campaign Management Logic

**Files:**
- Create: `src/strategy/management.py`
- Create: `tests/strategy/test_management.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_management_allows_add_after_latest_trade_reaches_two_r():
    ...

def test_management_tightens_all_positions_when_latest_trade_reaches_one_r():
    ...

def test_management_blocks_add_when_campaign_exposure_exceeds_limit():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_management.py -q`
Expected: FAIL because the campaign-management module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class CampaignAction:
    action: str
    reason: str
    new_stop_loss: float | None = None
    add_lot: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

def evaluate_campaign_action(
    *,
    positions,
    current_price,
    direction,
    latest_trade_r_multiple,
    default_lot,
    add_on_lot_increment,
    max_exposure_pct,
    margin_snapshot,
    reversal_confirmed,
):
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_management.py -q`
Expected: PASS for add eligibility, synchronized trailing, exposure-cap enforcement, and reversal exits.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/management.py tests/strategy/test_management.py
git commit -m "feat: add momentum campaign management"
```

### Task 9: Expand Trade Executor For Campaign Ownership And Stop Updates

**Files:**
- Modify: `src/trade_executor.py`
- Modify: `tests/test_trade_executor.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_update_position_stop_loss_builds_modify_request():
    ...

def test_normalize_lot_size_uses_symbol_volume_step_and_limits():
    ...

def test_list_bot_positions_filters_by_comment_prefix():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trade_executor.py -q`
Expected: FAIL because the executor does not yet support stop updates, ownership filtering, or lot normalization.

- [ ] **Step 3: Write minimal implementation**

```python
def normalize_lot_size(symbol_info, requested_lot):
    ...

def build_modify_sl_tp_request(mt5_module, position, stop_loss, take_profit=None):
    ...

class TradeExecutor:
    def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
        ...

    def update_position_stop_loss(self, position, stop_loss, take_profit=None):
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trade_executor.py -q`
Expected: PASS for stop-modification requests, ownership filtering, and lot normalization.

- [ ] **Step 5: Commit**

```bash
git add src/trade_executor.py tests/test_trade_executor.py
git commit -m "feat: add campaign execution helpers"
```

### Task 10: Integrate The Decision Tree And Campaign Manager Into The Live Loop

**Files:**
- Modify: `src/live_trade_loop.py`
- Modify: `tests/test_live_trade_loop.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_live_loop_opens_entry_when_decision_tree_returns_trade():
    ...

def test_live_loop_trails_existing_campaign_before_looking_for_new_entries():
    ...

def test_live_loop_adds_position_when_management_returns_add_action():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_live_trade_loop.py -q`
Expected: FAIL because the live loop still uses `build_live_strategy_input`, `evaluate_break_and_retest_setup`, and the old top-down fallback.

- [ ] **Step 3: Write minimal implementation**

```python
def run_live_signal_loop(...):
    campaign_positions = executor.list_bot_positions(symbol)
    if campaign_positions:
        action = evaluate_campaign_action(...)
        ...
    else:
        result = evaluate_top_down_decision_tree(...)
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_live_trade_loop.py -q`
Expected: PASS for entry, trail, add-on, and no-trade loop behavior.

- [ ] **Step 5: Commit**

```bash
git add src/live_trade_loop.py tests/test_live_trade_loop.py
git commit -m "feat: wire live loop to decision tree and campaign manager"
```

### Task 11: Remove Dry-Run And Old Fallback Runtime Debris

**Files:**
- Delete: `src/dry_run_loop.py`
- Delete: `tests/test_dry_run_loop.py`
- Delete: `src/strategy/m15_fallback.py`
- Delete: `tests/strategy/test_m15_fallback.py`
- Modify: `run.py`
- Modify: any tests or imports that still reference removed files

- [ ] **Step 1: Write or adjust the failing tests**

```python
def test_runtime_has_no_dry_run_imports_left():
    ...
```

- [ ] **Step 2: Run the affected tests to verify they fail**

Run: `python -m pytest tests/test_run.py tests/test_live_trade_loop.py -q`
Expected: FAIL because imports and expectations still reference the removed dry-run path.

- [ ] **Step 3: Write minimal implementation**

```python
# remove dry-run imports and branches entirely
from src.live_trade_loop import run_live_signal_loop
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run.py tests/test_live_trade_loop.py -q`
Expected: PASS with only the live runtime remaining.

- [ ] **Step 5: Commit**

```bash
git add run.py src/live_trade_loop.py tests/test_run.py tests/test_live_trade_loop.py
git rm src/dry_run_loop.py tests/test_dry_run_loop.py src/strategy/m15_fallback.py tests/strategy/test_m15_fallback.py
git commit -m "refactor: remove dry-run runtime path"
```

### Task 12: Full Verification And Demo-Account Smoke Check

**Files:**
- Modify: `.env.example` (if new configuration keys need documenting)
- Modify: `docs/superpowers/specs/2026-04-26-top-down-momentum-campaign-design.md` (only if implementation clarifications are needed)

- [ ] **Step 1: Run the full strategy-focused suite**

Run: `python -m pytest tests/strategy tests/test_trade_executor.py tests/test_live_trade_loop.py -q`
Expected: All new decision-tree, campaign-management, and execution tests pass together.

- [ ] **Step 2: Run the full project suite**

Run: `python -m pytest -q`
Expected: PASS for the full project without any dry-run runtime tests remaining.

- [ ] **Step 3: Run a bounded live/demo smoke check**

Run: `python run.py`
Expected: The bot launches MT5, logs into the configured HFM account, and either opens/manages a live demo campaign action or logs a precise live no-trade reason from the new decision tree.

- [ ] **Step 4: Review logs against the spec**

Confirm the live runtime now reflects:
- the `D1 -> H4 -> H1 -> M30 -> M15` tree
- exact failed-node no-trade reasons
- synchronized campaign trailing
- add-on gating at `+2R`
- `10%` campaign exposure cap

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "test: verify top-down momentum campaign runtime"
```
