# cTrader OpenAPI Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pip-based, read-only cTrader OpenAPI sidecar that enriches the MT5 bot with quote-driven microstructure data while preserving MT5-only fallback safety.

**Architecture:** Keep all Spotware/OpenApiPy and Twisted details isolated in a new `src/ctrader_stub` package. The live MT5 runtime should consume only normalized dataclasses and cached enrichment snapshots, and it must degrade cleanly to MT5-only mode whenever cTrader is disabled, unavailable, or stale.

**Tech Stack:** Python 3.11/3.12, `ctrader-open-api`, `python-dotenv`, `pytest`, MetaTrader5

---

## File Map

**Existing files to modify**

- `requirements.txt`
  Add the Spotware OpenAPI client dependency.
- `src/config.py`
  Add optional cTrader settings to the `Settings` dataclass and `.env` loader.
- `src/live_trade_loop.py`
  Start and query the optional sidecar, and route stale/degraded fallback behavior.
- `src/strategy/execution_cost.py`
  Accept optional cTrader microstructure snapshots and use them to refine execution penalties.
- `tests/test_config.py`
  Cover cTrader enabled/disabled parsing and validation.
- `tests/test_live_trade_loop.py`
  Cover sidecar startup, degradation, and fallback behavior.

**New files to create**

- `src/ctrader_stub/__init__.py`
  Package entrypoint for the sidecar.
- `src/ctrader_stub/models.py`
  Bot-facing normalized dataclasses.
- `src/ctrader_stub/cache.py`
  In-memory cache for quotes, bars, metadata, and freshness state.
- `src/ctrader_stub/indicators.py`
  Derived microstructure calculations.
- `src/ctrader_stub/client.py`
  Low-level Spotware/OpenApiPy connection wrapper with dependency guards.
- `src/ctrader_stub/adapter.py`
  High-level sidecar interface used by the rest of the bot.
- `tests/test_ctrader_stub_models.py`
  Model normalization tests.
- `tests/test_ctrader_stub_cache.py`
  Cache freshness and stale-state tests.
- `tests/test_ctrader_stub_indicators.py`
  Microstructure indicator math tests.
- `tests/test_ctrader_stub_adapter.py`
  Adapter lifecycle and fallback tests.

**Workspace note**

This workspace is not currently a git repository, so commit steps below are included as required hygiene but will be skipped locally unless git is initialized later.

### Task 1: Add Dependency And Optional Settings

**Files:**
- Modify: `requirements.txt`
- Modify: `src/config.py`
- Modify: `.env.example`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_load_settings_allows_ctrader_disabled_without_credentials(tmp_path):
    ...

def test_load_settings_requires_ctrader_credentials_when_enabled(tmp_path):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL because `Settings` does not yet expose cTrader keys or validation.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class Settings:
    ...
    ctrader_enabled: bool = False
    ctrader_environment: str = "live"
    ctrader_client_id: str = ""
    ctrader_client_secret: str = ""
    ctrader_account_id: str = ""
```

```text
requirements.txt
ctrader-open-api
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS with valid enabled/disabled cTrader parsing.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/config.py .env.example tests/test_config.py
git commit -m "feat: add ctrader sidecar settings"
```

### Task 2: Create Normalized Models And Cache

**Files:**
- Create: `src/ctrader_stub/__init__.py`
- Create: `src/ctrader_stub/models.py`
- Create: `src/ctrader_stub/cache.py`
- Create: `tests/test_ctrader_stub_models.py`
- Create: `tests/test_ctrader_stub_cache.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_quote_model_computes_mid_and_spread():
    ...

def test_cache_marks_symbol_stale_after_timeout():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ctrader_stub_models.py tests/test_ctrader_stub_cache.py -q`
Expected: FAIL because the package and cache do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class CTraderQuote:
    symbol: str
    bid: float
    ask: float
    timestamp: datetime

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0
```

```python
class CTraderCache:
    def update_quote(self, quote: CTraderQuote) -> None:
        ...

    def is_symbol_stale(self, symbol: str, *, now: datetime, max_age_seconds: int) -> bool:
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ctrader_stub_models.py tests/test_ctrader_stub_cache.py -q`
Expected: PASS for quote normalization and stale-state tracking.

- [ ] **Step 5: Commit**

```bash
git add src/ctrader_stub/__init__.py src/ctrader_stub/models.py src/ctrader_stub/cache.py tests/test_ctrader_stub_models.py tests/test_ctrader_stub_cache.py
git commit -m "feat: add ctrader sidecar models and cache"
```

### Task 3: Implement Microstructure Indicators

**Files:**
- Create: `src/ctrader_stub/indicators.py`
- Create: `tests/test_ctrader_stub_indicators.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_compute_microstructure_snapshot_includes_spread_percentile():
    ...

def test_compute_microstructure_snapshot_flags_spread_shock():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ctrader_stub_indicators.py -q`
Expected: FAIL because indicator functions do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def build_microstructure_snapshot(*, symbol: str, quotes: list[CTraderQuote], bars: list[CTraderBar] | None = None) -> CTraderMicrostructureSnapshot:
    ...
```

The first snapshot should compute:

- spread mean
- spread percentile
- spread shock flag
- quote velocity
- realized micro-volatility
- optional session VWAP/opening-range placeholders

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ctrader_stub_indicators.py -q`
Expected: PASS with deterministic indicator outputs from synthetic quote streams.

- [ ] **Step 5: Commit**

```bash
git add src/ctrader_stub/indicators.py tests/test_ctrader_stub_indicators.py
git commit -m "feat: add ctrader microstructure indicators"
```

### Task 4: Build The Low-Level Client And High-Level Adapter

**Files:**
- Create: `src/ctrader_stub/client.py`
- Create: `src/ctrader_stub/adapter.py`
- Create: `tests/test_ctrader_stub_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_adapter_reports_disabled_when_settings_disable_ctrader():
    ...

def test_adapter_degrades_gracefully_when_dependency_missing(monkeypatch):
    ...

def test_adapter_reads_from_cache_without_exposing_protocol_types():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ctrader_stub_adapter.py -q`
Expected: FAIL because the adapter and guarded client do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
class CTraderSidecarAdapter:
    def connect(self) -> None:
        ...

    def warmup(self, symbols: list[str]) -> None:
        ...

    def get_quote(self, symbol: str) -> CTraderQuote | None:
        ...

    def get_microstructure(self, symbol: str) -> CTraderMicrostructureSnapshot | None:
        ...
```

The low-level client should:

- import Spotware dependencies lazily
- raise a controlled sidecar error if the dependency is missing
- isolate all protocol-specific structures from the rest of the bot

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ctrader_stub_adapter.py -q`
Expected: PASS for disabled mode, degraded mode, and normalized cache reads.

- [ ] **Step 5: Commit**

```bash
git add src/ctrader_stub/client.py src/ctrader_stub/adapter.py tests/test_ctrader_stub_adapter.py
git commit -m "feat: add ctrader sidecar adapter"
```

### Task 5: Integrate Sidecar Into Runtime Fallback Flow

**Files:**
- Modify: `src/live_trade_loop.py`
- Modify: `tests/test_live_trade_loop.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_live_loop_runs_mt5_only_when_ctrader_disabled():
    ...

def test_live_loop_logs_and_continues_when_ctrader_sidecar_is_degraded():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_live_trade_loop.py -q`
Expected: FAIL because the runtime does not yet know about the cTrader sidecar.

- [ ] **Step 3: Write minimal implementation**

```python
if settings.ctrader_enabled:
    sidecar = build_ctrader_sidecar(settings)
    try:
        sidecar.connect()
        sidecar.warmup([settings.trading_symbol])
    except Exception:
        logger.exception("CTRADER SIDECAR DEGRADED")
        sidecar = None
```

The runtime must:

- never crash because the sidecar is unavailable
- expose sidecar health in logs
- keep MT5 execution behavior unchanged when no sidecar data is present

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_live_trade_loop.py -q`
Expected: PASS for disabled and degraded cTrader runtime behavior.

- [ ] **Step 5: Commit**

```bash
git add src/live_trade_loop.py tests/test_live_trade_loop.py
git commit -m "feat: add ctrader sidecar runtime fallback"
```

### Task 6: Feed Microstructure Enrichment Into Execution Cost

**Files:**
- Modify: `src/strategy/execution_cost.py`
- Modify: `src/live_trade_loop.py`
- Create or Modify: `tests/strategy/test_execution_cost.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_execution_cost_uses_ctrader_spread_percentile_when_available():
    ...

def test_execution_cost_falls_back_to_mt5_only_inputs_when_sidecar_missing():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/strategy/test_execution_cost.py -q`
Expected: FAIL because the execution-cost engine does not yet accept cTrader enrichment.

- [ ] **Step 3: Write minimal implementation**

```python
def assess_market_order_execution(..., ctrader_microstructure: CTraderMicrostructureSnapshot | None = None):
    ...
```

Use the cTrader snapshot to refine:

- spread pressure
- slippage pressure
- liquidity stress proxy
- execution penalty summary

If the snapshot is missing or stale, retain the current MT5-only path.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/strategy/test_execution_cost.py -q`
Expected: PASS for enriched and fallback execution-cost behavior.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/execution_cost.py src/live_trade_loop.py tests/strategy/test_execution_cost.py
git commit -m "feat: enrich execution cost with ctrader microstructure"
```

### Task 7: Run Full Verification And Smoke Checks

**Files:**
- Modify: any touched files if verification reveals issues

- [ ] **Step 1: Run targeted sidecar and runtime tests**

Run: `python -m pytest tests/test_ctrader_stub_models.py tests/test_ctrader_stub_cache.py tests/test_ctrader_stub_indicators.py tests/test_ctrader_stub_adapter.py tests/test_live_trade_loop.py tests/strategy/test_execution_cost.py -q`
Expected: PASS across the sidecar and integration surface.

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS with no regressions in MT5-only behavior.

- [ ] **Step 3: Perform a safe startup smoke check with cTrader disabled**

Run: `python run.py`
Expected: normal MT5 startup behavior identical to current runtime.

- [ ] **Step 4: Perform a bounded smoke check with cTrader enabled only if credentials and dependency are ready**

Run: `python run.py`
Expected: sidecar startup logs, no runtime crash, and clean MT5 fallback if cTrader is unavailable.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/config.py src/ctrader_stub src/live_trade_loop.py src/strategy/execution_cost.py tests
git commit -m "feat: add ctrader openapi enrichment sidecar"
```
