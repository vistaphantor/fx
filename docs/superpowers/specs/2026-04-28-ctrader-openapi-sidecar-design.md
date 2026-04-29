# cTrader OpenAPI Sidecar Design

## Goal

Add a local, pip-based, read-only cTrader sidecar that enriches the MT5 trading bot with higher-quality quote, spread, session, and microstructure features.

The sidecar should:

- connect through Spotware's official Python Open API client
- run locally inside the bot process space, not as a URI service
- expose normalized Python data objects rather than Open API protocol details
- improve execution-cost estimation, quant feature quality, and campaign-add timing
- fail safely so MT5 trading continues if cTrader is unavailable

## Scope

In scope for `v1`:

- local Python adapter around `ctrader-open-api`
- environment-driven cTrader credentials and account selection
- quote subscription and in-memory quote history
- symbol metadata retrieval
- recent bar retrieval for configured symbols and timeframes
- derived microstructure indicators from quote and bar history
- optional account snapshot retrieval
- integration points for execution-cost and quant enrichment

Out of scope for `v1`:

- direct cTrader execution
- using cTrader as the primary trading venue
- hard dependency of the MT5 runtime on cTrader availability
- guaranteed Level 2 / full order-book support
- FIX integration
- database/dashboard work

## Design Principles

The cTrader layer should behave like an optional intelligence sidecar, not a second trading engine.

The implementation should preserve these invariants:

1. MT5 remains the source of truth for execution and active position state
2. cTrader data must never crash or block the live MT5 runtime
3. Twisted/Open API protocol details must remain isolated inside the sidecar package
4. all bot-facing cTrader data must be normalized into plain Python data classes
5. order-book enrichment must remain optional so `v1` does not promise unsupported data

## Architecture

The sidecar will live in a new package:

- `src/ctrader_stub/client.py`
- `src/ctrader_stub/models.py`
- `src/ctrader_stub/cache.py`
- `src/ctrader_stub/indicators.py`
- `src/ctrader_stub/adapter.py`
- `src/ctrader_stub/__init__.py`

### `client.py`

Owns the Spotware/OpenApiPy lifecycle:

- authentication
- session setup
- quote subscriptions
- bar/history requests
- reconnect handling
- low-level event dispatch

This file is the only place that should know about Open API wire details and Twisted integration.

### `models.py`

Defines normalized bot-facing data objects:

- `CTraderQuote`
- `CTraderBar`
- `CTraderSymbolMeta`
- `CTraderAccountSnapshot`
- `CTraderMicrostructureSnapshot`
- `CTraderOrderBookSnapshot`

`CTraderOrderBookSnapshot` should exist in the model layer even if `v1` leaves it empty or unavailable. This gives the bot a stable forward-compatible interface for later depth integration.

### `cache.py`

Maintains in-memory state for:

- latest quote per symbol
- rolling quote history per symbol
- latest derived microstructure snapshot per symbol
- recent cached bars per symbol and timeframe
- latest symbol metadata
- optional latest account snapshot

The cache should track timestamps so stale-data checks are explicit and auditable.

### `indicators.py`

Computes quote-driven and bar-driven enrichments:

- rolling spread mean and percentile
- spread shock flag
- micro realized volatility
- quote velocity
- quote jump intensity
- session VWAP
- opening-range width and position
- liquidity stress proxy from spread and quote behavior

These indicators are the guaranteed `v1` enrichment set.

### `adapter.py`

Exposes the simple bot-facing interface:

- `connect()`
- `close()`
- `warmup(symbols)`
- `get_quote(symbol)`
- `get_recent_bars(symbol, timeframe, count)`
- `get_symbol_meta(symbol)`
- `get_microstructure(symbol)`
- `get_account_snapshot()`
- `get_order_book(symbol)`
- `is_healthy()`

`get_order_book(symbol)` should return `None` in `v1` unless depth becomes available.

## Runtime Model

The sidecar should be optional.

If `CTRADER_ENABLED=false`:

- the adapter is not started
- the bot behaves exactly as it does today

If `CTRADER_ENABLED=true`:

- the adapter starts during runtime boot
- it authenticates and warms quote and metadata caches for the configured symbol set
- it exposes enrichment snapshots to the live MT5 loop

The live runtime should treat cTrader as best-effort:

- if authentication fails, log and continue MT5-only
- if subscriptions fail, log and continue MT5-only
- if data becomes stale, mark the sidecar unhealthy and continue MT5-only

No cTrader failure should terminate `run.py`.

## Configuration Contract

The environment contract is:

- `CTRADER_ENABLED`
- `CTRADER_ENVIRONMENT`
- `CTRADER_CLIENT_ID`
- `CTRADER_CLIENT_SECRET`
- `CTRADER_ACCOUNT_ID`

Expected behavior:

- `CTRADER_ENABLED=false` means all other keys are ignored
- `CTRADER_ENABLED=true` requires non-empty credentials
- `CTRADER_ACCOUNT_ID` may remain optional if the first milestone only needs quote and symbol data, but account-specific features should guard accordingly

The `Settings` model should eventually grow optional cTrader fields so runtime code can read them without direct `.env` access.

## Data Contract

### Guaranteed `v1` Data

Per symbol, the bot should be able to read:

- latest bid
- latest ask
- latest mid
- latest spread
- quote timestamp
- recent bars
- symbol metadata such as pip size, tick size, and volume step

### Derived `v1` Indicators

Per symbol, the bot should be able to read:

- spread percentile
- rolling micro-volatility
- quote velocity
- spread shock flag
- session VWAP
- opening-range statistics
- liquidity stress proxy

### Optional Future Data

Per symbol, the adapter may later provide:

- top-of-book sizes
- bid/ask imbalance
- aggregated order-book ladder summary

The bot must not assume these values exist in `v1`.

## Integration Points

The first consumers should be:

- `src/strategy/execution_cost.py`
- `src/live_trade_loop.py`

### Execution-Cost Enrichment

The execution engine should use cTrader enrichment to improve:

- spread regime estimation
- slippage pressure heuristics
- liquidity stress estimation
- post-cost edge validation

This should raise decision quality for larger-account handling even without direct cTrader execution.

### Quant / Feature Enrichment

Later in the same milestone or a follow-up, the quant layer may consume:

- micro-volatility
- spread shock flags
- quote-velocity
- opening-range state

The quant engine should treat these as additive features, not a hard replacement of MT5-native signals.

## Failure Handling

The sidecar should define explicit failure states:

- `disconnected`
- `auth_failed`
- `warming`
- `healthy`
- `stale`
- `degraded`

Runtime behavior:

- `healthy`: all enrichment available
- `warming`: partial enrichment allowed if fresh enough
- `stale` or `degraded`: execution and quant layers should ignore cTrader enrichments and fall back to MT5-only calculations

All transitions should be logged clearly.

## Testing Strategy

The sidecar should be testable without a live Spotware session.

Required tests:

- config parsing for enabled and disabled cTrader mode
- normalized model conversion from synthetic Open API payloads
- cache freshness and stale-state transitions
- indicator calculations from synthetic quote streams
- graceful fallback when the adapter is unavailable
- integration tests proving the execution engine can consume a synthetic microstructure snapshot

If OpenApiPy is not installed in the local test environment, the adapter package should degrade gracefully and the tests should rely on stubs or mocks.

## Non-Goals

This milestone does not aim to:

- replace MT5 candles with cTrader candles
- execute or manage trades through cTrader
- promise full order-book support
- introduce a networked service boundary
- add portfolio-level cross-venue optimization

## Success Criteria

This design is successful when:

1. the bot can start with cTrader disabled and behave exactly as before
2. the bot can start with cTrader enabled and warm a usable quote-enrichment cache
3. execution and quant layers can read normalized cTrader enrichment without knowing about Spotware internals
4. cTrader failure results in a clean MT5-only fallback rather than a runtime crash
5. the codebase is ready to accept future FIX or depth enrichment without rewriting the bot-facing interface
