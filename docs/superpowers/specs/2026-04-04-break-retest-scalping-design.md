# Break And Retest Scalping Design

## Goal

Design a deterministic strategy layer for the existing MT5/HFM execution skeleton that trades `XAUUSD` and `EURJPY` using `M5` break-and-retest scalping. The strategy should use higher-timeframe structure for context, enter only on the first qualified retest, and produce clear, testable trade plans.

## Scope

In scope for this strategy milestone:

- use `H1` major structure zones
- use `M15` refinement zones
- detect valid `M5` breakouts through those zones
- allow only the first retest after a breakout
- confirm entry using either rejection candles or micro structure breaks
- trade only during London and New York sessions
- compute stop loss from retest structure
- compute take profit at `1:3`
- produce deterministic signals that can later drive MT5 execution

Out of scope for this milestone:

- machine learning or score-based ranking
- discretionary manual zone drawing
- news-event filtering
- portfolio-level risk balancing
- TradingView scraping
- live deployment concerns beyond existing execution plumbing

## Recommended Approach

Three approaches were considered:

1. Strict rule engine
2. Score-based setup engine
3. Hybrid strict-plus-filters

The recommended approach is the hybrid strict-plus-filters model. It keeps the core strategy fully explainable while allowing a small number of practical live-trading filters later, such as spread or abnormal-volatility guards.

## Strategy Architecture

The strategy sits above the current MT5 plumbing as a signal engine:

1. Build `H1` and `M15` zones
2. Observe `M5` candles for a breakout beyond one of those zones
3. Mark the broken zone as eligible for exactly one retest
4. Wait for the first retest during London or New York
5. Confirm entry with either:
   - a rejection candle at the zone, or
   - a micro structure break after the retest
6. Calculate stop loss beyond the retest structure
7. Set take profit at `1:3`
8. Invalidate the setup if the retest is missed or the context degrades

The core requirement is deterministic behavior. The same market data should always produce the same setup decisions.

## Components

### `src/market_data.py`

Fetches and normalizes candle data for `H1`, `M15`, and `M5`. This module should hide MT5-specific data fetching details from the strategy engine.

### `src/strategy/zones.py`

Builds major `H1` support/resistance zones and refines them using `M15` structure. These zones become the reference levels for breakout detection.

### `src/strategy/breakout.py`

Determines whether price has truly broken a zone using `M5` candle closes rather than wick-only excursions.

### `src/strategy/retest.py`

Tracks breakout state and ensures only the first valid retest is eligible for entry.

### `src/strategy/confirmation.py`

Implements the two allowed entry confirmations:

- rejection candle at the retest zone
- micro pullback structure break in the direction of the breakout

### `src/strategy/risk.py`

Calculates stop loss from retest structure and derives take profit at a fixed `1:3` risk-reward ratio.

### `src/strategy/session_filter.py`

Allows signals only during London and New York sessions.

### `src/strategy/engine.py`

Coordinates all decision steps and returns either:

- no trade
- a complete trade plan with entry direction, entry rationale, stop loss, take profit, and context metadata

## Trading Rules

### Structure Zones

- Use `H1` for major support/resistance zones
- Use `M15` to refine those zones into tighter actionable areas

### Breakout Definition

- A breakout requires an `M5` candle close beyond the zone
- Wick-only probes are not valid breakouts

### Retest Rule

- Only the first retest of a broken zone is tradable
- Later retests are ignored for the first version

### Entry Confirmation

A trade is allowed only after a valid first retest and one of:

- a rejection candle at the zone, or
- a micro structure break on `M5` in the breakout direction

### Session Filter

- Trade only during London session
- Trade only during New York session
- Ignore setups outside those session windows

### Risk Model

- Stop loss sits beyond the retest structure or rejection extreme
- Take profit is fixed at `1:3`

### Invalidation Conditions

Cancel the setup if:

- the first retest is missed
- price deeply violates the zone before confirmation
- the allowed session ends before entry
- later live filters reject the setup due to spread or abnormal conditions

## Data Flow

The expected flow per symbol is:

1. Pull fresh `H1`, `M15`, and `M5` candle data
2. Rebuild or refresh current structure zones
3. Detect whether an actionable breakout occurred
4. If yes, track the first retest state
5. If retest occurs in session, evaluate confirmation
6. If confirmed, compute stop and target
7. Emit a trade plan
8. Otherwise emit a structured no-trade reason

The no-trade reason is important because it will make debugging and backtesting much easier.

## Logging And Explainability

Every strategy decision should be explainable. Logs or decision objects should record:

- which zone was considered
- whether breakout criteria passed
- whether the retest was first touch or late
- which confirmation type triggered, if any
- why the setup was invalidated, if rejected
- the computed stop loss and take profit

This is a strategy where auditability matters as much as the trade itself.

## Testing Strategy

This strategy should be implemented with strong unit coverage against synthetic candles and scenario fixtures.

Test areas:

- zone construction from known candles
- breakout detection with wick-only false positives
- first-retest tracking
- rejection-candle confirmation
- micro-structure-break confirmation
- session filtering
- stop-loss and take-profit calculation at `1:3`
- invalidation when retest is late or structure fails

The engine should be testable without requiring a live MT5 terminal.

## Future Extensions

After the first strategy version is stable, likely next steps are:

- spread filters
- minimum breakout strength filters
- no-trade conditions around extreme spikes
- symbol-specific tuning for `XAUUSD` vs `EURJPY`
- backtesting harness and journals
- optional TradingView-style indicator overlays for analysis

## Success Criteria

This strategy design is successful when it can consistently:

- identify major and refined structure zones
- detect valid `M5` breakouts
- allow only the first qualified retest
- require one of the two approved confirmations
- enforce London/New York session limits
- calculate structure-based stop loss and `1:3` take profit
- emit deterministic, explainable trade plans for `XAUUSD` and `EURJPY`
