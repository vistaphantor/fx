# Top-Down Momentum Campaign Design

## Goal

Redesign the bot's live trading intelligence so it follows the user's real top-down workflow instead of a loose momentum fallback. The bot should make explainable live decisions from `D1` through `M15`, campaign into strong moves with synchronized trailing, and log the exact decision-tree node that allowed or blocked each trade.

## Scope

In scope for this redesign:

- replace the current simplified top-down fallback with a full decision tree
- build `D1`, `H4`, `H1`, `M30`, and `M15` strategy layers
- include daily highs/lows, previous session highs/lows, demand/supply context, and MT5-based volume-profile reference levels
- detect `three-drives` and related confluence patterns near meaningful context
- produce live trade plans with structure-based stop loss and directional objectives
- manage positions as an open-ended momentum campaign instead of single isolated trades
- add synchronized trailing and live add-on logic as momentum expands
- keep the bot explainable by returning structured no-trade reasons

Out of scope for this milestone:

- a frontend dashboard or API layer
- Flask/SQLAlchemy persistence
- machine-learning execution decisions
- Colab training pipelines
- PythonAnywhere deployment
- broker changes or non-MT5 execution plumbing

## Recommended Approach

Three approaches were considered:

1. Pure MT5 top-down engine
2. MT5-first engine with optional TradingView confluence
3. TradingView-led execution filter

The chosen approach is the MT5-first engine with optional TradingView confluence. MT5 remains the source of execution and core market context so the bot can act independently, while TradingView alerts can later strengthen confidence without becoming mandatory for every trade.

## Trading Model

The bot should evaluate one directional campaign per symbol using this top-down flow:

1. `D1` context
2. `H4` structure and session context
3. `H1` directional bias
4. `M30` setup refinement
5. `M15` trigger validation
6. pattern and confluence checks
7. trade entry
8. post-entry campaign management

At every stage, the bot should either:

- advance to the next decision node
- produce a trade plan
- or stop with a structured no-trade result that names the failed node

## Decision Tree

### `D1` Context

The `D1` layer should establish:

- current daily high
- current daily low
- range position of current price
- directional objective relative to daily extremes

This layer defines whether price is still moving toward a meaningful daily objective or whether that objective is already exhausted.

### `H4` Context

The `H4` layer should establish:

- previous session high and previous session low
- demand and supply candidates
- MT5-based volume-profile reference levels
- whether current price is reacting from or approaching important context

Volume profile is optional confluence, not a hard requirement. The bot should be able to trade without it, but when present it should improve context quality.

### `H1` Direction

The `H1` layer should decide directional bias from:

- structure
- candle reaction
- alignment with `D1` and `H4` context
- whether price is pushing from demand or rejecting supply

The bot should not reduce this layer to a simple moving average or close-to-close momentum check.

### `M30` Setup

The `M30` layer should answer:

- is price approaching a level with intent
- is price rejecting a level
- is price breaking away from a level
- is the setup location clean enough to trade

This layer refines whether the higher-timeframe bias is currently actionable.

### `M15` Trigger

The `M15` layer should confirm:

- directional velocity
- candle intent
- local structure support for the trade
- whether the setup is ready now or still incomplete

This is the final trigger layer before entry.

### Pattern Confluence

The bot should detect:

- `three-drives`
- other confluence patterns near key levels

These patterns should strengthen the setup, but not every valid trade must depend on a `three-drives` signal.

## Components

### `src/strategy/context.py`

Builds `D1` and `H4` context:

- daily highs/lows
- previous session highs/lows
- demand/supply candidates
- MT5-based volume-profile reference levels

### `src/strategy/direction.py`

Produces `H1` directional bias from:

- structure
- candle reaction
- context alignment

### `src/strategy/setup.py`

Refines the `M30` setup state and determines whether price is:

- approaching
- rejecting
- or expanding away from a usable level

### `src/strategy/trigger.py`

Handles `M15` trigger readiness:

- velocity
- candle intent
- local structural confirmation

### `src/strategy/patterns.py`

Detects `three-drives` and similar confluence patterns near meaningful levels.

### `src/strategy/decision_tree.py`

Coordinates the full top-down flow and returns either:

- a live trade plan
- or a structured no-trade result with the exact failed node

### `src/strategy/management.py`

Owns post-entry live trade management:

- initial structure-based SL
- synchronized trailing behavior
- add-on logic during momentum campaigns
- reversal handling

### `src/live_trade_loop.py`

Owns runtime orchestration:

- fetch strategy inputs
- evaluate the decision tree
- avoid invalid duplicate entries
- manage existing campaigns

### `src/trade_executor.py`

Keeps MT5 order placement and expands to support:

- broker-compatible stop updates
- trailing-stop modification
- identification of bot-managed campaign positions

## Trade Plan Output

When the tree reaches a valid trade, the plan should include:

- direction
- entry
- stop loss
- initial objective
- confluence notes
- decision metadata showing why the trade qualified

When the tree rejects a trade, the result should include explicit reasons such as:

- `d1_objective_exhausted`
- `h4_context_not_clean`
- `h1_bias_not_aligned`
- `m30_setup_not_ready`
- `m15_trigger_missing`
- `pattern_confluence_missing`

## Momentum Campaign Model

The bot should treat a run of aligned positions as one directional campaign per symbol.

### Campaign Start

A campaign starts when the first valid trade opens.

### Add-On Rule

The bot may add new positions when the newest open campaign trade reaches `+2R`, provided:

- higher-timeframe context is still valid
- the decision tree still supports the direction
- no confirmed reversal has appeared
- campaign exposure remains inside limits

There is no hard cap on the number of added positions. If momentum stays clean, the campaign may continue expanding.

### Lot Sizing

- the first position uses the configured default lot size
- each new add should try `default lot + 0.01`
- if margin or broker constraints reject that size, the bot falls back to the default lot size
- all lot requests should be normalized to broker min, max, and step constraints

## Trailing Logic

The bot should trail profit across the whole campaign in a synchronized way.

- each position starts with its own initial structure-based stop loss
- when the latest opened trade reaches `+1R`, all open campaign positions tighten their protection
- earlier trades should keep locking in more profit as later trades prove continuation
- structure-aware trailing is preferred, but the system should still guarantee minimum profit locking when structure is too loose

The core principle is that newer trades earn the right for the whole basket to tighten.

## Exposure Limit

The campaign should be limited by total exposure or margin usage, not only raw stop-loss distance.

- before each new add, the bot should estimate total campaign exposure
- if adding a new trade would push campaign exposure above `10%` of account, it should first try the fallback lot sizing rule
- if exposure would still exceed `10%`, it should skip the add and continue managing the existing campaign

## Reversal Handling

The bot should react in three levels:

1. If reversal is only suspected:
   - stop adding new positions
   - continue trailing existing ones

2. If trailing stops flatten the basket:
   - accept that as the campaign exit

3. If reversal is confirmed before trailers are hit:
   - close remaining open campaign positions early

This keeps the bot from giving back a strong run after the objective has effectively been achieved.

## Runtime Behavior

The live loop should behave like this on every cycle:

1. fetch fresh multi-timeframe candle data
2. inspect current bot-owned campaign positions
3. if no campaign is open, evaluate entry readiness
4. if a campaign is open, evaluate management and possible add-on readiness
5. execute only the next valid action
6. log the exact reason for no action when nothing changes

The loop should not mix old `dry_run` style logic into the production decision path.

## Error Handling

The bot should fail loudly or skip safely when it encounters:

- missing candle history
- broken context inputs
- invalid broker lot constraints
- unsupported MT5 stop updates
- rejected add-on orders
- inability to identify bot-owned positions clearly

Order failures should never cause uncontrolled rapid retries.

## Logging And Explainability

Every live decision should be explainable in logs and result objects.

For entries and no-trades, logs should capture:

- failed or passed decision node
- direction bias
- key context levels
- confluence notes
- objective and risk levels

For campaign management, logs should capture:

- active position count
- latest trade `R` state
- trailing actions
- add-on approvals or rejections
- reversal-confirmation exits

## Testing Strategy

This redesign should be implemented with strong unit coverage and loop-level tests.

Core test areas:

- `D1` daily context extraction
- `H4` previous-session, demand/supply, and volume-profile context
- `H1` directional bias decisions
- `M30` setup readiness decisions
- `M15` trigger validation
- `three-drives` detection
- decision-tree no-trade node reporting
- campaign add-on eligibility at `+2R`
- synchronized trailing after latest trade reaches `+1R`
- exposure-cap enforcement
- reversal-confirmation exits
- live loop prevention of invalid duplicate adds

The strategy should remain testable without needing a real MT5 terminal.

## Success Criteria

This redesign is successful when:

- the bot no longer relies on the current loose top-down fallback
- live decisions follow the user's `D1` to `M15` tree
- no-trade reasons identify the exact failed node
- campaign positions can be added during clean momentum
- synchronized trailing protects earlier profits as newer positions mature
- exposure stays inside the configured `10%` campaign cap
- confirmed reversals stop or exit campaigns decisively
- the live bot is more explainable, more faithful to the user's method, and safer to extend later
