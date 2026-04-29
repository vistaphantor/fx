# Live Only Runtime Design

## Goal

Redesign the MT5/HFM bot runtime so `python run.py` is always the live execution path. The bot should no longer carry `dry_run`, `test_trade`, or mixed-mode startup logic in its main runtime. The operator will switch between demo and live accounts manually through `.env`, and the bot should simply trade whichever account is configured.

## Scope

In scope for this redesign:

- remove runtime mode branching from `run.py`
- make `run.py` always launch MT5, log in, and start the live strategy loop
- remove `dry_run` and skeleton test trade paths from the production runtime
- keep strategy evaluation and order-building test coverage
- keep separate developer diagnostics out of the main runtime path
- tighten live-loop behavior around duplicate entries and position ownership

Out of scope for this redesign:

- changing brokers or MT5 integration libraries
- adding machine learning or signal scoring
- building a UI or control panel
- replacing the current strategy with a fully complete discretionary replica in one step
- deploying the bot as a Windows service

## Recommended Approach

Three approaches were considered:

1. Hard live-only entrypoint
2. Live-only core with separate developer tools
3. Service-style live bot refactor

The chosen approach is the hard live-only entrypoint. `run.py` becomes a pure live executor with no runtime modes. Any non-trading diagnostics should live in tests or separate helper scripts, not in the production command.

## Runtime Architecture

The runtime should follow one path only:

1. Load account, symbol, polling, and risk settings from `.env`
2. Launch MT5
3. Initialize and log into HFM
4. Start the live strategy loop
5. Evaluate the strategy on each polling cycle
6. If there is no valid setup, log the no-trade reason and wait
7. If there is a valid setup and no conflicting open bot-managed position, place a live MT5 order with SL/TP
8. If there is an active bot-managed position, manage or monitor that position instead of opening another

The main runtime should no longer contain any code path named `dry_run`, `test_trade`, or `live`. There is only live execution.

## Components

### `run.py`

Owns startup only:

- load settings
- launch MT5
- log into HFM
- optionally start TradingView webhook support if retained
- start the live strategy loop

It should not contain mode branching.

### `src/live_trade_loop.py`

Owns the live polling loop:

- fetch live market context
- evaluate strategy
- prevent duplicate entries on unchanged setups
- skip opening new positions when a bot-managed position is already active
- emit clear live logs for no-trade reasons and order actions

### `src/strategy/top_down.py`

Represents the main live strategy direction layer. It should evolve toward the user’s real process:

- `D1` daily high/low context
- `H4` session high/low plus supply/demand context
- `H1` directional confirmation
- `M30` setup refinement
- `M15` entry timing and velocity

The current simplified implementation is only an interim version and should not be treated as the finished trading model.

### `src/strategy/engine.py`

Keeps the structured break-and-retest logic if that remains part of the final decision pipeline. If it becomes redundant after the top-down refactor, it can be folded into the main strategy module later.

### `src/trade_executor.py`

Owns MT5 order submission and management actions:

- build broker-compatible market order requests
- choose a supported fill mode
- attach stop loss and take profit
- identify bot-managed positions
- later support stop updates and trailing logic

### `src/config.py`

Loads only live runtime settings. It should stop exposing:

- `BOT_MODE`
- `ENABLE_TEST_TRADE`
- `LIVE_TRADING_ENABLED`

The account choice itself is still controlled manually by editing `.env`.

## Safety And Behavior

Even with a hard live-only runtime, the bot should still behave conservatively:

- `python run.py` always trades the currently configured account
- only one bot-managed position per symbol at a time
- manual positions should not be silently mistaken for bot-owned positions
- duplicate entries on the same unchanged setup should be blocked
- order comments should clearly identify bot-owned trades, such as `strategy-live`
- startup failures should stop the process immediately with explicit logs
- order failures should not retry in a tight uncontrolled loop

The removal of dry-run does not remove the need for safety checks. It only removes alternate runtime modes.

## Error Handling

The live-only runtime should fail loudly on:

- missing MT5 terminal
- MT5 initialization failure
- HFM login failure
- symbol selection failure
- insufficient candle history for the strategy
- rejected broker orders

These failures should stop the bot or cleanly skip the current cycle depending on severity. Silent fallthroughs are not acceptable in a live-only command.

## Position Ownership

The bot must distinguish its own trades from unrelated ones as much as MT5 allows.

Minimum requirements:

- use a consistent bot comment on entries
- prefer matching positions by returned ticket
- avoid falling back to “first position for symbol” unless there is a strong ownership signal
- avoid opening another trade if a bot-owned position for the symbol is already active

This reduces the chance of the bot attaching to a manual trade or duplicating its own setup.

## Trade Lifecycle

The live loop should not stop at entry placement. The intended behavior should eventually include:

- initial SL/TP placement at order time
- breakeven or protective trailing after `1R`
- progressive trailing at later `R` milestones
- optional early exit when the trading objective is met but price is far from TP

This redesign does not need to fully implement trailing behavior yet, but the architecture should clearly support it.

## Testing Strategy

The test suite should be updated to match the new runtime truth:

- remove tests that assume `run.py` supports `dry_run` or `test_trade`
- add tests for the single live startup path
- keep unit tests for strategy evaluation
- keep unit tests for MT5 order request construction
- add or strengthen tests for duplicate-entry prevention and position ownership behavior

The tests should verify correctness without requiring real MT5 order placement.

## Success Criteria

This redesign is successful when:

- `python run.py` has exactly one runtime meaning: live execution
- no dry-run or skeleton test branches remain in the production path
- live order placement still works with broker-supported fill modes
- duplicate entries and position ownership are handled more safely
- tests reflect the live-only design
- the runtime is simpler, more predictable, and easier to operate when switching between demo and live accounts manually
