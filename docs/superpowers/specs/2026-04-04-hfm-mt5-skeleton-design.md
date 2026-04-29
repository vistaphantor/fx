# HFM MT5 Skeleton Design

## Goal

Build a minimal Python skeleton that launches a configured MetaTrader 5 terminal, logs into an HFM demo account, places one tiny test trade, waits for a short delay, closes that position, and exits with clear logs. This milestone is only for proving broker connectivity and order execution plumbing before adding any strategy logic.

## Scope

In scope for this milestone:

- Run with `python run.py`
- Load credentials and runtime settings from a local `.env`
- Launch MT5 from a configured terminal executable path
- Initialize the MetaTrader5 Python API
- Log into an HFM demo account
- Validate one configured symbol
- Place one test market order with a configured side and lot size
- Wait a configured hold period
- Close the opened position
- Print enough logging to confirm each stage

Out of scope for this milestone:

- Any trading strategy
- Multi-symbol orchestration
- Live-account trading
- Persistent bot loops
- Risk engine beyond basic guardrails
- Retry queues, dashboards, or notifications

## Recommended Approach

Three approaches were considered:

1. Single-file script
2. Thin app with small modules
3. Full daemon-style scaffold

The recommended approach is the thin app with small modules. It keeps the first version simple while creating clean extension points for later strategy, risk controls, and multi-symbol support.

## Architecture

The entry point is `run.py`. It coordinates a short one-shot workflow:

1. Load and validate configuration from `.env`
2. Start the configured MT5 terminal process
3. Initialize the MetaTrader5 Python package
4. Log into the HFM demo account
5. Check that the configured symbol is visible and tradable
6. Send one small market order
7. Confirm the position opened
8. Wait a configured number of seconds
9. Send a closing order for the same position
10. Shutdown MT5 cleanly and exit

This structure keeps the code deterministic and easy to troubleshoot. It also matches the future direction of a full trading bot, where `run.py` remains orchestration and the broker logic lives in smaller modules.

## Components

### `run.py`

Owns the top-level workflow. It should orchestrate config loading, terminal launch, MT5 initialization, login, trade placement, timed close, and final shutdown.

### `src/config.py`

Loads `.env` values and validates required settings. Expected configuration includes:

- `MT5_TERMINAL_PATH`
- `HFM_LOGIN`
- `HFM_PASSWORD`
- `HFM_SERVER`
- `TRADING_SYMBOL`
- `TEST_TRADE_SIDE`
- `TEST_TRADE_LOT`
- `HOLD_SECONDS`
- `MT5_STARTUP_WAIT_SECONDS`

Validation should fail fast with clear messages if required values are missing or malformed.

### `src/mt5_client.py`

Responsible for:

- launching the MT5 executable
- waiting briefly for terminal startup
- initializing the MetaTrader5 API
- performing login
- retrieving account info for confirmation
- shutting down the MT5 API

It should raise explicit errors when initialization or login fails.

### `src/trade_executor.py`

Responsible for:

- selecting the symbol if needed
- reading current tick data
- creating a market order request
- sending the order
- confirming a position was opened
- creating a closing request for the exact position
- sending the close request

It should use a unique order comment such as `skeleton-test` so the test trade is easy to identify in MT5 history.

### `.env.example`

Documents the required environment variables and gives a runnable template without storing secrets.

## Runtime Behavior

The program is intentionally a one-shot smoke test rather than a long-running bot. When `python run.py` is executed:

- it loads validated config
- starts MT5
- logs into HFM demo
- prints confirmation of the logged-in account
- opens one tiny test position on the configured symbol
- waits a configured hold time
- closes the same position
- prints the result of each stage
- exits

If any required step fails, execution stops immediately and returns a non-zero exit path.

## Safety Guardrails

This skeleton must be conservative:

- use an HFM demo account only
- trade only one configured symbol
- use a very small configured lot size
- require explicit `buy` or `sell` instead of random direction
- refuse to continue if MT5 initialization fails
- refuse to continue if login fails
- refuse to continue if the symbol cannot be selected or priced
- close only the position opened by this run

Although this is only a plumbing test, deterministic behavior is important. A configured trade side is safer than a random trade because it makes logs, debugging, and later tests reproducible.

## Error Handling

Failures should be surfaced with clear messages for:

- missing or invalid environment variables
- MT5 executable path not found
- MT5 initialize failure
- HFM login failure
- missing symbol or unavailable tick data
- rejected order placement
- missing opened position after order placement
- rejected close order

Where available, MT5 return codes and messages should be included in logs to speed up diagnosis.

## Logging

The first version should use simple structured console logging. The logs should show:

- configuration validation success
- MT5 launch attempt
- MT5 initialization result
- login result and account confirmation
- symbol check result
- order request summary
- order send result
- position identifier and open confirmation
- close request summary
- close result
- final success or failure

The logging does not need file rotation or external sinks yet.

## Testing Strategy

This milestone should include both automated and manual verification where practical.

Automated tests:

- config validation tests for missing and malformed values
- request-building tests for buy and sell order payload creation
- close-request tests that ensure the correct position id is targeted

Manual verification:

- run `python run.py`
- confirm MT5 launches successfully
- confirm login succeeds against the HFM demo account
- confirm one small trade opens
- confirm it closes after the configured delay
- confirm logs reflect the full lifecycle

Because external broker connectivity depends on a local MT5 terminal and demo credentials, the final end-to-end broker test is expected to be manual.

## Future Extensions

The intended first strategy after the execution skeleton is validated is:

- `M5` break-and-retest scalping
- initial target symbols: `XAUUSD` and `EURJPY`

That strategy is intentionally out of scope for this plumbing milestone, but the module boundaries in this design are chosen so the later strategy layer can plug in cleanly.

Once this skeleton works, the next milestones can safely build on it:

- add strategy modules for `XAUUSD` and `EURJPY`
- implement `M5` break-and-retest scalping entry and exit rules
- add position sizing and max-risk rules
- add spread, session, and slippage filters
- support multiple symbols in one run
- move from one-shot execution to scheduled or persistent bot operation
- add persistent logs and trade journals

## Success Criteria

This milestone is successful when:

- `python run.py` launches the configured MT5 terminal
- the script logs into the HFM demo account successfully
- the script opens one configured test trade
- the script closes the same position after the configured delay
- failures produce clear actionable logs
