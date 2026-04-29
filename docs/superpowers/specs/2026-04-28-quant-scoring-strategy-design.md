# Quant Scoring Strategy Design

## Goal

Upgrade the live strategy from mostly categorical decision nodes into a volatility-normalized, regime-aware scoring engine that can make stronger live decisions for serious capital. The bot should continue to use the current top-down `D1 -> H4 -> H1 -> M30 -> M15` structure, but replace hard gates with side-by-side bullish and bearish scoring, uncertainty penalties, and campaign-aware thresholds.

## Scope

In scope for this redesign:

- add a quantitative feature pipeline to the existing top-down strategy
- normalize distances, momentum, and trigger quality by recent volatility
- classify live market regime before evaluating directional edge
- compute bullish and bearish scores independently instead of relying on simple pass/fail rules
- add uncertainty penalties for disagreement, unstable volatility, and noisy conditions
- make decision thresholds symbol-aware, regime-aware, session-aware, and campaign-aware
- keep live execution deterministic by using stable parameter sets
- log the full score breakdown so later tuning can happen offline

Out of scope for this milestone:

- self-training or online model updates during live trading
- replacing MT5 execution with external execution engines
- adding a database, dashboard, or API layer
- building a backtesting or Colab training pipeline in this pass

## Recommended Approach

Three approaches were considered:

1. scored decision engine with volatility normalization
2. volatility-only upgrade without broader scoring changes
3. campaign-only probability math without upgrading entry logic

The chosen approach is the scored decision engine with volatility normalization. It improves the whole strategy stack, not only one layer, and preserves explainability while making the live engine more mathematically serious.

## Strategy Model

The upgraded engine should evaluate every loop with the following sequence:

1. build raw multi-timeframe context from `D1`, `H4`, `H1`, `M30`, and `M15`
2. compute volatility features
3. classify regime
4. compute location, momentum, setup, trigger, gap, and external confluence features
5. score both bullish and bearish cases independently
6. apply uncertainty penalties
7. compare net edge against adaptive thresholds
8. either return a trade plan, a no-trade result, or campaign-management instructions

This keeps the current top-down architecture but makes each node contribute numeric evidence instead of acting like a brittle hard stop.

## Quant Pipeline

For each symbol and loop, the bot should compute:

- `volatility_state`
  - short ATR
  - medium ATR
  - realized range expansion
  - candle body-to-range efficiency
- `regime_state`
  - trend
  - pullback
  - compression
  - expansion
  - gap-reversion
  - gap-acceptance
- `location_features`
  - daily range percentile
  - H4 demand distance
  - H4 supply distance
  - previous session high/low distance
  - MT5 volume-profile distance
- `momentum_features`
  - H1 slope
  - M30 slope
  - normalized impulse strength
  - push-vs-pullback quality
- `trigger_features`
  - M15 body strength
  - wick imbalance
  - follow-through quality
  - local break quality
- `uncertainty_features`
  - cross-timeframe disagreement
  - volatility instability
  - spread abnormality
  - noisy candle clustering

The engine should then compute per-side scores:

- `location_score`
- `momentum_score`
- `setup_score`
- `trigger_score`
- `gap_score`
- `external_confluence_score`
- `uncertainty_penalty`
- `expected_move_score`

Conceptually:

```text
net_score(side) =
  regime_weight * (
    location_score +
    momentum_score +
    setup_score +
    trigger_score +
    gap_score +
    external_confluence_score
  )
  - uncertainty_penalty
```

Decision edge:

```text
edge = net_score(bullish) - net_score(bearish)
```

The bot should only trade when:

- `abs(edge)` exceeds a minimum edge threshold
- expected move supports the intended reward profile
- uncertainty remains below the maximum allowed threshold
- the detected regime is tradable for the intended direction

## Calibration Model

The bot should not share one universal parameter set across symbols.

### Symbol Profiles

Each symbol should have its own profile, starting with:

- `XAUUSD`
- `EURJPY`

Each profile should define:

- feature weights
- volatility bands
- gap sensitivity
- regime thresholds
- uncertainty tolerances
- campaign add thresholds

### Regime Profiles

Thresholds should adapt by regime:

- expansion and trend regimes can allow stronger continuation entries
- pullback regimes can support cleaner re-entry math
- compression and chop regimes should require more edge or block trading entirely
- gap-reversion and gap-acceptance regimes should influence how much the gap bias matters

### Session Profiles

Thresholds should also adapt by session:

- London open
- New York overlap
- thinner periods

Thin periods should require more edge and lower uncertainty before allowing trades or add-ons.

### Campaign-Aware Thresholds

Add-on entries inside a live campaign should require stricter thresholds than the initial entry. When exposure is already building, continuation quality must be better than for the first trade.

## Components

### `src/strategy/volatility.py`

New module for:

- ATR calculations
- recent range normalization
- body-efficiency measures
- volatility state extraction

### `src/strategy/regime.py`

New module for:

- trend classification
- pullback/compression/expansion detection
- gap-reversion vs gap-acceptance tagging

### `src/strategy/scoring.py`

New central scoring module for:

- bullish/bearish feature scoring
- uncertainty penalties
- expected-move scoring
- threshold comparison

### `src/strategy/context.py`

Should continue to build raw higher-timeframe context, but also expose normalized feature inputs suitable for scoring.

### `src/strategy/direction.py`

Should stop owning most of the directional math itself and instead consume the scoring outputs for `H1` directional contribution.

### `src/strategy/setup.py`

Should evolve from categorical readiness into quantified `M30` setup quality scoring.

### `src/strategy/trigger.py`

Should score `M15` trigger quality rather than mainly returning a binary ready/not-ready result.

### `src/strategy/decision_tree.py`

Should become the top-level coordinator for:

- context building
- volatility extraction
- regime classification
- per-side scoring
- uncertainty adjustment
- trade/no-trade decisioning

### `src/strategy/management.py`

Should consume the stronger scoring model for campaign add-ons, especially by requiring higher continuation quality when exposure is already active.

## Decision Outputs

The upgraded engine should still return explicit outcomes:

1. `trade ready`
2. `no trade`
3. `manage active campaign`

But each outcome should include a richer score breakdown in metadata:

- bullish score total
- bearish score total
- edge
- uncertainty penalty
- regime
- volatility state
- expected move score
- threshold used

This preserves explainability while increasing mathematical depth.

## Risk and Stability Rules

The live bot should use stable parameter sets only. It should not update weights or thresholds dynamically from live outcomes during runtime.

Instead, the bot should log:

- raw features
- regime classification
- per-side scores
- penalties
- final decision
- outcome after the trade or campaign phase

Those records can be used later for offline tuning, research, or model training without destabilizing the live account.

## Testing Strategy

This upgrade should be implemented with strong unit coverage around the new math:

- volatility normalization correctness
- regime classification behavior
- bullish/bearish scoring balance
- uncertainty penalties
- adaptive threshold logic
- symbol-specific profile selection
- campaign add threshold behavior
- regression coverage for known live scenarios that previously over-blocked or overtraded

Decision-level tests should verify that the same market context can legitimately produce:

- a trade in one regime
- a no-trade in another regime
- a stricter add-on rejection once campaign exposure is already active

## Expected Outcome

After this redesign, the bot should stop behaving like a collection of loose hard-coded rules and start acting like a more disciplined quantitative engine:

- more selective in noisy conditions
- more adaptive across symbols and sessions
- more explainable about why edge exists
- stricter when exposure is already active
- safer to tune offline for larger capital
