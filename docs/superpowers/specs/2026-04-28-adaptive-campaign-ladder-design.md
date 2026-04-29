# Adaptive Campaign Ladder Design

## Goal

Upgrade campaign management from a coarse `1R trail / 2R add` rule into an audit-grade ladder model suitable for larger accounts. The bot should:

- protect each position independently using explicit `R`-based math
- lock realized profit progressively as trades mature
- add positions using a dedicated continuation model rather than the first-entry model
- relax or tighten add conditions dynamically based on momentum acceleration, volatility, execution quality, and quant state
- expose the full decision chain in logs and metadata so reviewers can reconstruct why an add, hold, trail, or block occurred

## Scope

In scope for this redesign:

- per-position breakeven and profit-lock ladder
- adaptive add trigger centered around `+1.5R`
- campaign-native continuation scoring using `M15`, `M10`, and `M5`
- quant soft-block vs hard-block behavior for campaign adds
- replacement of blunt campaign gates with more specific add-block reasons
- symbol-configurable breakeven distance for metals and FX

Out of scope for this milestone:

- portfolio-level cross-symbol optimization
- online parameter learning
- external execution venues or order-book integration
- database/dashboard work

## Design Principles

This campaign engine should read like a risk-control system, not a pile of heuristics.

The code should preserve these invariants:

1. older entries should always be more protected than newer entries once a ladder is underway
2. campaign exposure should only increase after existing trades have earned protection
3. new adds should be the first trades to fail on reversal, while older trades retain more locked profit
4. quant should never be allowed to silently overrule campaign logic without logging whether it soft-blocked or hard-blocked
5. all threshold relaxations should be explainable by measurable improvements in continuation quality

## Position-Level Risk Ladder

For each open position `i`:

```text
R_i = |entry_i - initial_stop_i|
progress_i = favorable_move_i / R_i
```

### Stage 0: Initial Risk

The trade starts with its original structural stop.

### Stage 1: Early Protection

Each symbol defines a configured early-protection distance:

```text
breakeven_trigger_i = min(symbol_breakeven_distance, 1R_i)
```

When trade `i` reaches `breakeven_trigger_i`, move stop to `entry_i`.

This ensures the bot does not allow a decent early favorable move to become a full loser.

### Stage 2+: Profit Lock Ladder

After breakeven:

```text
locked_r_i = max(floor(progress_i) - 1, 0)
```

Stop placement:

```text
Bullish:
SL_i = entry_i + locked_r_i * R_i

Bearish:
SL_i = entry_i - locked_r_i * R_i
```

This yields:

- at `+2R`, lock `+1R`
- at `+3R`, lock `+2R`
- at `+4R`, lock `+3R`
- continue indefinitely while the campaign remains healthy

This gives the campaign a clean convex profile:

- oldest trade accumulates the most locked profit
- newest trade remains the least protected
- reversal should cut the newest trade first and the oldest trade last

## Campaign Add Model

Campaign adds should no longer depend on whether the full first-entry engine would open a completely fresh trade.

Instead, once a campaign is live and eligible to add, the bot should evaluate a dedicated continuation model.

### Base Add Trigger

The newest position becomes eligible for a new add when its own progress reaches a base threshold:

```text
base_add_trigger_R = 1.5
```

This base threshold is then adjusted by acceleration, execution, and volatility.

### Adaptive Add Trigger

```text
trigger_R = clamp(
    1.5 - accel_bonus + execution_penalty + volatility_penalty,
    1.25,
    1.75
)
```

Interpretation:

- strong continuation and clean execution let the campaign add earlier
- messy volatility or degraded execution force the bot to wait longer before pressing

## Momentum Acceleration Model

Add logic should only relax when short-term continuation quality is improving in measurable terms.

Define normalized acceleration:

```text
accel_score =
    0.35 * M10_delta
  + 0.30 * M5_delta
  + 0.20 * range_expansion
  + 0.15 * body_efficiency
```

Where each component is normalized to `[0, 1]`.

Definitions:

- `M10_delta`: change in `M10` continuation quality relative to the prior evaluation
- `M5_delta`: change in `M5` execution quality relative to the prior evaluation
- `range_expansion`: normalized short-horizon range expansion versus recent baseline
- `body_efficiency`: body-to-range ratio, preferring decisive continuation candles over indecisive wicks

Use:

- higher `accel_score` lowers the add trigger and lowers the add threshold modestly
- lower `accel_score` raises the trigger and threshold, preserving discipline in a fading move

## Campaign Add Score

The add decision should use a dedicated continuation score:

```text
add_score =
    0.28 * M15_quality
  + 0.30 * M10_quality
  + 0.27 * M5_quality
  + 0.10 * progress_bonus
  + 0.05 * quant_bonus
```

Definitions:

- `M15_quality`: continuation validity of the higher confirmation leg
- `M10_quality`: refinement/continuation quality on the intermediate execution layer
- `M5_quality`: micro execution quality of the actual add timing
- `progress_bonus`: normalized reward for strong newest-trade progress beyond the base threshold
- `quant_bonus`: small positive adjustment when quant is aligned with the campaign rather than merely neutral

The bot should add only when:

```text
add_score >= adaptive_add_threshold
```

## Adaptive Add Threshold

```text
adaptive_add_threshold =
    base_threshold
  - protected_bonus
  - accel_bonus
  + execution_penalty
  + quant_penalty
```

With the final threshold clamped to a safe range.

Interpretation:

- `protected_bonus`: older positions have already locked profit, so controlled exposure expansion is more acceptable
- `accel_bonus`: if continuation is strengthening, the bar to add can relax slightly
- `execution_penalty`: widen threshold if spread/slippage conditions deteriorate
- `quant_penalty`: widen threshold if the quant engine becomes skeptical but not enough to hard-block

## Quant Soft Block vs Hard Block

Campaign adds should not be vetoed the same way as fresh first entries.

### Hard Block

Quant should hard-block a campaign add when any of these is true:

- inferred quant direction flips against the live campaign direction
- execution-adjusted CVaR exceeds the configured add-risk tolerance
- drawdown dampener is materially active
- expected return after cost is strongly negative

Result:

- no add allowed
- log reason as `campaign_add_quant_blocked`

### Soft Block / Reduced Size

Quant should soft-block, not veto, when:

- the campaign is already protected
- continuation score remains strong
- quant is flat or only mildly skeptical
- execution remains acceptable

Result:

- allow add with reduced lot multiplier
- log reason as `campaign_add_quant_reduced`

This is important for auditability: the bot should distinguish “quant says absolutely no” from “quant says smaller size only”.

## Previous Gate Fixes

The older campaign gates are too coarse.

### Replace These Behaviors

- `campaign_edge_below_add_threshold` as the dominant add blocker
- `campaign_add_alignment_missing` as a catch-all
- full dependency on the first-entry trade plan for campaign adds

### With These Behaviors

Management should decide only:

- whether reversal is confirmed
- whether exposure room exists
- whether newest-trade progress is sufficient to consider adding

Then the campaign-add engine should decide the actual add result with explicit reasons:

- `campaign_add_m15_not_ready`
- `campaign_add_m10_not_ready`
- `campaign_add_m5_not_ready`
- `campaign_add_edge_too_weak`
- `campaign_add_execution_rejected`
- `campaign_add_quant_reduced`
- `campaign_add_quant_blocked`

This separation makes the code easier to audit because each layer owns one class of decision.

## Components

### `src/strategy/management.py`

Should continue to own:

- reversal exit decision
- exposure ceiling logic
- position progress tracking
- lock-step trailing and stop updates

Should be upgraded to compute ladder-based stop levels per position instead of a single coarse trail step.

### `src/strategy/campaign_add.py`

Should own:

- `M15 -> M10 -> M5` continuation-based add scoring
- adaptive add trigger math
- adaptive add threshold math
- campaign-add decision object

### `src/live_trade_loop.py`

Should:

- route eligible add scenarios into the campaign-add engine
- apply quant soft/hard block behavior explicitly
- pass execution-adjusted add decisions into the existing execution-cost overlay
- log final add, hold, trail, reduce, or block outcomes clearly

### `src/config.py`

Should expose symbol-configurable early-protection distances, starting with `XAUUSD`.

### `src/strategy/execution_cost.py`

Should remain the final pre-order filter, including:

- spread pressure
- slippage pressure
- stop degradation
- effective RR after cost
- size haircut recommendations

## Auditability and Logging

Every add-hold decision should expose enough information for an audit reviewer to reproduce the judgment:

- newest trade progress in `R`
- current locked `R` on each open trade
- adaptive add trigger
- add score
- add threshold
- acceleration score
- execution penalty
- quant soft/hard block state
- resulting lot decision

This is especially important for large-account review. Reviewers should see a structured decision process rather than infer intent from scattered log lines.

## Testing

Add or extend tests for:

- breakeven move at `min(15 pips, 1R)`
- profit lock at `2R`, `3R`, `4R`
- add allowed at strong protected `+1.5R` continuation
- add delayed when execution penalty rises
- add hard-blocked by quant direction flip
- add reduced by soft quant skepticism
- oldest trade keeping the most locked profit while newest trade remains least protected

## Recommendation

Implement the ladder in two tightly-coupled passes:

1. per-position early protection and profit-lock ladder
2. adaptive campaign add threshold plus quant soft/hard block split

This keeps the rollout controlled while materially improving how the campaign behaves in strong momentum without sacrificing auditability.
