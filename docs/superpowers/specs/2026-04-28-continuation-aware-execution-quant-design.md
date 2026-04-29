# Continuation-Aware Execution And Quant Design

## Goal

Upgrade the bot so strong momentum continuation trades are no longer rejected by fresh-entry math.

The redesigned model should:

- stop using a blunt fixed `effective_rr >= 1.5` continuation veto
- allow lower immediate `R:R` when continuation persistence is strong and post-cost expected value is positive
- distinguish fresh-entry quant logic from continuation-entry quant logic
- preserve hard disaster floors so weak or expensive continuation trades are still blocked
- expose explicit continuation metrics in logs and metadata for audit review

## Scope

In scope for this redesign:

- continuation-aware execution gating
- dynamic `R:R` floor logic
- continuation probability and expected-value math
- continuation-specific quant override behavior
- richer continuation metadata emitted by the decision tree
- clear runtime reasons such as `continuation_quant_approved` and `continuation_quant_blocked`

Out of scope for this milestone:

- retraining the full quant model from historical trade data
- external venue routing
- portfolio optimization across symbols
- replacing the existing top-down strategy tree

## Problem Statement

The current system is too harsh on live momentum continuation.

Observed behavior:

- the execution engine rejects trades with `execution_rr_degraded` when `effective_rr < 1.5`
- the quant engine independently flattens continuation trades as `master_equation_flat`
- this happens even when live market behavior shows strong continuation momentum

The root issue is structural:

- the execution engine still judges continuation entries with a fresh-entry payoff filter
- the quant engine still applies a raw return-vs-cost gate that is too pessimistic for mature continuation states

The bot therefore undervalues trades where:

- immediate remaining room is modest
- but continuation probability is high enough to make the post-cost expected value positive

## Design Principles

The continuation model should preserve discipline without forcing every continuation trade to look like an untouched first entry.

It should preserve these invariants:

1. continuation trades must still satisfy a hard minimum payoff floor
2. lower `R:R` is acceptable only when supported by measurable continuation persistence
3. execution cost and tail risk remain first-class penalties
4. quant must distinguish between fresh-entry flatness and continuation approval
5. every relaxation must be traceable to explicit, logged numeric inputs

## Execution Engine Redesign

The execution engine should stop using:

```text
reject if effective_rr < 1.5
```

for all continuation states.

Instead, it should apply a two-layer gate:

### Layer 1: Disaster Floor

Reject immediately when continuation quality is structurally unacceptable.

Suggested disaster-floor rules:

- reject if `effective_rr < 0.60`
- reject if spread pressure is extreme
- reject if slippage pressure is extreme
- reject if stop degradation is extreme

This floor prevents clearly poor continuation entries from being justified by aggressive probability assumptions.

### Layer 2: Continuation Expected-Value Gate

If the disaster floor passes, evaluate continuation expected value:

```text
EV_cont = p_cont * effective_rr - (1 - p_cont) - execution_penalty - tail_penalty
```

Interpretation:

- `p_cont`: continuation probability
- `effective_rr`: post-slippage reward-to-risk ratio
- `execution_penalty`: spread/slippage/capacity friction
- `tail_penalty`: volatility or CVaR-style downside penalty

Trade continuation when either:

- `effective_rr >= dynamic_rr_floor`
- or `EV_cont > 0` with adequate continuation persistence

This allows continuation trades with lower immediate `R:R` if the probability-adjusted edge remains positive.

## Continuation Probability Model

Continuation probability should be derived from measurable state, not hardcoded optimism.

Define:

```text
p_cont = sigmoid(
    β1 * M15_quality
  + β2 * M10_quality
  + β3 * M5_quality
  + β4 * range_expansion
  + β5 * body_efficiency
  + β6 * regime_confidence
  - β7 * execution_penalty
)
```

Suggested interpretation of terms:

- `M15_quality`: higher-timeframe continuation validity
- `M10_quality`: intermediate continuation/refinement quality
- `M5_quality`: execution-layer continuation quality
- `range_expansion`: current expansion versus local baseline
- `body_efficiency`: conviction of recent candles versus wick-heavy noise
- `regime_confidence`: directional confidence from the regime classifier
- `execution_penalty`: friction from spread/slippage/capacity

All components should be normalized before combining them.

## Dynamic `R:R` Floor

The immediate `R:R` floor should be adaptive rather than fixed.

Suggested formula:

```text
dynamic_rr_floor = clamp(
    1.20
    - 0.45 * p_cont
    - 0.20 * trend_confidence
    + 0.25 * execution_penalty,
    0.60,
    1.50
)
```

Interpretation:

- weak continuation keeps the floor close to historical strictness
- strong continuation can relax the floor materially
- degraded execution raises the bar again

This preserves safety while allowing momentum trades that would be unfairly rejected by a hard `1.5R` rule.

## Decision Tree Metadata

The decision tree should emit richer continuation-state metadata so execution and quant can reason explicitly.

Required additions:

- `is_continuation_setup`
- `m15_quality`
- `m10_quality`
- `m5_quality`
- `range_expansion_ratio`
- `body_efficiency`
- `regime_confidence`
- `continuation_persistence_score`

This metadata should be attached to `TopDownTradePlan.metadata` for downstream consumers.

## Quant Engine Redesign

The quant engine should distinguish between:

- `fresh_entry_quant`
- `continuation_quant`

The current binary flattening behavior:

- `master_equation_flat`

is too coarse for continuation entries.

### Fresh Entry

Keep current quant behavior relatively strict.

Fresh trades should still require:

- strong `Ω_t`
- positive post-cost expectancy
- acceptable tail risk

### Continuation Entry

Continuation quant should use a softer three-state outcome:

- `continuation_quant_approved`
- `continuation_quant_reduced`
- `continuation_quant_blocked`

This means the continuation quant layer can:

- approve the trade at full size
- approve it at reduced size
- block it entirely

rather than flattening every borderline continuation case.

## Continuation Quant Gate

Continuation quant should use continuation-adjusted expectancy:

```text
EV_quant_cont = p_cont * payoff_cont - (1 - p_cont) * loss_cont - cost_penalty - η * CVaR
```

Where:

- `payoff_cont` is based on effective post-cost reward
- `loss_cont` is based on effective post-cost stop risk
- `cost_penalty` includes transaction-cost friction
- `CVaR` remains a tail-risk penalty

Decision policy:

- full approval if `EV_quant_cont` is strongly positive and `Ω_t` is acceptable
- reduced approval if `EV_quant_cont` is positive but confidence is mediocre
- block if `EV_quant_cont <= 0` or quant direction opposes continuation

## Runtime Reasons

The runtime should stop overloading all continuation failures into `master_equation_flat`.

Required reason codes:

- `continuation_quant_approved`
- `continuation_quant_reduced`
- `continuation_quant_blocked`
- `continuation_rr_below_disaster_floor`
- `continuation_ev_negative`
- `continuation_execution_penalty_too_high`

These reasons should be visible in logs so reviewers can distinguish:

- poor execution quality
- weak continuation probability
- negative continuation EV
- strict quant block

## Integration Points

Files that should change:

- `src/strategy/execution_cost.py`
  - add continuation probability
  - add dynamic `R:R` floor
  - add continuation EV
  - return richer reasons and metrics

- `src/strategy/decision_tree.py`
  - emit richer continuation metadata

- `src/strategy/quant_engine.py`
  - support continuation-aware quant states
  - differentiate fresh-entry and continuation paths

- `src/live_trade_loop.py`
  - log continuation-specific quant and execution reasons cleanly

## Testing Strategy

Required tests:

- strong continuation with `effective_rr < 1.5` but positive continuation EV is allowed
- continuation with `effective_rr < 0.60` is still rejected
- dynamic `R:R` floor tightens and relaxes as expected
- continuation quant can approve, reduce, or block
- fresh-entry quant behavior remains strict
- runtime logs continuation-specific reasons instead of generic flatness

## Success Criteria

This redesign is successful when:

1. strong continuation trades with modest immediate `R:R` are no longer automatically rejected
2. weak continuation trades are still blocked by disaster-floor or negative-EV logic
3. quant flattening becomes continuation-aware rather than binary
4. live logs expose why continuation trades were approved, reduced, or blocked
5. the resulting model reads as explicit, auditable quant-finance logic rather than ad hoc threshold tweaks
