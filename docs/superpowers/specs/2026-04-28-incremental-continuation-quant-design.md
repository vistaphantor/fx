# Incremental Continuation Quant Redesign

## Goal

Upgrade the quant layer so the bot stops flattening valid momentum continuations because of weak absolute-wealth utility math.

The redesigned model should:

- optimize incremental trade value instead of total normalized wealth
- replace raw generic expected return inputs with continuation-conditioned expectancy
- make downside tail risk directional to the active trade
- preserve the true blocking reason in runtime logs
- strengthen `M10` continuation scoring so execution and quant consume richer state
- remain easy to audit in quant-finance terms

## Problem Statement

Recent live behavior shows two recurring failure modes:

1. before warmup completes, the strategy often blocks at `m10_setup_not_ready`
2. after warmup, the quant layer often blocks with `master_equation_flat`

The current quant design is structurally too harsh because it evaluates utility on:

```text
W + trade_pnl - cost - tail_risk
```

with normalized `W = 1.0`.

This compresses the action utilities for `-1`, `0`, and `1` into values that are too close together, so `flat` wins too often unless the trade edge is extremely large.

At the same time, the strategy and execution layers are not feeding quant a sufficiently economic continuation signal:

- raw expected return is too small and too generic
- CVaR is not directional enough
- `M10` continuation remains closer to a threshold gate than a proper state score

The result is that the bot undervalues live momentum continuation even when market structure is supportive.

## Design Principles

The redesign should preserve risk discipline while aligning the math with how continuation trades actually earn money.

Required principles:

1. action selection must optimize incremental trade value, not wealth-level utility
2. continuation entries must be judged on continuation-conditioned expectancy
3. downside tail risk must be directional
4. runtime logs must preserve the true blocking layer
5. the model must remain decomposable into explicit, reviewable terms

## Scope

In scope:

- quant engine action-selection redesign
- continuation-conditioned expectancy inputs
- directional CVaR design
- runtime reason-preservation changes
- stronger `M10` continuation scoring
- explicit audit-facing logging metadata

Out of scope:

- retraining from historical trade archives
- multi-symbol portfolio optimization
- replacing the top-down strategy tree
- external venue routing

## Core Quant Redesign

### Existing Action Logic

Current quant behavior effectively optimizes utility on:

```text
U(a) = -exp(-gamma * (W + trade_term(a) - penalties(a)))
```

This is inappropriate for the current scale because:

- `W` dominates the exponent
- the incremental trade contribution is too small relative to `W`
- action separation becomes too weak

### New Action Logic

Replace wealth-level action selection with incremental certainty-equivalent action selection:

```text
CE(a) = a * w_t * mu_cont - lambda * c_t - eta * CVaR_dir - rho * DD_t
```

Where:

- `a in {-1, 0, 1}` is short, flat, or long
- `w_t` is the proposed trade fraction
- `mu_cont` is continuation-conditioned expected return
- `c_t` is total execution cost
- `CVaR_dir` is directional tail loss
- `DD_t` is drawdown penalty

Action selection becomes:

```text
A_t = argmax_a CE(a)
```

If a CARA-style transformation is still desired for reporting consistency, apply it to the certainty-equivalent trade value, not to wealth:

```text
U(a) = -exp(-gamma * CE(a))
```

This preserves risk aversion while making action choice sensitive to actual trade economics.

## Continuation-Conditioned Expectancy

The quant engine should no longer use a tiny generic expected return term.

Instead compute:

```text
mu_cont = p_cont * G_eff - (1 - p_cont) * L_eff - c_exec
```

Where:

- `p_cont` is continuation probability
- `G_eff` is effective gain left after execution degradation
- `L_eff` is effective loss if the continuation fails
- `c_exec` is execution cost after spread, slippage, and stop degradation

Interpretation:

- if continuation quality improves, `p_cont` rises
- if the move is late and room is small, `G_eff` falls
- if stop vulnerability rises, `L_eff` rises
- if execution degrades, `c_exec` rises

This makes the quant layer evaluate the actual continuation trade rather than a weak proxy.

## Continuation Probability Model

Continuation probability should be derived from measurable state:

```text
p_cont = sigmoid(
    b1 * M15_quality
  + b2 * M10_quality
  + b3 * M5_quality
  + b4 * range_expansion
  + b5 * body_efficiency
  + b6 * regime_confidence
  - b7 * execution_penalty
  - b8 * retrace_damage
)
```

Definitions:

- `M15_quality`: higher-timeframe continuation validity
- `M10_quality`: refinement and continuation integrity
- `M5_quality`: execution-layer continuation quality
- `range_expansion`: directional expansion against local baseline
- `body_efficiency`: body dominance versus wick-heavy noise
- `regime_confidence`: directional confidence from the regime layer
- `execution_penalty`: spread, slippage, and stop degradation
- `retrace_damage`: continuation pullback deterioration

All inputs should be normalized before aggregation.

## Directional Tail Risk

Current CVaR treatment is too blunt because it does not fully align with trade direction.

Replace it conceptually with:

```text
CVaR_dir = E[adverse directional loss | tail event]
```

Directional rule:

- for longs, evaluate lower-tail adverse excursion
- for shorts, evaluate upper-tail adverse excursion

This should use recent return samples transformed into adverse directional loss samples relative to the candidate trade.

The directional CVaR term should then flow into:

```text
CE(a) = a * w_t * mu_cont - lambda * c_t - eta * CVaR_dir - rho * DD_t
```

This gives the risk penalty a clearer economic meaning.

## Runtime Reason Preservation

The live runtime should stop replacing the true strategy block reason with a quant reason when no trade was already present.

Required rule:

- if the strategy tree returns `is_trade=False`, preserve the strategy reason such as `m10_setup_not_ready`
- quant should only override the final reason if the strategy actually wanted a trade

This prevents misleading logs such as:

- strategy no-trade at `M10`
- final log says `master_equation_flat`

The logging stack should help the operator identify the actual active bottleneck.

## M10 Continuation Scoring Redesign

`M10` already has a continuation helper, but live behavior suggests it is still too close to a gate.

It should become a richer state score using:

- retrace depth
- local structure intactness
- reclaim quality
- local slope persistence
- expansion decay

Suggested conceptual score:

```text
M10_quality = w1 * retrace_quality
            + w2 * structure_integrity
            + w3 * reclaim_quality
            + w4 * slope_persistence
            + w5 * expansion_persistence
            - w6 * retrace_damage
```

Desired behavior:

- shallow, orderly continuation pullbacks remain valid
- broken structure degrades the score rapidly
- momentum decay is visible before the final hard block

The output should remain explainable in metadata so the execution and quant layers can consume it directly.

## Code-Level Impact

### `src/strategy/quant_engine.py`

Required changes:

- replace wealth-level utility selection with incremental certainty-equivalent selection
- accept `mu_cont` and directional `CVaR_dir`
- preserve three-state continuation outcomes:
  - `continuation_quant_approved`
  - `continuation_quant_reduced`
  - `continuation_quant_blocked`
- expose certainty-equivalent components in metadata

Suggested metadata additions:

- `mu_cont`
- `cvar_dir`
- `ce_scores`
- `cost_penalty`
- `drawdown_penalty`
- `continuation_probability`

### `src/strategy/execution_cost.py`

Required changes:

- compute `G_eff`, `L_eff`, and `c_exec`
- compute continuation-conditioned `mu_cont`
- expose execution inputs required by the new quant math
- keep the continuation disaster floor already introduced

Suggested metadata additions:

- `effective_gain_remaining`
- `effective_loss_if_failed`
- `continuation_mu`
- `directional_tail_proxy`

### `src/live_trade_loop.py`

Required changes:

- preserve non-trade reasons from the strategy tree
- only invoke quant override messaging when the strategy produced a trade plan
- route continuation-conditioned quantities into quant evaluation

### `src/strategy/setup.py`

Required changes:

- strengthen `M10` continuation scoring
- expose richer `M10` continuation metadata even when not ready

Suggested metadata additions:

- `slope_persistence`
- `expansion_decay`
- `retrace_damage`

## Logging And Auditability

The redesigned runtime should emit terms that map directly to the model:

```text
QUANT XAUUSD p_cont=... mu_cont=... CVaR_dir=... CE={-1:...,0:...,1:...}
```

and when continuation is evaluated:

```text
QUANT CONTINUATION XAUUSD state=approved|reduced|blocked
```

This is preferable to a generic `master_equation_flat` because it makes the decision interpretable:

- continuation probability
- continuation-conditioned expectancy
- directional tail loss
- certainty-equivalent action values

## Verification Plan

Required regression coverage:

1. `quant_engine`
   - certainty-equivalent action logic prefers trade over flat when `mu_cont` is clearly positive
   - flat still wins when `mu_cont` is negative or directional tail risk is too large
   - continuation reduced-size path still works

2. `execution_cost`
   - `mu_cont` rises when continuation quality improves
   - `mu_cont` falls when execution degrades
   - directional tail proxy behaves differently for long versus short

3. `live_trade_loop`
   - preserves `m10_setup_not_ready` when strategy already blocked
   - only logs quant override when a trade plan existed

4. `setup`
   - `M10` continuation score remains valid on shallow orderly pullbacks
   - broken structure degrades score and blocks

5. read-only live diagnostic
   - capture one current MT5 snapshot
   - report whether the active blocker is strategy, execution, or quant
   - confirm no live order or modification is sent

## Rollout Notes

This redesign should be implemented incrementally in this order:

1. rebase quant to certainty-equivalent trade utility
2. replace raw expected return with `mu_cont`
3. make directional CVaR available
4. fix runtime reason preservation
5. upgrade `M10` continuation scoring

This order minimizes the chance of mixing multiple behavioral shifts at once while preserving clear test boundaries.

## Success Criteria

The redesign is successful when:

- valid momentum continuation trades no longer flatten by construction
- `flat` still wins when continuation expectancy is truly weak
- logs reveal the real blocking layer
- `M10` contributes graded continuation quality instead of a mostly binary veto
- the math can be explained in a simple audit chain:
  - state extraction
  - continuation probability
  - continuation-conditioned expectancy
  - directional tail risk
  - certainty-equivalent action selection
