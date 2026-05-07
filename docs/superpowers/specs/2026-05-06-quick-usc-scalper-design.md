# Quick USC Scalper Design

## Goal

Add a separate quick scalping runtime for USC accounts. The main `run.py` strategy remains for larger accounts, while `quick.py` forces quick scalping on from the same `.env`.

## Behavior

`quick.py` initializes MT5 with the configured HFM login, reads closed M1 candles, opens quick trades in the latest M1 candle direction, and closes quick trades as soon as floating profit is greater than the target. The default profit target is `0.2` account currency units, so that can mean greater than 0.2 USD on a USD account or greater than 20 KSH on a KSH account if configured that way.

Quick trades use the comment prefix `quick-scalp`, so they do not manage or close existing `strategy-live` trades from the main bot.

## Risk And Capacity

The quick loop can open up to `QUICK_MAX_POSITIONS`, default `100`, but it stops earlier if the account cannot support more margin or MT5 rejects an order. Capacity is based on free margin when MT5 exposes `account_info` and `order_calc_margin`.

## Settings

The existing `.env` is reused. `quick.py` forces `quick_scalp_enabled=True` in memory. `run.py` leaves quick mode off.

Settings:

- `QUICK_SCALP_ENABLED`, default `false`
- `QUICK_TRADE_LOT`, default `DEFAULT_TRADE_LOT`
- `QUICK_MAX_POSITIONS`, default `100`
- `QUICK_PROFIT_TARGET`, default `0.2`
- `QUICK_POLL_SECONDS`, default `1`
- `QUICK_MIN_FREE_MARGIN`, default `0.0`

## Testing

Tests cover loading quick settings, detecting M1 direction from closed candles, closing profitable quick positions, opening up to the configured maximum, stopping when margin is insufficient, and verifying the runner forces quick mode on.
