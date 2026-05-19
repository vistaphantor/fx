"""
One-shot patch: inserts the MT5-native orderflow polling block into
live_trade_loop.py, right before the orderflow_signal lookup on each loop.
Safe to run multiple times — checks for the marker before patching.
"""
import pathlib, sys

TARGET = pathlib.Path(__file__).parent.parent / "src" / "live_trade_loop.py"

MARKER = "# ── MT5 orderflow polling"

OLD = (
    "        orderflow_signal = None\n"
    "        if orderflow_signal_store is not None:\n"
    "            orderflow_signal = orderflow_signal_store.latest_for(\n"
    "                symbol,\n"
    '                now=getattr(live_input.m15_candles[-1], "timestamp", None),\n'
    "            )\n"
    "        strategy_result = evaluate_top_down_decision_tree("
)

NEW = (
    "        # ── MT5 orderflow polling ─────────────────────────────────────────\n"
    "        # Fetch latest ticks from MT5 and push into the engine every cycle.\n"
    "        # The computed signal is stored in orderflow_signal_store so the\n"
    "        # decision tree consumes it exactly as if it arrived via a webhook.\n"
    "        if _live_orderflow_engine is not None and orderflow_signal_store is not None:\n"
    "            try:\n"
    "                from src.quick_scalp_loop import fetch_recent_ticks\n"
    "                from src.strategy.orderflow import parse_orderflow_payload\n"
    "\n"
    "                raw_ticks = fetch_recent_ticks(mt5_module, symbol, count=500)\n"
    "                added = _live_orderflow_engine.ingest(raw_ticks)\n"
    "                if added > 0:\n"
    "                    payload = _live_orderflow_engine.to_signal_payload()\n"
    "                    if payload is not None:\n"
    "                        sig = parse_orderflow_payload(payload)\n"
    "                        orderflow_signal_store.record(sig)\n"
    "                        log_fn(\n"
    '                            f"ORDERFLOW {symbol} "\n'
    '                            f"ticks={len(_live_orderflow_engine._ticks)} "\n'
    "                            f\"delta={payload['delta']:.2f} \"\n"
    "                            f\"cvd_slope={payload['cvd_slope']:.3f} \"\n"
    "                            f\"vwap_bias={payload['vwap_bias']:.3f} \"\n"
    "                            f\"imbalance={payload['imbalance']:.3f}\"\n"
    "                        )\n"
    "            except Exception as _of_poll_err:\n"
    '                log_fn(f"ORDERFLOW POLL WARN {symbol} reason={_of_poll_err}")\n'
    "        # ──────────────────────────────────────────────────────────────────\n"
    "\n"
    "        orderflow_signal = None\n"
    "        if orderflow_signal_store is not None:\n"
    "            orderflow_signal = orderflow_signal_store.latest_for(\n"
    "                symbol,\n"
    '                now=getattr(live_input.m15_candles[-1], "timestamp", None),\n'
    "            )\n"
    "        strategy_result = evaluate_top_down_decision_tree("
)

src = TARGET.read_text(encoding="utf-8")
# Normalise line endings
src_lf = src.replace("\r\n", "\n")

if MARKER in src_lf:
    print("Already patched — nothing to do.")
    sys.exit(0)

if OLD not in src_lf:
    # Dump context around the lookup to help debug
    idx = src_lf.find("orderflow_signal = None")
    ctx = repr(src_lf[max(0, idx - 120): idx + 300])
    print(f"PATTERN NOT FOUND. Context around 'orderflow_signal = None':\n{ctx}")
    sys.exit(1)

patched = src_lf.replace(OLD, NEW, 1)
TARGET.write_text(patched, encoding="utf-8")
print(f"OK — orderflow polling block inserted into {TARGET}")
