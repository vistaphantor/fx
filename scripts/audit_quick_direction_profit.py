from __future__ import annotations

import csv
import json
import os
from pathlib import Path


COMMON_FILES = Path(os.environ.get("APPDATA", "")) / "MetaQuotes" / "Terminal" / "Common" / "Files"
HISTORY_FILE = COMMON_FILES / "fx_bridge_history.csv"


def _side(order_type: str) -> str:
    return "BUY" if str(order_type).strip() == "0" else "SELL"


def _signed_price_move(side: str, open_price: float, close_price: float) -> float:
    if side == "BUY":
        return close_price - open_price
    return open_price - close_price


def main() -> int:
    if not HISTORY_FILE.exists():
        raise SystemExit(f"history file not found: {HISTORY_FILE}")

    trades = []
    with HISTORY_FILE.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 12:
                continue
            symbol = row[1]
            comment = row[11]
            if symbol != "XAUUSD" or not comment.startswith("quick-scalp"):
                continue
            side = _side(row[3])
            volume = float(row[2] or 0.0)
            open_price = float(row[4] or 0.0)
            close_price = float(row[5] or 0.0)
            profit = float(row[8] or 0.0)
            signed_move = _signed_price_move(side, open_price, close_price)
            move_profit_alignment = (
                "aligned"
                if (signed_move > 0 and profit > 0) or (signed_move < 0 and profit < 0) or (signed_move == 0 and profit == 0)
                else "misaligned"
            )
            trades.append(
                {
                    "ticket": int(float(row[0] or 0)),
                    "side": side,
                    "volume": volume,
                    "open_price": open_price,
                    "close_price": close_price,
                    "signed_price_move": signed_move,
                    "profit": profit,
                    "alignment": move_profit_alignment,
                    "open_time": row[9],
                    "close_time": row[10],
                    "comment": comment,
                }
            )

    summary = {
        "history_path": str(HISTORY_FILE),
        "trade_count": len(trades),
        "misaligned_count": sum(1 for trade in trades if trade["alignment"] == "misaligned"),
        "largest_losses": sorted(trades, key=lambda trade: trade["profit"])[:8],
        "recent_trades": trades[-12:],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
