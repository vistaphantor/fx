from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.mt4_bridge import Mt4BridgeModule, install_bridge_ea
from src.quick_scalp_loop import QUICK_COMMENT_PREFIX


@dataclass(frozen=True)
class QuickDeal:
    ticket: int
    symbol: str
    volume: float
    order_type: int
    price_open: float
    price_close: float
    profit: float
    open_time: str
    close_time: str
    comment: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit real MT4 quick-scalp broker history.")
    parser.add_argument("--symbol", default="")
    parser.add_argument("--date", default="", help="MT4 date prefix, e.g. 2026.08.13. Defaults to today UTC.")
    parser.add_argument("--common-files-dir", default="")
    parser.add_argument("--output", default="data/quick_history_audit.json")
    args = parser.parse_args(argv)

    settings = load_settings()
    symbol = args.symbol or settings.mt4_chart_symbol or settings.trading_symbol
    date_prefix = args.date or datetime.now(timezone.utc).strftime("%Y.%m.%d")
    common_dir = Path(args.common_files_dir) if args.common_files_dir else _resolve_common_dir(settings)
    history_path = common_dir / "fx_bridge_history.csv"
    deals = load_quick_deals(history_path, symbol=symbol, date_prefix=date_prefix)
    report = build_report(deals, symbol=symbol, date_prefix=date_prefix, history_path=history_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def load_quick_deals(path: str | Path, *, symbol: str, date_prefix: str) -> list[QuickDeal]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    deals: list[QuickDeal] = []
    for line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 12:
            continue
        if parts[1] != symbol:
            continue
        if not parts[10].startswith(date_prefix):
            continue
        if not parts[11].startswith(QUICK_COMMENT_PREFIX):
            continue
        try:
            deals.append(
                QuickDeal(
                    ticket=int(float(parts[0] or 0)),
                    symbol=parts[1],
                    volume=float(parts[2] or 0.0),
                    order_type=int(float(parts[3] or 0)),
                    price_open=float(parts[4] or 0.0),
                    price_close=float(parts[5] or 0.0),
                    profit=float(parts[8] or 0.0),
                    open_time=parts[9],
                    close_time=parts[10],
                    comment=parts[11],
                )
            )
        except ValueError:
            continue
    return deals


def build_report(deals: list[QuickDeal], *, symbol: str, date_prefix: str, history_path: Path) -> dict:
    profits = [deal.profit for deal in deals]
    wins = [value for value in profits if value > 0.0]
    losses = [value for value in profits if value < 0.0]
    buy_deals = [deal for deal in deals if deal.order_type == 0]
    sell_deals = [deal for deal in deals if deal.order_type == 1]
    total = sum(profits)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    win_rate = len(wins) / len(deals) if deals else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0)
    max_loss_streak = _max_loss_streak(profits)
    expectancy = total / len(deals) if deals else 0.0
    return {
        "symbol": symbol,
        "date": date_prefix,
        "history_path": str(history_path),
        "trade_count": len(deals),
        "net_profit": round(total, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": max(wins) if wins else 0.0,
        "largest_loss": min(losses) if losses else 0.0,
        "max_loss_streak": max_loss_streak,
        "buy": _side_report(buy_deals),
        "sell": _side_report(sell_deals),
        "recommendation": _recommendation(
            trade_count=len(deals),
            net_profit=total,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_loss_streak=max_loss_streak,
        ),
        "recent_trades": [
            {
                "ticket": deal.ticket,
                "side": "BUY" if deal.order_type == 0 else "SELL",
                "profit": deal.profit,
                "open_time": deal.open_time,
                "close_time": deal.close_time,
                "comment": deal.comment,
            }
            for deal in deals[-20:]
        ],
    }


def _side_report(deals: list[QuickDeal]) -> dict:
    if not deals:
        return {"count": 0, "net_profit": 0.0, "win_rate": 0.0, "expectancy": 0.0}
    profits = [deal.profit for deal in deals]
    wins = [value for value in profits if value > 0.0]
    return {
        "count": len(deals),
        "net_profit": round(sum(profits), 2),
        "win_rate": len(wins) / len(deals),
        "expectancy": sum(profits) / len(deals),
    }


def _max_loss_streak(profits: list[float]) -> int:
    current = 0
    longest = 0
    for profit in profits:
        if profit < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _recommendation(
    *,
    trade_count: int,
    net_profit: float,
    win_rate: float,
    profit_factor: float,
    expectancy: float,
    max_loss_streak: int,
) -> dict:
    allow_live = (
        trade_count >= 30
        and net_profit > 0.0
        and win_rate >= 0.55
        and profit_factor >= 1.20
        and expectancy > 0.02
        and max_loss_streak <= 3
    )
    reason = "ok_to_continue" if allow_live else "negative_or_unproven_quick_edge"
    return {
        "allow_live_trading": allow_live,
        "reason": reason,
        "minimum_before_lift_daily_stop": {
            "trade_count": 30,
            "win_rate": 0.55,
            "profit_factor": 1.20,
            "expectancy_per_trade": 0.02,
            "max_loss_streak": 3,
        },
    }


def _resolve_common_dir(settings) -> Path:
    try:
        install = install_bridge_ea(Path(".").resolve(), getattr(settings, "mt4_data_path", ""))
        return install.common_files_dir
    except Exception:
        return Path(os.environ.get("APPDATA", "")) / "MetaQuotes" / "Terminal" / "Common" / "Files"


if __name__ == "__main__":
    raise SystemExit(main())
