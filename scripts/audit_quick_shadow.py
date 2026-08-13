from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.quick_scalp_loop import QUICK_SHADOW_JOURNAL_FILE, resolve_quick_shadow_edge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit resolved no-trade quick shadow decisions.")
    parser.add_argument("--journal", default=QUICK_SHADOW_JOURNAL_FILE)
    parser.add_argument("--symbol", default="")
    parser.add_argument("--min-samples", type=int, default=60)
    parser.add_argument("--min-win-rate", type=float, default=0.58)
    parser.add_argument("--min-expectancy-proxy", type=float, default=0.02)
    parser.add_argument("--max-loss-streak", type=int, default=4)
    parser.add_argument("--output", default="data/quick_shadow_audit.json")
    args = parser.parse_args(argv)

    settings = load_settings()
    symbol = args.symbol or settings.mt4_chart_symbol or settings.trading_symbol
    edge = resolve_quick_shadow_edge(
        path=args.journal,
        symbol=symbol,
        min_samples=args.min_samples,
        min_win_rate=args.min_win_rate,
        min_expectancy_proxy=args.min_expectancy_proxy,
        max_loss_streak=args.max_loss_streak,
    )
    report = {
        "symbol": symbol,
        "journal": str(Path(args.journal)),
        "allowed": edge.allowed,
        "reason": edge.reason,
        "sample_count": edge.sample_count,
        "win_rate": edge.win_rate,
        "avg_favorable": edge.avg_favorable,
        "avg_adverse": edge.avg_adverse,
        "expectancy_proxy": edge.expectancy_proxy,
        "max_loss_streak": edge.max_loss_streak,
        "minimum_before_live_consideration": {
            "sample_count": args.min_samples,
            "win_rate": args.min_win_rate,
            "expectancy_proxy": args.min_expectancy_proxy,
            "max_loss_streak": args.max_loss_streak,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
