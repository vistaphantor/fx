from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.quick_scalp_loop import QUICK_SHADOW_JOURNAL_FILE
from src.quick_shadow_trainer import load_resolved_shadow_rows, save_shadow_policy_report, train_shadow_policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an explainable quick-scalp policy from resolved shadow decisions.")
    parser.add_argument("--journal", default=QUICK_SHADOW_JOURNAL_FILE)
    parser.add_argument("--symbol", default="")
    parser.add_argument("--min-samples", type=int, default=60)
    parser.add_argument("--min-selected", type=int, default=20)
    parser.add_argument("--min-win-rate", type=float, default=0.58)
    parser.add_argument("--min-expectancy", type=float, default=0.02)
    parser.add_argument("--min-profit-factor", type=float, default=1.20)
    parser.add_argument("--max-loss-streak", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--min-validation-selected", type=int, default=8)
    parser.add_argument("--output", default="data/quick_shadow_policy.json")
    args = parser.parse_args(argv)

    settings = load_settings()
    symbol = args.symbol or settings.mt4_chart_symbol or settings.trading_symbol
    rows = load_resolved_shadow_rows(args.journal, symbol=symbol)
    report = train_shadow_policy(
        rows,
        min_samples=args.min_samples,
        min_selected=args.min_selected,
        min_win_rate=args.min_win_rate,
        min_expectancy=args.min_expectancy,
        min_profit_factor=args.min_profit_factor,
        max_loss_streak=args.max_loss_streak,
        validation_fraction=args.validation_fraction,
        min_validation_selected=args.min_validation_selected,
    )
    output = Path(args.output)
    save_shadow_policy_report(report, output)
    payload = report.to_dict()
    payload["symbol"] = symbol
    payload["journal"] = str(Path(args.journal))
    payload["output"] = str(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
