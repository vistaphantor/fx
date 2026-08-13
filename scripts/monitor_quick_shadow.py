from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.quick_scalp_loop import BOT_STATE_FILE, QUICK_SHADOW_JOURNAL_FILE, QUICK_SHADOW_POLICY_FILE
from src.quick_shadow_monitor import build_quick_shadow_monitor_report, save_monitor_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monitor live quick-scalp shadow training without placing trades.")
    parser.add_argument("--journal", default=QUICK_SHADOW_JOURNAL_FILE)
    parser.add_argument("--bot-state", default=BOT_STATE_FILE)
    parser.add_argument("--policy", default=QUICK_SHADOW_POLICY_FILE)
    parser.add_argument("--positions", default="")
    parser.add_argument("--symbol", default="")
    parser.add_argument("--retrain-policy", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--min-validation-selected", type=int, default=8)
    parser.add_argument("--output", default="data/quick_shadow_monitor.json")
    args = parser.parse_args(argv)

    settings = load_settings()
    symbol = args.symbol or settings.mt4_chart_symbol or settings.trading_symbol
    positions_path = args.positions or _default_mt4_positions_path()
    report = build_quick_shadow_monitor_report(
        journal_path=args.journal,
        bot_state_path=args.bot_state,
        policy_path=args.policy,
        positions_path=positions_path,
        symbol=symbol,
        retrain_policy=args.retrain_policy,
        validation_fraction=args.validation_fraction,
        min_validation_selected=args.min_validation_selected,
    )
    save_monitor_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _default_mt4_positions_path() -> str:
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        return ""
    return str(Path(appdata) / "MetaQuotes" / "Terminal" / "Common" / "Files" / "fx_bridge_positions.csv")


if __name__ == "__main__":
    raise SystemExit(main())
