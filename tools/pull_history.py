import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import load_settings
from src.mt4_bridge import install_bridge_ea, Mt4BridgeModule

def main():
    print("=" * 130)
    print("                                                  MT4 BOT HISTORY PULLER")
    print("=" * 130)
    
    settings = load_settings()
    
    # 1. Install/Update the EA in MT4
    print("Checking MT4 bridge EA installation...")
    try:
        install = install_bridge_ea(project_root, getattr(settings, "mt4_data_path", ""))
        print(f"[OK] EA copied to MetaTrader 4: {install.destination}")
        print(f"[OK] Common files directory: {install.common_files_dir}")
    except Exception as e:
        print(f"[WARN] Could not auto-install EA: {e}")
        print("Reading history from default common files folder...")

    common_files_dir = Path(os.environ.get("APPDATA", "")) / "MetaQuotes" / "Terminal" / "Common" / "Files"
    bridge = Mt4BridgeModule(common_files_dir)
    
    # 2. Read history
    print("\nReading trade history from MT4 bridge...")
    history = bridge.history_get()
    
    if not history:
        print("\nNo historical trades found in MT4 history pool.")
        print("Note: Ensure the FxPythonBridge EA is running on an active MT4 chart,")
        print("      AutoTrading is enabled, and there are closed trades in the Account History tab.")
        print("=" * 130)
        return
        
    print(f"\nFound {len(history)} historical trades:")
    print("-" * 130)
    print(f"{'Ticket':<10} | {'Symbol':<8} | {'Type':<6} | {'Lots':<6} | {'Open Price':<10} | {'Close Price':<10} | {'PnL':<8} | {'Magic':<10} | {'Open Time':<19} | {'Close Time':<19}")
    print("-" * 130)
    
    total_pnl = 0.0
    wins = 0
    losses = 0
    
    for deal in history:
        type_str = "BUY" if deal.type == 0 else "SELL"
        pnl = deal.profit
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
            
        magic = getattr(deal, "magic_number", 0)
        print(f"{deal.ticket:<10} | {deal.symbol:<8} | {type_str:<6} | {deal.volume:<6.2f} | {deal.price_open:<10.4f} | {deal.price_close:<10.4f} | {pnl:<8.2f} | {magic:<10} | {deal.open_time:<19} | {deal.close_time:<19}")
        
    print("-" * 130)
    total_trades = len(history)
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    print(f"Total Trades: {total_trades}")
    print(f"Net Profit/Loss: {total_pnl:.2f}")
    print(f"Win Rate: {win_rate:.2f}% (Wins: {wins}, Losses: {losses})")
    print("=" * 130)

if __name__ == '__main__':
    main()
