import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone

def diagnostic():
    print("Fetching data...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=15)
    
    # Use download for better reliability
    df = yf.download("GC=F", start=start, end=end, interval="1m", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    print(f"Total bars fetched: {len(df)}")
    
    # Check for volatility
    df['returns'] = df['Close'].diff()
    print(f"Average absolute 1m move: {df['returns'].abs().mean():.4f}")
    print(f"Max 1m move: {df['returns'].abs().max():.4f}")
    
    # Check for gaps
    df['gap'] = df.index.to_series().diff().dt.total_seconds()
    gaps = df[df['gap'] > 60]
    print(f"Number of gaps (>1min): {len(gaps)}")

diagnostic()
