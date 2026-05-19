"""
session_engine.py
-----------------
Handles time-of-day filtering and institutional session gating.
"""

from datetime import datetime, time, timezone

class SessionEngine:
    def __init__(self):
        # Define institutional sessions (UTC)
        self.london_open = time(8, 0)
        self.london_close = time(16, 0)
        self.ny_open = time(12, 0)
        self.ny_close = time(20, 0)
        self.overlap_start = time(12, 0)
        self.overlap_end = time(16, 0)

    def compute_session_score(self, current_dt: datetime) -> float:
        """Returns a score between 0.2 (dead) and 1.0 (peak liquidity)."""
        # Ensure UTC
        if current_dt.tzinfo is None:
            current_dt = current_dt.replace(tzinfo=timezone.utc)
        else:
            current_dt = current_dt.astimezone(timezone.utc)
            
        now_time = current_dt.time()
        
        # 1. Peak Overlap (London + NY)
        if self.overlap_start <= now_time <= self.overlap_end:
            return 1.0
            
        # 2. Main Sessions
        if self.london_open <= now_time <= self.ny_close:
            return 0.8
            
        # 3. Asian Session (Higher noise)
        if time(0, 0) <= now_time <= time(8, 0):
            return 0.4
            
        # 4. Dead Zone (Post NY Close)
        return 0.2

    def is_tradeable(self, current_dt: datetime) -> bool:
        score = self.compute_session_score(current_dt)
        return score >= 0.4
