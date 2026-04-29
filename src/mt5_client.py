from __future__ import annotations

from pathlib import Path
import subprocess
import time


class Mt5Session:
    def __init__(self, terminal_path, startup_wait_seconds, subprocess_module=None, sleep_fn=None, mt5_module=None):
        self.terminal_path = Path(terminal_path)
        self.startup_wait_seconds = startup_wait_seconds
        self.subprocess_module = subprocess_module or subprocess
        self.sleep_fn = sleep_fn or time.sleep
        self.mt5_module = mt5_module

    def launch_terminal(self):
        if not self.terminal_path.exists():
            raise FileNotFoundError(f"MT5 terminal not found: {self.terminal_path}")
        self.subprocess_module.Popen([str(self.terminal_path)])
        self.sleep_fn(self.startup_wait_seconds)

    def initialize_and_login(self, login, password, server):
        if self.mt5_module is None:
            raise RuntimeError("MT5 module is not configured")
        if not self.mt5_module.initialize():
            code, message = self.mt5_module.last_error()
            raise RuntimeError(f"MT5 initialize failed: {code} {message}")
        if not self.mt5_module.login(login, password=password, server=server):
            code, message = self.mt5_module.last_error()
            raise RuntimeError(f"HFM login failed: {code} {message}")
        return getattr(self.mt5_module, "account_info", lambda: None)()

    def shutdown(self):
        if self.mt5_module is not None:
            self.mt5_module.shutdown()
