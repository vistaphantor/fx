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


class Mt4FallbackSession:
    def __init__(
        self,
        terminal_path,
        startup_wait_seconds,
        config_path=".runtime/mt4-login.ini",
        subprocess_module=None,
        sleep_fn=None,
    ):
        self.terminal_path = Path(terminal_path)
        self.startup_wait_seconds = startup_wait_seconds
        self.config_path = Path(config_path)
        self.subprocess_module = subprocess_module or subprocess
        self.sleep_fn = sleep_fn or time.sleep

    def launch_terminal(self, login, password, server, symbol="", period="M1", expert="FxPythonBridge"):
        if not self.terminal_path.exists():
            raise FileNotFoundError(f"MT4 terminal not found: {self.terminal_path}")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        config_lines = [
            "[Common]",
            f"Login={int(login)}",
            f"Password={password}",
            f"Server={server}",
            "AutoConfiguration=true",
            "ProxyEnable=false",
            "",
            "[Experts]",
            "Enabled=true",
            "AllowLiveTrading=true",
            "AllowDllImport=false",
            "AllowExternalExperts=true",
            "ExpertsEnable=true",
            "ExpertsTrades=true",
            "ExpertsDllImport=false",
            "ExpertsExpImport=true",
        ]
        if symbol:
            config_lines.extend(
                [
                    "",
                    "[Charts]",
                    f"Symbol={symbol}",
                    f"Period={period or 'M1'}",
                    f"Expert={expert}",
                ]
            )
        config_lines.append("")
        self.config_path.write_text("\n".join(config_lines), encoding="utf-8")
        self.subprocess_module.Popen([str(self.terminal_path), str(self.config_path.resolve())])
        self.sleep_fn(self.startup_wait_seconds)
