from pathlib import Path

import pytest

from src.mt5_client import Mt4FallbackSession, Mt5Session


class FakeSubprocess:
    def __init__(self):
        self.calls = []

    def Popen(self, args):
        self.calls.append(args)
        return object()


class FakeMt5:
    def __init__(self, initialize_result=True, login_result=True):
        self.initialize_result = initialize_result
        self.login_result = login_result
        self.initialize_calls = []
        self.login_calls = []
        self.shutdown_called = False

    def initialize(self):
        self.initialize_calls.append(True)
        return self.initialize_result

    def login(self, login, password, server):
        self.login_calls.append((login, password, server))
        return self.login_result

    def shutdown(self):
        self.shutdown_called = True

    def last_error(self):
        return (500, "boom")

    def account_info(self):
        return None


def test_launch_terminal_raises_when_path_missing():
    session = Mt5Session(
        terminal_path=Path("missing.exe"),
        startup_wait_seconds=1,
        subprocess_module=FakeSubprocess(),
        sleep_fn=lambda _: None,
        mt5_module=FakeMt5(),
    )

    with pytest.raises(FileNotFoundError):
        session.launch_terminal()


def test_initialize_and_login_calls_mt5_after_launch(tmp_path):
    terminal = tmp_path / "terminal64.exe"
    terminal.write_text("", encoding="utf-8")
    fake_subprocess = FakeSubprocess()
    fake_mt5 = FakeMt5()
    session = Mt5Session(
        terminal_path=terminal,
        startup_wait_seconds=1,
        subprocess_module=fake_subprocess,
        sleep_fn=lambda _: None,
        mt5_module=fake_mt5,
    )

    session.launch_terminal()
    session.initialize_and_login(login=123, password="pw", server="demo")

    assert fake_subprocess.calls == [[str(terminal)]]
    assert fake_mt5.initialize_calls == [True]
    assert fake_mt5.login_calls == [(123, "pw", "demo")]


def test_mt4_fallback_launch_writes_login_config_and_starts_terminal(tmp_path):
    terminal = tmp_path / "terminal.exe"
    terminal.write_text("", encoding="utf-8")
    config_path = tmp_path / "mt4-login.ini"
    fake_subprocess = FakeSubprocess()
    session = Mt4FallbackSession(
        terminal_path=terminal,
        startup_wait_seconds=1,
        config_path=config_path,
        subprocess_module=fake_subprocess,
        sleep_fn=lambda _: None,
    )

    session.launch_terminal(login=123, password="pw", server="demo", symbol="XAUUSD", period="M1")

    assert fake_subprocess.calls == [[str(terminal), str(config_path.resolve())]]
    config_text = config_path.read_text(encoding="utf-8")
    assert "Login=123" in config_text
    assert "Password=pw" in config_text
    assert "Server=demo" in config_text
    assert "ExpertsEnable=true" in config_text
    assert "ExpertsTrades=true" in config_text
    assert "Symbol=XAUUSD" in config_text
    assert "Period=M1" in config_text
    assert "Expert=FxPythonBridge" in config_text
