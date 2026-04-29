import pytest

from src.config import load_settings


def test_load_settings_reads_live_only_required_values(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "MT5_TERMINAL_PATH=C:\\Program Files\\MT5\\terminal64.exe",
                "HFM_LOGIN=123456",
                "HFM_PASSWORD=secret",
                "HFM_SERVER=HFMarketsGlobal-Demo",
                "TRADING_SYMBOL=XAUUSD",
                "MT5_STARTUP_WAIT_SECONDS=10",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_path)

    assert settings.mt5_terminal_path.endswith("terminal64.exe")
    assert settings.hfm_login == 123456
    assert settings.hfm_password == "secret"
    assert settings.hfm_server == "HFMarketsGlobal-Demo"
    assert settings.trading_symbol == "XAUUSD"
    assert settings.mt5_startup_wait_seconds == 10
    assert settings.loop_poll_seconds == 60
    assert settings.max_live_loops is None
    assert settings.default_trade_lot == pytest.approx(0.01)
    assert settings.add_on_lot_increment == pytest.approx(0.01)
    assert settings.campaign_max_exposure_pct == pytest.approx(10.0)


def test_load_settings_reads_live_loop_and_campaign_values(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "MT5_TERMINAL_PATH=C:\\Program Files\\MT5\\terminal64.exe",
                "HFM_LOGIN=123456",
                "HFM_PASSWORD=secret",
                "HFM_SERVER=HFMarketsGlobal-Demo",
                "TRADING_SYMBOL=XAUUSD",
                "MT5_STARTUP_WAIT_SECONDS=10",
                "LOOP_POLL_SECONDS=15",
                "MAX_LIVE_LOOPS=2",
                "DEFAULT_TRADE_LOT=0.02",
                "ADD_ON_LOT_INCREMENT=0.03",
                "CAMPAIGN_MAX_EXPOSURE_PCT=8.5",
                "RISK_BUFFER=0.08",
                "MAX_CANDLES_SINCE_BREAKOUT=4",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_path)

    assert settings.loop_poll_seconds == 15
    assert settings.max_live_loops == 2
    assert settings.default_trade_lot == pytest.approx(0.02)
    assert settings.add_on_lot_increment == pytest.approx(0.03)
    assert settings.campaign_max_exposure_pct == pytest.approx(8.5)
    assert settings.risk_buffer == pytest.approx(0.08)
    assert settings.max_candles_since_breakout == 4


def test_load_settings_reads_tradingview_webhook_settings(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "MT5_TERMINAL_PATH=C:\\Program Files\\MT5\\terminal64.exe",
                "HFM_LOGIN=123456",
                "HFM_PASSWORD=secret",
                "HFM_SERVER=HFMarketsGlobal-Demo",
                "TRADING_SYMBOL=XAUUSD",
                "MT5_STARTUP_WAIT_SECONDS=10",
                "TRADINGVIEW_WEBHOOK_ENABLED=true",
                "TRADINGVIEW_WEBHOOK_HOST=127.0.0.1",
                "TRADINGVIEW_WEBHOOK_PORT=8080",
                "TRADINGVIEW_WEBHOOK_SECRET=tv-secret",
                "TRADINGVIEW_ALERT_MAX_AGE_SECONDS=900",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_path)

    assert settings.tradingview_webhook_enabled is True
    assert settings.tradingview_webhook_host == "127.0.0.1"
    assert settings.tradingview_webhook_port == 8080
    assert settings.tradingview_webhook_secret == "tv-secret"
    assert settings.tradingview_alert_max_age_seconds == 900


def test_load_settings_reads_symbol_specific_thresholds(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "MT5_TERMINAL_PATH=C:\\Program Files\\MT5\\terminal64.exe",
                "HFM_LOGIN=123456",
                "HFM_PASSWORD=secret",
                "HFM_SERVER=HFMarketsGlobal-Demo",
                "TRADING_SYMBOL=XAUUSD",
                "MT5_STARTUP_WAIT_SECONDS=10",
                "XAUUSD_MIN_EDGE_THRESHOLD=2.4",
                "XAUUSD_MAX_UNCERTAINTY_THRESHOLD=1.3",
                "XAUUSD_MIN_EXPECTED_MOVE_MULTIPLE=2.1",
                "XAUUSD_ADD_ON_EDGE_MULTIPLIER=1.35",
                "XAUUSD_TREND_REGIME_WEIGHT=1.2",
                "XAUUSD_COMPRESSION_REGIME_WEIGHT=0.65",
                "EURJPY_MIN_EDGE_THRESHOLD=1.8",
                "EURJPY_MAX_UNCERTAINTY_THRESHOLD=0.9",
                "EURJPY_MIN_EXPECTED_MOVE_MULTIPLE=1.7",
                "EURJPY_ADD_ON_EDGE_MULTIPLIER=1.5",
                "EURJPY_TREND_REGIME_WEIGHT=1.1",
                "EURJPY_COMPRESSION_REGIME_WEIGHT=0.8",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_path)

    xau_profile = settings.strategy_profiles["XAUUSD"]
    eurjpy_profile = settings.strategy_profiles["EURJPY"]

    assert xau_profile.symbol == "XAUUSD"
    assert xau_profile.min_edge_threshold == pytest.approx(2.4)
    assert xau_profile.max_uncertainty_threshold == pytest.approx(1.3)
    assert xau_profile.minimum_expected_move_multiple == pytest.approx(2.1)
    assert xau_profile.add_on_edge_multiplier == pytest.approx(1.35)
    assert xau_profile.trend_regime_weight == pytest.approx(1.2)
    assert xau_profile.compression_regime_weight == pytest.approx(0.65)
    assert xau_profile.breakeven_distance == pytest.approx(1.5)
    assert xau_profile.campaign_base_add_trigger_r == pytest.approx(1.5)
    assert xau_profile.campaign_add_trigger_floor_r == pytest.approx(1.25)
    assert xau_profile.campaign_add_trigger_ceiling_r == pytest.approx(1.75)

    assert eurjpy_profile.symbol == "EURJPY"
    assert eurjpy_profile.min_edge_threshold == pytest.approx(1.8)
    assert eurjpy_profile.max_uncertainty_threshold == pytest.approx(0.9)
    assert eurjpy_profile.minimum_expected_move_multiple == pytest.approx(1.7)
    assert eurjpy_profile.add_on_edge_multiplier == pytest.approx(1.5)
    assert eurjpy_profile.trend_regime_weight == pytest.approx(1.1)
    assert eurjpy_profile.compression_regime_weight == pytest.approx(0.8)
    assert eurjpy_profile.breakeven_distance == pytest.approx(0.15)
    assert eurjpy_profile.campaign_base_add_trigger_r == pytest.approx(1.5)
    assert eurjpy_profile.campaign_add_trigger_floor_r == pytest.approx(1.25)
    assert eurjpy_profile.campaign_add_trigger_ceiling_r == pytest.approx(1.75)


def test_load_settings_reads_symbol_breakeven_distance(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "MT5_TERMINAL_PATH=C:\\Program Files\\MT5\\terminal64.exe",
                "HFM_LOGIN=123456",
                "HFM_PASSWORD=secret",
                "HFM_SERVER=HFMarketsGlobal-Demo",
                "TRADING_SYMBOL=XAUUSD",
                "MT5_STARTUP_WAIT_SECONDS=10",
                "XAUUSD_BREAKEVEN_DISTANCE=1.5",
                "XAUUSD_CAMPAIGN_BASE_ADD_TRIGGER_R=1.5",
                "XAUUSD_CAMPAIGN_ADD_TRIGGER_FLOOR_R=1.25",
                "XAUUSD_CAMPAIGN_ADD_TRIGGER_CEILING_R=1.75",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_path)

    xau_profile = settings.strategy_profiles["XAUUSD"]
    assert xau_profile.breakeven_distance == pytest.approx(1.5)
    assert xau_profile.campaign_base_add_trigger_r == pytest.approx(1.5)
    assert xau_profile.campaign_add_trigger_floor_r == pytest.approx(1.25)
    assert xau_profile.campaign_add_trigger_ceiling_r == pytest.approx(1.75)


def test_load_settings_requires_webhook_secret_when_webhook_enabled(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "MT5_TERMINAL_PATH=C:\\Program Files\\MT5\\terminal64.exe",
                "HFM_LOGIN=123456",
                "HFM_PASSWORD=secret",
                "HFM_SERVER=HFMarketsGlobal-Demo",
                "TRADING_SYMBOL=XAUUSD",
                "MT5_STARTUP_WAIT_SECONDS=10",
                "TRADINGVIEW_WEBHOOK_ENABLED=true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="TRADINGVIEW_WEBHOOK_SECRET"):
        load_settings(env_path)


def test_load_settings_rejects_invalid_live_numbers(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "MT5_TERMINAL_PATH=C:\\Program Files\\MT5\\terminal64.exe",
                "HFM_LOGIN=123456",
                "HFM_PASSWORD=secret",
                "HFM_SERVER=HFMarketsGlobal-Demo",
                "TRADING_SYMBOL=XAUUSD",
                "MT5_STARTUP_WAIT_SECONDS=10",
                "DEFAULT_TRADE_LOT=0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="DEFAULT_TRADE_LOT"):
        load_settings(env_path)


def test_load_settings_rejects_negative_uncertainty_threshold(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "MT5_TERMINAL_PATH=C:\\Program Files\\MT5\\terminal64.exe",
                "HFM_LOGIN=123456",
                "HFM_PASSWORD=secret",
                "HFM_SERVER=HFMarketsGlobal-Demo",
                "TRADING_SYMBOL=XAUUSD",
                "MT5_STARTUP_WAIT_SECONDS=10",
                "XAUUSD_MAX_UNCERTAINTY_THRESHOLD=-0.1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="XAUUSD_MAX_UNCERTAINTY_THRESHOLD"):
        load_settings(env_path)


def test_load_settings_rejects_non_positive_campaign_add_trigger(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "MT5_TERMINAL_PATH=C:\\Program Files\\MT5\\terminal64.exe",
                "HFM_LOGIN=123456",
                "HFM_PASSWORD=secret",
                "HFM_SERVER=HFMarketsGlobal-Demo",
                "TRADING_SYMBOL=XAUUSD",
                "MT5_STARTUP_WAIT_SECONDS=10",
                "XAUUSD_CAMPAIGN_BASE_ADD_TRIGGER_R=0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="XAUUSD_CAMPAIGN_BASE_ADD_TRIGGER_R"):
        load_settings(env_path)
