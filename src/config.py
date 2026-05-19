from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class SymbolStrategyProfile:
    symbol: str
    min_edge_threshold: float
    max_uncertainty_threshold: float
    minimum_expected_move_multiple: float
    add_on_edge_multiplier: float
    trend_regime_weight: float
    compression_regime_weight: float
    breakeven_distance: float
    campaign_base_add_trigger_r: float
    campaign_add_trigger_floor_r: float
    campaign_add_trigger_ceiling_r: float


@dataclass(frozen=True)
class Settings:
    mt5_terminal_path: str
    hfm_login: int
    hfm_password: str
    hfm_server: str
    trading_symbol: str
    mt5_startup_wait_seconds: int
    loop_poll_seconds: int
    max_live_loops: int | None
    risk_buffer: float
    max_candles_since_breakout: int
    default_trade_lot: float
    add_on_lot_increment: float
    campaign_max_exposure_pct: float
    tradingview_webhook_enabled: bool
    tradingview_webhook_host: str
    tradingview_webhook_port: int
    tradingview_webhook_secret: str
    tradingview_alert_max_age_seconds: int
    strategy_profiles: dict[str, SymbolStrategyProfile]
    # Quant engine parameters
    quant_gamma: float = 2.0
    quant_cvar_alpha: float = 0.05
    quant_cvar_eta: float = 1.2          # Lowered from 1.5: reduces CVaR penalty weight
    quant_dd_max: float = 0.20
    quant_dd_rho: float = 0.5
    quant_omega_threshold: float = 0.45  # Lowered from 0.6: allows more setups to pass the omega gate
    quant_position_r_max: float = 0.05   # Raised from 0.02: positions can now be up to 5% risk
    quant_transaction_lambda: float = 1.0
    quant_zscore_window: int = 100
    quant_enabled: bool = True
    # ML settings
    ml_model_path: str = ""
    ml_enabled: bool = False
    feature_logging_enabled: bool = True
    feature_log_path: str = "data/features.jsonl"
    equity_log_path: str = "data/equity_history.jsonl"
    # Quick USC scalper settings
    quick_scalp_enabled: bool = False
    quick_trade_lot: float = 0.01
    quick_max_positions: int = 1
    quick_profit_target: float = 0.2
    quick_max_loss: float = 0.0
    quick_poll_seconds: int = 1
    quick_min_free_margin: float = 0.0
    quick_atr_sl_mult: float = 0.8
    quick_atr_tp_mult: float = 2.0
    quick_max_spread_pips: float = 2.0


_PROFILE_DEFAULTS = {
    "XAUUSD": {
        "min_edge_threshold": 2.5,
        "max_uncertainty_threshold": 10.0,
        "minimum_expected_move_multiple": 1.5,
        "add_on_edge_multiplier": 1.25,
        "trend_regime_weight": 1.1,
        "compression_regime_weight": 0.75,
        "breakeven_distance": 5.0,
        "campaign_base_add_trigger_r": 1.5,
        "campaign_add_trigger_floor_r": 1.25,
        "campaign_add_trigger_ceiling_r": 1.75,
    },
    "EURJPY": {
        "min_edge_threshold": 1.8,
        "max_uncertainty_threshold": 1.0,
        "minimum_expected_move_multiple": 1.6,
        "add_on_edge_multiplier": 1.35,
        "trend_regime_weight": 1.0,
        "compression_regime_weight": 0.8,
        "breakeven_distance": 0.15,
        "campaign_base_add_trigger_r": 1.5,
        "campaign_add_trigger_floor_r": 1.25,
        "campaign_add_trigger_ceiling_r": 1.75,
    },
}


def load_settings(env_path: str | Path = ".env") -> Settings:
    env_file = Path(env_path)
    values = {key: value for key, value in dotenv_values(env_file).items() if value is not None}

    if not values and env_file.name == ".env":
        values = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("MT5_")
            or key.startswith("HFM_")
            or key in {
                "TRADING_SYMBOL",
                "LOOP_POLL_SECONDS",
                "MAX_LIVE_LOOPS",
                "RISK_BUFFER",
                "MAX_CANDLES_SINCE_BREAKOUT",
                "DEFAULT_TRADE_LOT",
                "LIVE_TRADE_LOT",
                "ADD_ON_LOT_INCREMENT",
                "CAMPAIGN_MAX_EXPOSURE_PCT",
                "TRADINGVIEW_WEBHOOK_ENABLED",
                "TRADINGVIEW_WEBHOOK_HOST",
                "TRADINGVIEW_WEBHOOK_PORT",
                "TRADINGVIEW_WEBHOOK_SECRET",
                "TRADINGVIEW_ALERT_MAX_AGE_SECONDS",
                "QUANT_ENABLED",
                "QUANT_GAMMA",
                "QUANT_CVAR_ALPHA",
                "QUANT_CVAR_ETA",
                "QUANT_DD_MAX",
                "QUANT_DD_RHO",
                "QUANT_OMEGA_THRESHOLD",
                "QUANT_POSITION_R_MAX",
                "QUANT_TRANSACTION_LAMBDA",
                "QUANT_ZSCORE_WINDOW",
                "ML_ENABLED",
                "ML_MODEL_PATH",
                "FEATURE_LOGGING_ENABLED",
                "FEATURE_LOG_PATH",
                "EQUITY_LOG_PATH",
            }
            or key.startswith("XAUUSD_")
            or key.startswith("EURJPY_")
            or key.startswith("QUICK_")
        }

    required = [
        "MT5_TERMINAL_PATH",
        "HFM_LOGIN",
        "HFM_PASSWORD",
        "HFM_SERVER",
        "TRADING_SYMBOL",
        "MT5_STARTUP_WAIT_SECONDS",
    ]
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(f"Missing required settings: {', '.join(missing)}")

    try:
        login = int(str(values["HFM_LOGIN"]).strip())
        startup_wait = int(str(values["MT5_STARTUP_WAIT_SECONDS"]).strip())
        loop_poll_seconds = int(str(values.get("LOOP_POLL_SECONDS", "60")).strip())
        max_live_loops_value = str(values.get("MAX_LIVE_LOOPS", "")).strip()
        max_live_loops = int(max_live_loops_value) if max_live_loops_value else None
        risk_buffer = float(str(values.get("RISK_BUFFER", "0.05")).strip())
        max_candles_since_breakout = int(str(values.get("MAX_CANDLES_SINCE_BREAKOUT", "3")).strip())
        tradingview_webhook_port = int(str(values.get("TRADINGVIEW_WEBHOOK_PORT", "8080")).strip())
        tradingview_alert_max_age_seconds = int(str(values.get("TRADINGVIEW_ALERT_MAX_AGE_SECONDS", "900")).strip())
        default_trade_lot = float(
            str(values.get("DEFAULT_TRADE_LOT", values.get("LIVE_TRADE_LOT", "0.01"))).strip()
        )
        add_on_lot_increment = float(str(values.get("ADD_ON_LOT_INCREMENT", "0.01")).strip())
        campaign_max_exposure_pct = float(str(values.get("CAMPAIGN_MAX_EXPOSURE_PCT", "10.0")).strip())
        quick_trade_lot = float(str(values.get("QUICK_TRADE_LOT", default_trade_lot)).strip())
        quick_max_positions = int(str(values.get("QUICK_MAX_POSITIONS", "1")).strip())
        quick_profit_target = float(str(values.get("QUICK_PROFIT_TARGET", "0.2")).strip())
        quick_max_loss = float(str(values.get("QUICK_MAX_LOSS", "0.0")).strip())
        quick_poll_seconds = int(str(values.get("QUICK_POLL_SECONDS", "1")).strip())
        quick_min_free_margin = float(str(values.get("QUICK_MIN_FREE_MARGIN", "0.0")).strip())
        quick_atr_sl_mult = float(str(values.get("QUICK_ATR_SL_MULT", "0.8")).strip())
        quick_atr_tp_mult = float(str(values.get("QUICK_ATR_TP_MULT", "2.0")).strip())
        quick_max_spread_pips = float(str(values.get("QUICK_MAX_SPREAD_PIPS", "2.0")).strip())
        strategy_profiles = _load_strategy_profiles(values)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric setting: {exc}") from exc

    if startup_wait < 0:
        raise ValueError("MT5_STARTUP_WAIT_SECONDS must be 0 or greater")
    if loop_poll_seconds <= 0:
        raise ValueError("LOOP_POLL_SECONDS must be greater than 0")
    if max_live_loops is not None and max_live_loops <= 0:
        raise ValueError("MAX_LIVE_LOOPS must be greater than 0 when set")
    if risk_buffer <= 0:
        raise ValueError("RISK_BUFFER must be greater than 0")
    if max_candles_since_breakout <= 0:
        raise ValueError("MAX_CANDLES_SINCE_BREAKOUT must be greater than 0")
    if not 0 < tradingview_webhook_port < 65536:
        raise ValueError("TRADINGVIEW_WEBHOOK_PORT must be between 1 and 65535")
    if tradingview_alert_max_age_seconds <= 0:
        raise ValueError("TRADINGVIEW_ALERT_MAX_AGE_SECONDS must be greater than 0")
    if default_trade_lot <= 0:
        raise ValueError("DEFAULT_TRADE_LOT must be greater than 0")
    if add_on_lot_increment <= 0:
        raise ValueError("ADD_ON_LOT_INCREMENT must be greater than 0")
    if campaign_max_exposure_pct <= 0:
        raise ValueError("CAMPAIGN_MAX_EXPOSURE_PCT must be greater than 0")
    if quick_trade_lot <= 0:
        raise ValueError("QUICK_TRADE_LOT must be greater than 0")
    if quick_max_positions <= 0:
        raise ValueError("QUICK_MAX_POSITIONS must be greater than 0")
    if quick_profit_target < 0:
        raise ValueError("QUICK_PROFIT_TARGET must be 0 or greater")
    if quick_max_loss < 0:
        raise ValueError("QUICK_MAX_LOSS must be 0 or greater")
    if quick_poll_seconds <= 0:
        raise ValueError("QUICK_POLL_SECONDS must be greater than 0")
    if quick_min_free_margin < 0:
        raise ValueError("QUICK_MIN_FREE_MARGIN must be 0 or greater")

    tradingview_webhook_enabled = (
        str(values.get("TRADINGVIEW_WEBHOOK_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
    )
    quick_scalp_enabled = (
        str(values.get("QUICK_SCALP_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
    )
    tradingview_webhook_secret = str(values.get("TRADINGVIEW_WEBHOOK_SECRET", "")).strip()
    if tradingview_webhook_enabled and not tradingview_webhook_secret:
        raise ValueError("TRADINGVIEW_WEBHOOK_SECRET is required when TRADINGVIEW_WEBHOOK_ENABLED=true")

    # Quant engine parameters
    quant_gamma = float(str(values.get("QUANT_GAMMA", "2.0")).strip())
    quant_cvar_alpha = float(str(values.get("QUANT_CVAR_ALPHA", "0.05")).strip())
    quant_cvar_eta = float(str(values.get("QUANT_CVAR_ETA", "1.2")).strip())
    quant_dd_max = float(str(values.get("QUANT_DD_MAX", "0.20")).strip())
    quant_dd_rho = float(str(values.get("QUANT_DD_RHO", "0.5")).strip())
    quant_omega_threshold = float(str(values.get("QUANT_OMEGA_THRESHOLD", "0.45")).strip())
    quant_position_r_max = float(str(values.get("QUANT_POSITION_R_MAX", "0.05")).strip())
    quant_transaction_lambda = float(str(values.get("QUANT_TRANSACTION_LAMBDA", "1.0")).strip())
    quant_zscore_window = int(str(values.get("QUANT_ZSCORE_WINDOW", "100")).strip())
    quant_enabled = str(values.get("QUANT_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
    ml_model_path = str(values.get("ML_MODEL_PATH", "")).strip()
    ml_enabled = str(values.get("ML_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
    feature_logging_enabled = str(values.get("FEATURE_LOGGING_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
    feature_log_path = str(values.get("FEATURE_LOG_PATH", "data/features.jsonl")).strip()
    equity_log_path = str(values.get("EQUITY_LOG_PATH", "data/equity_history.jsonl")).strip()

    return Settings(
        mt5_terminal_path=str(values["MT5_TERMINAL_PATH"]).strip(),
        hfm_login=login,
        hfm_password=str(values["HFM_PASSWORD"]).strip(),
        hfm_server=str(values["HFM_SERVER"]).strip(),
        trading_symbol=str(values["TRADING_SYMBOL"]).strip().replace("XAUUSDc", "XAUUSD") if "Demo" in str(values["HFM_SERVER"]) else str(values["TRADING_SYMBOL"]).strip(),
        mt5_startup_wait_seconds=startup_wait,
        loop_poll_seconds=loop_poll_seconds,
        max_live_loops=max_live_loops,
        risk_buffer=risk_buffer,
        max_candles_since_breakout=max_candles_since_breakout,
        default_trade_lot=default_trade_lot,
        add_on_lot_increment=add_on_lot_increment,
        campaign_max_exposure_pct=campaign_max_exposure_pct,
        tradingview_webhook_enabled=tradingview_webhook_enabled,
        tradingview_webhook_host=str(values.get("TRADINGVIEW_WEBHOOK_HOST", "127.0.0.1")).strip(),
        tradingview_webhook_port=tradingview_webhook_port,
        tradingview_webhook_secret=tradingview_webhook_secret,
        tradingview_alert_max_age_seconds=tradingview_alert_max_age_seconds,
        strategy_profiles=strategy_profiles,
        quant_gamma=quant_gamma,
        quant_cvar_alpha=quant_cvar_alpha,
        quant_cvar_eta=quant_cvar_eta,
        quant_dd_max=quant_dd_max,
        quant_dd_rho=quant_dd_rho,
        quant_omega_threshold=quant_omega_threshold,
        quant_position_r_max=quant_position_r_max,
        quant_transaction_lambda=quant_transaction_lambda,
        quant_zscore_window=quant_zscore_window,
        quant_enabled=quant_enabled,
        ml_model_path=ml_model_path,
        ml_enabled=ml_enabled,
        feature_logging_enabled=feature_logging_enabled,
        feature_log_path=feature_log_path,
        equity_log_path=equity_log_path,
        quick_scalp_enabled=quick_scalp_enabled,
        quick_trade_lot=quick_trade_lot,
        quick_max_positions=quick_max_positions,
        quick_profit_target=quick_profit_target,
        quick_max_loss=quick_max_loss,
        quick_poll_seconds=quick_poll_seconds,
        quick_min_free_margin=quick_min_free_margin,
        quick_atr_sl_mult=quick_atr_sl_mult,
        quick_atr_tp_mult=quick_atr_tp_mult,
        quick_max_spread_pips=quick_max_spread_pips,
    )


def _load_strategy_profiles(values: dict[str, str]) -> dict[str, SymbolStrategyProfile]:
    profiles: dict[str, SymbolStrategyProfile] = {}
    for symbol, defaults in _PROFILE_DEFAULTS.items():
        prefix = f"{symbol}_"
        profile = SymbolStrategyProfile(
            symbol=symbol,
            min_edge_threshold=_read_positive_float(values, f"{prefix}MIN_EDGE_THRESHOLD", defaults["min_edge_threshold"]),
            max_uncertainty_threshold=_read_positive_float(
                values,
                f"{prefix}MAX_UNCERTAINTY_THRESHOLD",
                defaults["max_uncertainty_threshold"],
            ),
            minimum_expected_move_multiple=_read_positive_float(
                values,
                f"{prefix}MIN_EXPECTED_MOVE_MULTIPLE",
                defaults["minimum_expected_move_multiple"],
            ),
            add_on_edge_multiplier=_read_positive_float(
                values,
                f"{prefix}ADD_ON_EDGE_MULTIPLIER",
                defaults["add_on_edge_multiplier"],
            ),
            trend_regime_weight=_read_positive_float(
                values,
                f"{prefix}TREND_REGIME_WEIGHT",
                defaults["trend_regime_weight"],
            ),
            compression_regime_weight=_read_positive_float(
                values,
                f"{prefix}COMPRESSION_REGIME_WEIGHT",
                defaults["compression_regime_weight"],
            ),
            breakeven_distance=_read_strict_positive_float(
                values,
                f"{prefix}BREAKEVEN_DISTANCE",
                defaults["breakeven_distance"],
            ),
            campaign_base_add_trigger_r=_read_strict_positive_float(
                values,
                f"{prefix}CAMPAIGN_BASE_ADD_TRIGGER_R",
                defaults["campaign_base_add_trigger_r"],
            ),
            campaign_add_trigger_floor_r=_read_strict_positive_float(
                values,
                f"{prefix}CAMPAIGN_ADD_TRIGGER_FLOOR_R",
                defaults["campaign_add_trigger_floor_r"],
            ),
            campaign_add_trigger_ceiling_r=_read_strict_positive_float(
                values,
                f"{prefix}CAMPAIGN_ADD_TRIGGER_CEILING_R",
                defaults["campaign_add_trigger_ceiling_r"],
            ),
        )
        if profile.campaign_add_trigger_floor_r > profile.campaign_base_add_trigger_r:
            raise ValueError(f"{prefix}CAMPAIGN_ADD_TRIGGER_FLOOR_R must be less than or equal to {prefix}CAMPAIGN_BASE_ADD_TRIGGER_R")
        if profile.campaign_base_add_trigger_r > profile.campaign_add_trigger_ceiling_r:
            raise ValueError(f"{prefix}CAMPAIGN_BASE_ADD_TRIGGER_R must be less than or equal to {prefix}CAMPAIGN_ADD_TRIGGER_CEILING_R")
        profiles[symbol] = profile
    return profiles


def _read_positive_float(values: dict[str, str], key: str, default: float) -> float:
    raw_value = str(values.get(key, default)).strip()
    value = float(raw_value)
    if value < 0:
        raise ValueError(f"{key} must be 0 or greater")
    return value


def _read_strict_positive_float(values: dict[str, str], key: str, default: float) -> float:
    raw_value = str(values.get(key, default)).strip()
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"{key} must be greater than 0")
    return value
