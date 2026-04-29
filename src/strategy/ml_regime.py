"""ML-powered regime classifier using LightGBM.

Provides a drop-in replacement for the rule-based ``classify_regime()``
function in ``regime.py``.  When a trained model is available, it predicts
regime probabilities and maps them to a ``RegimeState``.  When no model is
loaded, it transparently falls back to the rule-based classifier.

Training is performed **offline** via ``ml_trainer.py`` — this module
only handles inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.strategy.features import FeatureSnapshot
from src.strategy.regime import RegimeState

logger = logging.getLogger(__name__)

# Regime class labels in a fixed order for model output
REGIME_LABELS = ("expansion", "trend", "pullback", "compression")

# Default regime parameters per class — calibrated to match rule-based outputs
_REGIME_DEFAULTS: dict[str, dict[str, Any]] = {
    "expansion": {
        "tradable": True,
        "continuation_bias": 1.25,
        "reversion_bias": 0.55,
    },
    "trend": {
        "tradable": True,
        "continuation_bias": 1.1,
        "reversion_bias": 0.6,
    },
    "pullback": {
        "tradable": True,
        "continuation_bias": 0.95,
        "reversion_bias": 0.8,
    },
    "compression": {
        "tradable": False,
        "continuation_bias": 0.5,
        "reversion_bias": 0.85,
    },
}


def _features_to_array(features: FeatureSnapshot) -> list[float]:
    """Convert a FeatureSnapshot to a flat feature vector for the model.

    Returns 14 features: 7 raw + 7 z-scored values.
    """
    return [
        features.momentum_raw,
        features.trend_raw,
        features.volume_raw,
        features.order_block_raw,
        features.volatility_risk_raw,
        features.entry_distance_raw,
        features.spread_danger_raw,
        features.momentum_z,
        features.trend_z,
        features.volume_z,
        features.order_block_z,
        features.volatility_risk_z,
        features.entry_distance_z,
        features.spread_danger_z,
    ]


FEATURE_NAMES = [
    "momentum_raw", "trend_raw", "volume_raw", "order_block_raw",
    "volatility_risk_raw", "entry_distance_raw", "spread_danger_raw",
    "momentum_z", "trend_z", "volume_z", "order_block_z",
    "volatility_risk_z", "entry_distance_z", "spread_danger_z",
]


class MLRegimeClassifier:
    """LightGBM-based regime classifier with rule-based fallback.

    Parameters
    ----------
    model_path : str or Path or None
        Path to a saved LightGBM model file.  If ``None`` or the file
        does not exist, all predictions fall back to rule-based logic.
    fallback_fn : callable or None
        A callable that takes keyword arguments matching
        ``classify_regime()`` and returns a ``RegimeState``.
        Defaults to ``None`` (returns a neutral pullback state).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        fallback_fn: Any = None,
    ) -> None:
        self._model = None
        self._model_path = Path(model_path) if model_path else None
        self._fallback_fn = fallback_fn
        self._load_attempted = False

        if self._model_path and self._model_path.exists():
            self._try_load_model()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _try_load_model(self) -> None:
        """Attempt to load the LightGBM model."""
        self._load_attempted = True
        try:
            import lightgbm as lgb

            self._model = lgb.Booster(model_file=str(self._model_path))
            logger.info("ML regime model loaded from %s", self._model_path)
        except ImportError:
            logger.warning(
                "lightgbm not installed — ML regime classifier disabled. "
                "Install with: pip install lightgbm"
            )
        except Exception as exc:
            logger.warning("Failed to load ML regime model: %s", exc)

    def predict(self, features: FeatureSnapshot, **fallback_kwargs: Any) -> RegimeState:
        """Predict the market regime from features.

        If a model is loaded, it predicts regime probabilities and returns
        the most likely regime as a ``RegimeState``.  If no model, it
        delegates to the fallback function.

        Parameters
        ----------
        features : FeatureSnapshot
            Current bar features.
        **fallback_kwargs
            Keyword arguments forwarded to the fallback function if needed.
        """
        if self._model is not None:
            return self._predict_with_model(features)
        return self._predict_fallback(**fallback_kwargs)

    def _predict_with_model(self, features: FeatureSnapshot) -> RegimeState:
        """Run inference with the loaded LightGBM model."""
        try:
            feature_vector = [_features_to_array(features)]
            probabilities = self._model.predict(feature_vector)[0]

            # Map probabilities to regime labels
            if hasattr(probabilities, "__len__") and len(probabilities) == len(REGIME_LABELS):
                best_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
                confidence = float(probabilities[best_idx])
                regime_name = REGIME_LABELS[best_idx]
            else:
                # Binary or regression output — map to regime
                regime_name = "trend" if float(probabilities) > 0.5 else "pullback"
                confidence = abs(float(probabilities) - 0.5) * 2

            defaults = _REGIME_DEFAULTS.get(regime_name, _REGIME_DEFAULTS["pullback"])
            return RegimeState(
                name=f"ml_{regime_name}",
                tradable=defaults["tradable"],
                continuation_bias=defaults["continuation_bias"],
                reversion_bias=defaults["reversion_bias"],
                confidence=max(0.1, min(1.0, confidence)),
            )
        except Exception as exc:
            logger.warning("ML regime prediction failed, using fallback: %s", exc)
            return self._predict_fallback()

    def _predict_fallback(self, **kwargs: Any) -> RegimeState:
        """Delegate to the rule-based fallback."""
        if self._fallback_fn is not None and kwargs:
            try:
                return self._fallback_fn(**kwargs)
            except Exception:
                pass
        # Ultimate fallback: neutral pullback
        return RegimeState(
            name="pullback",
            tradable=True,
            continuation_bias=0.95,
            reversion_bias=0.8,
            confidence=0.62,
        )
