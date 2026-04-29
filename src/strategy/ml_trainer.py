"""Offline ML training pipeline for the regime classifier.

Reads feature snapshots from ``data/features.jsonl``, engineers labels
from forward returns, trains a LightGBM classifier using walk-forward
cross-validation, and saves the model + training report.

Usage:
    python -m src.strategy.ml_trainer                     # defaults
    python -m src.strategy.ml_trainer --features data/features.jsonl --output data/models/regime_classifier.lgb
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label engineering
# ---------------------------------------------------------------------------

def _compute_forward_return(rows: list[dict], index: int, horizon: int = 5) -> float | None:
    """Mean return over the next ``horizon`` bars."""
    if index + horizon >= len(rows):
        return None
    current_close = rows[index].get("momentum_raw", 0)
    future = rows[index + horizon].get("momentum_raw", 0)
    return future - current_close


def _classify_regime_label(
    forward_return: float,
    volatility_risk_raw: float,
    trend_raw: float,
) -> str:
    """Assign a regime label from forward-looking features.

    Heuristic labelling:
    - expansion: high volatility + strong trend + large forward return
    - trend: moderate volatility + directional
    - compression: low volatility + small forward return
    - pullback: everything else
    """
    abs_return = abs(forward_return)
    abs_trend = abs(trend_raw - 1.0)  # trend_raw is continuation_bias ~ 1.0

    if volatility_risk_raw >= 1.15 and abs_return > 0.5 and abs_trend > 0.1:
        return "expansion"
    if volatility_risk_raw <= 0.8 and abs_return < 0.2:
        return "compression"
    if abs_trend > 0.05 and abs_return > 0.1:
        return "trend"
    return "pullback"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_feature_rows(path: str | Path) -> list[dict]:
    """Load feature snapshots from JSONL."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    rows: list[dict] = []
    with open(file_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def prepare_dataset(rows: list[dict], horizon: int = 5) -> tuple[list[list[float]], list[int]]:
    """Prepare feature matrix X and label vector y.

    Returns
    -------
    X : list[list[float]]
        Feature vectors (14 features per sample).
    y : list[int]
        Regime labels as integers (expansion=0, trend=1, pullback=2, compression=3).
    """
    label_map = {"expansion": 0, "trend": 1, "pullback": 2, "compression": 3}

    X: list[list[float]] = []
    y: list[int] = []

    for i in range(len(rows) - horizon):
        row = rows[i]
        forward_ret = _compute_forward_return(rows, i, horizon)
        if forward_ret is None:
            continue

        label_str = _classify_regime_label(
            forward_return=forward_ret,
            volatility_risk_raw=row.get("volatility_risk_raw", 1.0),
            trend_raw=row.get("trend_raw", 1.0),
        )

        features = [
            row.get("momentum_raw", 0),
            row.get("trend_raw", 0),
            row.get("volume_raw", 0),
            row.get("order_block_raw", 0),
            row.get("volatility_risk_raw", 0),
            row.get("entry_distance_raw", 0),
            row.get("spread_danger_raw", 0),
            row.get("momentum_z", 0),
            row.get("trend_z", 0),
            row.get("volume_z", 0),
            row.get("order_block_z", 0),
            row.get("volatility_risk_z", 0),
            row.get("entry_distance_z", 0),
            row.get("spread_danger_z", 0),
        ]

        X.append(features)
        y.append(label_map[label_str])

    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X: list[list[float]],
    y: list[int],
    *,
    n_splits: int = 5,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    n_estimators: int = 200,
    min_child_samples: int = 10,
) -> tuple[Any, dict[str, Any]]:
    """Train a LightGBM classifier with walk-forward cross-validation.

    Returns the final model and a training report dict.
    """
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Training requires lightgbm and numpy. "
            "Install with: pip install lightgbm numpy"
        ) from exc

    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.int32)

    num_classes = len(set(y))
    if num_classes < 2:
        raise ValueError(f"Need at least 2 classes for training, got {num_classes}")

    from src.strategy.ml_regime import FEATURE_NAMES

    # Walk-forward cross-validation
    fold_size = len(X_arr) // n_splits
    fold_accuracies: list[float] = []
    fold_log_losses: list[float] = []

    for fold in range(1, n_splits):
        train_end = fold * fold_size
        val_end = min(train_end + fold_size, len(X_arr))

        if train_end >= len(X_arr) or val_end <= train_end:
            continue

        X_train, y_train = X_arr[:train_end], y_arr[:train_end]
        X_val, y_val = X_arr[train_end:val_end], y_arr[train_end:val_end]

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)

        params = {
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": ["multi_logloss", "multi_error"],
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "verbosity": -1,
            "seed": 42,
        }

        callbacks = [lgb.log_evaluation(period=0)]
        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # Evaluate
        preds = model.predict(X_val)
        pred_labels = np.argmax(preds, axis=1)
        accuracy = float(np.mean(pred_labels == y_val))
        fold_accuracies.append(accuracy)

        # Log loss
        eps = 1e-15
        preds_clipped = np.clip(preds, eps, 1 - eps)
        log_loss_val = -float(np.mean(
            np.log(preds_clipped[np.arange(len(y_val)), y_val])
        ))
        fold_log_losses.append(log_loss_val)

        logger.info(
            "Fold %d: accuracy=%.4f log_loss=%.4f (train=%d val=%d)",
            fold, accuracy, log_loss_val, len(X_train), len(X_val),
        )

    # Train final model on all data
    full_train_data = lgb.Dataset(X_arr, label=y_arr, feature_name=FEATURE_NAMES)
    final_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_child_samples": min_child_samples,
        "verbosity": -1,
        "seed": 42,
    }
    final_model = lgb.train(final_params, full_train_data, num_boost_round=n_estimators)

    # Feature importance
    importance = dict(zip(FEATURE_NAMES, final_model.feature_importance().tolist()))

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "num_samples": len(X_arr),
        "num_classes": num_classes,
        "n_splits": n_splits,
        "mean_accuracy": float(np.mean(fold_accuracies)) if fold_accuracies else 0.0,
        "std_accuracy": float(np.std(fold_accuracies)) if fold_accuracies else 0.0,
        "mean_log_loss": float(np.mean(fold_log_losses)) if fold_log_losses else 0.0,
        "fold_accuracies": fold_accuracies,
        "feature_importance": importance,
        "hyperparameters": {
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_samples": min_child_samples,
        },
    }

    return final_model, report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train ML regime classifier")
    parser.add_argument(
        "--features",
        default="data/features.jsonl",
        help="Path to feature JSONL file",
    )
    parser.add_argument(
        "--output",
        default="data/models/regime_classifier.lgb",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--report",
        default="data/models/training_report.json",
        help="Path to save training report",
    )
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon in bars")
    parser.add_argument("--n-splits", type=int, default=5, help="Walk-forward CV splits")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    args = parser.parse_args()

    logger.info("Loading features from %s", args.features)
    rows = load_feature_rows(args.features)
    logger.info("Loaded %d feature snapshots", len(rows))

    if len(rows) < 50:
        logger.error("Need at least 50 feature snapshots for training, got %d", len(rows))
        return 1

    logger.info("Preparing dataset with horizon=%d", args.horizon)
    X, y = prepare_dataset(rows, horizon=args.horizon)
    logger.info("Dataset: %d samples, %d features", len(X), len(X[0]) if X else 0)

    if len(X) < 20:
        logger.error("Not enough valid samples for training: %d", len(X))
        return 1

    logger.info("Training model with %d splits, %d estimators", args.n_splits, args.n_estimators)
    model, report = train_model(
        X, y,
        n_splits=args.n_splits,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
    )

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    logger.info("Model saved to %s", output_path)

    # Save report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Training report saved to %s", report_path)

    logger.info(
        "Training complete: accuracy=%.4f ± %.4f log_loss=%.4f",
        report["mean_accuracy"],
        report["std_accuracy"],
        report["mean_log_loss"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
