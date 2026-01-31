"""
Model training pipeline.
Multiple models, time-based split, hyperparameter tuning, persistence, evaluation, baseline comparison.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

from config import MODELS_DIR, TEST_SIZE, RANDOM_STATE, CV_FOLDS, MODEL_NAMES
from src.data_loader import load_and_clean
from src.features import build_feature_matrix


def time_based_split(
    X: pd.DataFrame, y: pd.Series, dates: pd.Series, test_size: float = TEST_SIZE
):
    """Train-test split by time: train on earlier matches, test on later."""
    n = len(X)
    split_idx = int(n * (1 - test_size))
    train_idx = X.index[:split_idx]
    test_idx = X.index[split_idx:]
    return (
        X.loc[train_idx],
        X.loc[test_idx],
        y.loc[train_idx],
        y.loc[test_idx],
    )


def get_models():
    """Return dict of model name -> (sklearn-style model, param_grid for tuning)."""
    return {
        "logistic_regression": (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "saga"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_leaf": [1, 2, 5],
            },
        ),
        "xgboost": (
            xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="logloss",
                n_estimators=100,
            ),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "min_child_weight": [1, 3, 5],
            },
        ),
    }


def optimize_hyperparameters(model_name: str, X_train, y_train):
    """Hyperparameter optimization using time-series cross-validation."""
    from sklearn.model_selection import RandomizedSearchCV

    models = get_models()
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    base_model, param_grid = models[model_name]
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=min(20, np.prod([len(v) for v in param_grid.values()])),
        cv=tscv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calibration: Brier score and ECE (expected calibration error)."""
    brier = brier_score_loss(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ece = np.abs(prob_true - prob_pred).mean()
    return {"brier_score": float(brier), "ece": float(ece)}


def evaluate_model(y_true, y_pred, y_prob, name: str = "") -> dict:
    """Accuracy, ROC-AUC, Log Loss, calibration."""
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "log_loss": float(log_loss(y_true, y_prob)),
        **calibration_metrics(y_true, y_prob),
    }


def baseline_heuristics(X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Baseline: higher-ranked player wins, higher Elo wins.
    Assumes X_test has rank_diff and elo_diff (negative => player2 favored).
    """
    results = {}
    if "rank_diff" in X_test.columns:
        # rank_diff = player1_rank - player2_rank. Lower rank number = better. So negative rank_diff => player1 better => predict 1
        pred_rank = (X_test["rank_diff"] < 0).astype(int)
        prob_rank = np.where(X_test["rank_diff"] < 0, 0.6, 0.4)  # dummy prob
        results["higher_rank_wins"] = evaluate_model(
            y_test.values, pred_rank, prob_rank, "higher_rank_wins"
        )
    if "elo_diff" in X_test.columns:
        pred_elo = (X_test["elo_diff"] > 0).astype(int)
        prob_elo = 1.0 / (1.0 + np.exp(-X_test["elo_diff"] / 200.0))
        results["higher_elo_wins"] = evaluate_model(
            y_test.values, pred_elo, prob_elo, "higher_elo_wins"
        )
    return results


def train_and_evaluate(
    csv_path: Path = None,
    dataframe: pd.DataFrame = None,
    optimize: bool = True,
    save_dir: Path = None,
):
    """
    Load data, build features, time split, train models, evaluate, compare baselines, persist.
    """
    if save_dir is None:
        save_dir = MODELS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(csv_path=csv_path, dataframe=dataframe)
    X, y, feature_names = build_feature_matrix(df)
    # X is ordered by match_date (from sorted df); split by position = time-based
    X_train, X_test, y_train, y_test = time_based_split(X, y, X.index.to_series(), test_size=TEST_SIZE)

    all_metrics = []
    best_model = None
    best_auc = -1
    best_name = None

    for model_name in MODEL_NAMES:
        if optimize:
            model = optimize_hyperparameters(model_name, X_train, y_train)
        else:
            base, _ = get_models()[model_name]
            model = base.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = evaluate_model(y_test.values, y_pred, y_prob, model_name)
        all_metrics.append(metrics)
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model = model
            best_name = model_name

    # Baselines
    baseline_results = baseline_heuristics(X_test, y_test)
    for b in baseline_results.values():
        all_metrics.append(b)

    # Persist best model and metadata
    all_players = sorted(set(df["player1_name"].dropna().astype(str).str.strip()) | set(df["player2_name"].dropna().astype(str).str.strip()))
    meta = {
        "feature_names": feature_names,
        "model_name": best_name,
        "metrics": {m["model"]: m for m in all_metrics},
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "players": all_players,
    }
    def _json_serial(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    joblib.dump(best_model, save_dir / "model.joblib")
    joblib.dump(feature_names, save_dir / "feature_names.joblib")
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=_json_serial)

    # Also save each model by name for GET /model-info
    for model_name in MODEL_NAMES:
        base, _ = get_models()[model_name]
        if optimize:
            m = optimize_hyperparameters(model_name, X_train, y_train)
        else:
            m = base.fit(X_train, y_train)
        joblib.dump(m, save_dir / f"model_{model_name}.joblib")

    return all_metrics, best_model, feature_names, meta
