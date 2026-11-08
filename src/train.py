"""
Model Training, Purged Group Cross-Validation, & Comparative Benchmark Suite.

Implements:
1. Purged Group Time-Series Splitter (grouped by calendar day with multi-day embargo buffer).
2. Bayesian Hyperparameter Optimization via Optuna.
3. Gradient Boosted Trees (LightGBM, XGBoost) and Random Forest.
4. Classical Baselines (Ridge, MLP) and Physical Baselines (Kimber, Piedra-Gueymard).
5. Comprehensive benchmark metrics export for publication Table 2 (R2, RMSE, MAE, Pearson r).
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ModelTrainer")


@dataclass
class ValidationMetrics:
    model_name: str
    target: str
    rmse: float
    mae: float
    r2: float
    pearson_r: float
    spearman_rho: float
    train_samples: int
    val_samples: int


class PurgedGroupTimeSeriesSplit:
    """
    Purged Group Time-Series Cross-Validator.
    Splits data by distinct calendar dates and inserts a multi-day purge embargo
    between train and test periods to eliminate diurnal and synoptic autocorrelation leakage.
    """

    def __init__(self, n_splits: int = 5, purge_embargo_days: int = 2, test_ratio: float = 0.20):
        self.n_splits = n_splits
        self.purge_embargo_days = purge_embargo_days
        self.test_ratio = test_ratio

    def split(self, df: pd.DataFrame, date_col: str = "date") -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yields (train_indices, val_indices) respecting temporal day groups with embargo."""
        dates = pd.Series(df[date_col].values).drop_duplicates().sort_values().values
        n_dates = len(dates)

        if n_dates < 20:
            raise ValueError(f"Insufficient distinct calendar dates for PurgedGroupTimeSeriesSplit: {n_dates}")

        val_days = max(5, int(n_dates * self.test_ratio / self.n_splits))
        min_train_days = int(n_dates * 0.40)
        step_days = max(1, (n_dates - min_train_days - val_days) // self.n_splits)

        for i in range(self.n_splits):
            train_end_idx = min_train_days + i * step_days
            val_start_idx = train_end_idx + self.purge_embargo_days
            val_end_idx = min(val_start_idx + val_days, n_dates)

            if val_start_idx >= n_dates or val_end_idx <= val_start_idx:
                break

            train_dates = set(dates[:train_end_idx])
            val_dates = set(dates[val_start_idx:val_end_idx])

            train_indices = np.where(df[date_col].isin(train_dates))[0]
            val_indices = np.where(df[date_col].isin(val_dates))[0]

            if len(train_indices) > 0 and len(val_indices) > 0:
                yield train_indices, val_indices


class PhysicalKimberBaseline:
    """
    Classical empirical Kimber Soiling Model (Kimber et al., IEEE PVSC 2006).
    Predicts constant linear daily soiling rate proportional to PM10, reset by rain.
    """

    def __init__(self):
        self.k_dep = 0.00008
        self.k_clean = 0.85

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PhysicalKimberBaseline:
        if "pm10" in X.columns or "pm10_mean" in X.columns:
            col = "pm10_mean" if "pm10_mean" in X.columns else "pm10"
            pm = X[col].values
            valid = pm > 0
            if np.sum(valid) > 10:
                self.k_dep = max(1e-6, float(-np.median(y.values[valid] / pm[valid])))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        delta = np.zeros(len(X), dtype=float)
        col = "pm10_mean" if "pm10_mean" in X.columns else "pm10"
        if col in X.columns:
            delta -= self.k_dep * X[col].values
        if "dry_wind_clean_potential" in X.columns:
            delta += 0.0003 * np.maximum(0, X["dry_wind_clean_potential"].values - 6.0)
        if "rainfall_sum" in X.columns:
            rain_wash = np.where(X["rainfall_sum"].values > 1.0, 0.05, 0.0)
            delta += rain_wash

        if "soiling_ratio_prior" in X.columns:
            return np.clip(X["soiling_ratio_prior"].values + delta, 0.5, 1.0)
        return delta


class PhysicalPiedraGueymardBaseline:
    """
    Piedra-Gueymard Aerosol Deposition & Resuspension Physics Proxy (Solar Energy 2017).
    Models dry deposition flux and aerodynamic shear velocity resuspension.
    """

    def __init__(self):
        self.v_dep_const = 0.00005
        self.v_crit = 6.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PhysicalPiedraGueymardBaseline:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pm_col = "pm10_mean" if "pm10_mean" in X.columns else "pm10"
        ws_col = "wind_speed_peak" if "wind_speed_peak" in X.columns else "wind_speed"
        rh_col = "relative_humidity_mean" if "relative_humidity_mean" in X.columns else "relative_humidity"

        pm = X[pm_col].values if pm_col in X.columns else np.full(len(X), 50.0)
        ws = X[ws_col].values if ws_col in X.columns else np.full(len(X), 3.0)
        rh = X[rh_col].values if rh_col in X.columns else np.full(len(X), 50.0)

        deposition = -self.v_dep_const * (pm / 50.0)
        excess_v = np.maximum(0.0, ws - self.v_crit)
        resuspension = np.where((ws > self.v_crit) & (rh < 65.0), 0.0007 * (excess_v**1.5), 0.0)
        delta = deposition + resuspension

        if "soiling_ratio_prior" in X.columns:
            return np.clip(X["soiling_ratio_prior"].values + delta, 0.5, 1.0)
        return delta


class ModelTrainer:
    """
    Coordinates model training, purged walk-forward cross-validation,
    Bayesian hyperparameter optimization, and scientific comparative benchmarking.
    """

    def __init__(self, output_dir: Union[str, Path] = "models", random_state: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, ValidationMetrics] = {}

    def build_regressor(self, model_type: str = "lightgbm", params: Optional[Dict[str, Any]] = None) -> Any:
        """Instantiates configured regressor."""
        model_type = model_type.lower()
        if model_type in ["lightgbm", "lgbm"]:
            if lgb is None:
                raise ImportError("lightgbm is not installed.")
            default_params = {
                "n_estimators": 300,
                "learning_rate": 0.035,
                "max_depth": 6,
                "num_leaves": 31,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "min_child_samples": 20,
                "random_state": self.random_state,
                "n_jobs": -1,
                "verbose": -1,
            }
            if params:
                default_params.update(params)
            return lgb.LGBMRegressor(**default_params)

        elif model_type in ["xgboost", "xgb"]:
            if xgb is None:
                raise ImportError("xgboost is not installed.")
            default_params = {
                "n_estimators": 300,
                "learning_rate": 0.035,
                "max_depth": 5,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "random_state": self.random_state,
                "n_jobs": -1,
                "tree_method": "hist",
            }
            if params:
                default_params.update(params)
            return xgb.XGBRegressor(**default_params)

        elif model_type in ["random_forest", "rf"]:
            return RandomForestRegressor(
                n_estimators=200, max_depth=8, random_state=self.random_state, n_jobs=-1
            )

        elif model_type in ["ridge", "linear"]:
            return Ridge(alpha=10.0, random_state=self.random_state)

        elif model_type in ["mlp", "neural_network"]:
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            max_iter=400,
                            random_state=self.random_state,
                            early_stopping=True,
                        ),
                    ),
                ]
            )

        elif model_type in ["kimber", "kimber_physical"]:
            return PhysicalKimberBaseline()

        elif model_type in ["piedra_gueymard", "pg_physical"]:
            return PhysicalPiedraGueymardBaseline()

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "delta_soiling_ratio",
        filter_daylight_only: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Prepares X and y matrices while strictly preserving the underlying
        date column and temporal alignment for purged cross-validation.
        """
        working_df = df.copy()

        if "date" not in working_df.columns and "datetime" in working_df.columns:
            working_df["date"] = pd.to_datetime(working_df["datetime"]).dt.date

        if filter_daylight_only and "is_daylight_observation" in working_df.columns:
            working_df = working_df[working_df["is_daylight_observation"]].copy()

        available_features = [col for col in feature_cols if col in working_df.columns]
        valid_mask = working_df[available_features + [target_col]].notna().all(axis=1)
        clean_df = working_df[valid_mask].reset_index(drop=True)

        X = clean_df[available_features]
        y = clean_df[target_col]

        logger.info(
            "Prepared dataset for target '%s': %d clean samples, %d features, %d unique calendar days.",
            target_col,
            len(clean_df),
            len(available_features),
            clean_df["date"].nunique() if "date" in clean_df.columns else 0,
        )
        return X, y, clean_df

    def optimize_hyperparameters_optuna(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        clean_df: pd.DataFrame,
        model_type: str = "lightgbm",
        n_trials: int = 20,
    ) -> Dict[str, Any]:
        """Uses Optuna to find optimal hyperparameters minimizing Purged CV RMSE."""
        if optuna is None or model_type not in ["lightgbm", "xgboost"]:
            return {}

        splitter = PurgedGroupTimeSeriesSplit(n_splits=4, purge_embargo_days=2)

        def objective(trial: optuna.Trial) -> float:
            if model_type == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 150, 400),
                    "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
                    "max_depth": trial.suggest_int("max_depth", 4, 8),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "subsample": trial.suggest_float("subsample", 0.70, 0.95),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 0.95),
                }
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 150, 400),
                    "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 7),
                    "subsample": trial.suggest_float("subsample", 0.70, 0.95),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 0.95),
                }

            rmses = []
            for train_idx, val_idx in splitter.split(clean_df):
                model = self.build_regressor(model_type, params=params)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X.iloc[val_idx])
                rmses.append(float(np.sqrt(mean_squared_error(y.iloc[val_idx], preds))))

            return float(np.mean(rmses)) if rmses else float("inf")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info("Optuna best trial for %s: RMSE=%.6f, params=%s", model_type, study.best_value, study.best_params)
        return study.best_params

    def train_and_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        clean_df: pd.DataFrame,
        target_name: str = "delta_soiling_ratio",
        model_type: str = "lightgbm",
        n_splits: int = 5,
        purge_embargo_days: int = 2,
        use_optuna: bool = False,
    ) -> Tuple[Any, ValidationMetrics]:
        """
        Performs Purged Group Time-Series CV on clean_df and fits final production model.
        """
        splitter = PurgedGroupTimeSeriesSplit(n_splits=n_splits, purge_embargo_days=purge_embargo_days)

        best_params = {}
        if use_optuna and model_type in ["lightgbm", "xgboost"]:
            best_params = self.optimize_hyperparameters_optuna(X, y, clean_df, model_type=model_type, n_trials=15)

        fold_rmses, fold_maes, fold_r2s = [], [], []
        fold_pearsons, fold_spearmans = [], []

        for fold, (train_idx, val_idx) in enumerate(splitter.split(clean_df), start=1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            fold_model = self.build_regressor(model_type, params=best_params)
            fold_model.fit(X_train, y_train)
            val_preds = fold_model.predict(X_val)

            fold_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
            fold_mae = float(mean_absolute_error(y_val, val_preds))
            fold_r2 = float(r2_score(y_val, val_preds))

            # Statistical correlations
            r_val, _ = pearsonr(y_val, val_preds) if len(y_val) > 2 and np.std(val_preds) > 1e-9 else (0.0, 1.0)
            rho_val, _ = spearmanr(y_val, val_preds) if len(y_val) > 2 and np.std(val_preds) > 1e-9 else (0.0, 1.0)

            fold_rmses.append(fold_rmse)
            fold_maes.append(fold_mae)
            fold_r2s.append(fold_r2)
            fold_pearsons.append(float(r_val))
            fold_spearmans.append(float(rho_val))

        mean_rmse = float(np.mean(fold_rmses)) if fold_rmses else 0.0
        mean_mae = float(np.mean(fold_maes)) if fold_maes else 0.0
        mean_r2 = float(np.mean(fold_r2s)) if fold_r2s else 0.0
        mean_pearson = float(np.mean(fold_pearsons)) if fold_pearsons else 0.0
        mean_spearman = float(np.mean(fold_spearmans)) if fold_spearmans else 0.0

        logger.info(
            "Purged CV Results for %s: R2=%.4f, RMSE=%.6f, MAE=%.6f, Pearson r=%.4f",
            model_type.upper(),
            mean_r2,
            mean_rmse,
            mean_mae,
            mean_pearson,
        )

        final_model = self.build_regressor(model_type, params=best_params)
        final_model.fit(X, y)

        key = f"{model_type}_{target_name}"
        self.models[key] = final_model

        metrics = ValidationMetrics(
            model_name=model_type,
            target=target_name,
            rmse=mean_rmse,
            mae=mean_mae,
            r2=mean_r2,
            pearson_r=mean_pearson,
            spearman_rho=mean_spearman,
            train_samples=len(X),
            val_samples=int(np.mean([len(val_idx) for _, val_idx in splitter.split(clean_df)])) if fold_rmses else 0,
        )
        self.metrics[key] = metrics

        model_path = self.output_dir / f"{key}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": final_model, "feature_names": list(X.columns), "metrics": asdict(metrics)}, f)

        return final_model, metrics

    def run_full_comparative_benchmark_suite(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        clean_df: pd.DataFrame,
        target_name: str = "delta_soiling_ratio",
    ) -> Tuple[str, Any, pd.DataFrame]:
        """
        Executes complete peer-review benchmark suite across all candidate models:
        1. Proposed: LightGBM (with Mechanistic Features)
        2. Proposed: XGBoost (with Mechanistic Features)
        3. Baseline: Random Forest
        4. Baseline: Ridge Regression
        5. Baseline: Multilayer Perceptron (MLP)
        6. Baseline: Physical Kimber Model
        7. Baseline: Physical Piedra-Gueymard Model

        Generates structured DataFrame corresponding to Table 2 of the manuscript.
        """
        candidate_models = [
            "lightgbm",
            "xgboost",
            "random_forest",
            "ridge",
            "mlp",
            "kimber",
            "piedra_gueymard",
        ]

        benchmark_records = []
        trained_models: Dict[str, Tuple[Any, ValidationMetrics]] = {}

        for m_type in candidate_models:
            try:
                model, met = self.train_and_validate(
                    X=X, y=y, clean_df=clean_df, target_name=target_name, model_type=m_type, n_splits=5
                )
                trained_models[m_type] = (model, met)
                benchmark_records.append(asdict(met))
            except Exception as e:
                logger.error("Error benchmarking %s: %s", m_type, e)

        benchmark_df = pd.DataFrame(benchmark_records).sort_values("r2", ascending=False).reset_index(drop=True)

        # Champion model
        champion_type = str(benchmark_df.iloc[0]["model_name"])
        champion_model = trained_models[champion_type][0]

        logger.info("Benchmark complete. Champion: %s (R2=%.4f)", champion_type.upper(), benchmark_df.iloc[0]["r2"])

        # Persist benchmark table
        bench_json_path = self.output_dir / "comparative_benchmark_table2.json"
        with open(bench_json_path, "w") as f:
            json.dump(benchmark_records, f, indent=2)

        return champion_type, champion_model, benchmark_df

    def compare_and_select_best(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        clean_df: pd.DataFrame,
        target_name: str = "delta_soiling_ratio",
    ) -> Tuple[str, Any, ValidationMetrics]:
        """
        Retrieves the pre-trained champion model or executes the comparative benchmark suite,
        returning (champion_type, champion_model, champion_metrics).
        """
        # Attempt to load pre-trained champion if available in output_dir
        for candidate in ["random_forest", "xgboost", "lightgbm"]:
            key = f"{candidate}_{target_name}"
            model_path = self.output_dir / f"{key}_model.pkl"
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        data = pickle.load(f)
                    met_dict = data.get("metrics", {})
                    metrics = ValidationMetrics(**met_dict)
                    self.models[key] = data["model"]
                    self.metrics[key] = metrics
                    logger.info("Loaded pre-trained champion model: %s (%s)", candidate.upper(), key)
                    return candidate, data["model"], metrics
                except Exception as e:
                    logger.warning("Could not deserialize checkpoint for %s: %s", key, e)

        # Fallback to full benchmark suite
        champion_type, champion_model, benchmark_df = self.run_full_comparative_benchmark_suite(
            X, y, clean_df, target_name=target_name
        )
        champion_metrics = self.metrics[f"{champion_type}_{target_name}"]
        return champion_type, champion_model, champion_metrics


if __name__ == "__main__":
    from src.data_loader import PVDataLoader
    from src.feature_engineering import FeatureEngineer

    loader = PVDataLoader()
    raw = loader.load_data()
    fe = FeatureEngineer()
    feat = fe.extract_features(raw)
    daily_df = fe.extract_daily_dataset(feat)

    trainer = ModelTrainer()
    feature_cols = fe.get_daily_feature_names()
    X = daily_df[feature_cols]
    y = daily_df["delta_soiling_ratio_daily"]
    clean_df = daily_df.copy()

    champ_name, champ_model, bench_table = trainer.run_full_comparative_benchmark_suite(
        X, y, clean_df, target_name="delta_soiling_ratio_daily"
    )
    print("\n=== Daily Soiling Rate Model Benchmark Suite (Table 2) ===")
    print(bench_table[["model_name", "r2", "rmse", "mae", "pearson_r"]])
