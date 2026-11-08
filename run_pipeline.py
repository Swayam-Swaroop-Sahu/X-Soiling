"""
End-to-End Orchestration Pipeline for PV Soiling Tipping Point Discovery & XAI.

Executes:
1. Ingestion of raw PV and weather time series (Mendeley 4jdt83yy43 empirical cohort).
2. Mechanistic feature engineering (IEC 61724-1 soiling ratio, nocturnal memory, aerosol thermodynamics).
3. Purged walk-forward cross-validation and benchmark suite across tree ensembles and physical baselines.
4. SHAP TreeExplainer attributions and 2D interaction matrices (PNG + PDF formats).
5. Programmatic threshold mining with 95% bootstrapped CIs and Chow test structural break verification.
6. Operational 48h forecast risk evaluation and techno-economic dispatch simulator.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import PVDataLoader
from src.feature_engineering import FeatureEngineer
from src.operational_engine import OperationalEngine
from src.techno_economic import TechnoEconomicSimulator
from src.threshold_miner import ThresholdMiner
from src.train import ModelTrainer
from src.xai_engine import XAIEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PVSoilingPipeline")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PV Soiling Explainable AI: Physical Tipping Point & Operational Cleaning Pipeline"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to raw CSV or Parquet file (defaults to data/raw/ directory scan).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="soiling_ratio_daily",
        choices=["soiling_ratio_daily", "delta_soiling_ratio_daily", "delta_soiling_ratio"],
        help="Target variable for explanatory modeling (default: soiling_ratio_daily).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "lightgbm", "xgboost", "random_forest", "ridge"],
        help="Model architecture to train (auto runs full comparative benchmark suite).",
    )
    parser.add_argument(
        "--shap-samples",
        type=int,
        default=400,
        help="Maximum sample count for computing SHAP attributions and 2D interactions.",
    )
    parser.add_argument(
        "--forecast-hours",
        type=int,
        default=48,
        help="Duration of operational forecast horizon to evaluate (hours).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save JSON reports and metric files.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="figures",
        help="Directory to save generated SHAP diagnostic plots.",
    )
    return parser.parse_args()


def main():
    start_time = time.time()
    args = parse_arguments()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print(" [*] PV SOILING EXPLAINABLE AI PIPELINE (PHYSICAL TIPPING POINTS) [*]")
    print("=" * 80)

    # 1. Ingestion
    logger.info("Step 1/6: Ingesting empirical multi-site dataset...")
    loader = PVDataLoader(raw_dir="data/raw", processed_dir="data/processed")
    raw_df = loader.load_data(file_path=args.data_path, auto_generate_if_empty=True)
    logger.info("Dataset loaded successfully: %d records, %d columns.", len(raw_df), len(raw_df.columns))

    # 2. Mechanistic Feature Engineering & Daily Dataset Aggregation
    logger.info("Step 2/6: Extracting mechanistic thermodynamic and diurnal memory features...")
    fe = FeatureEngineer()
    featured_df = fe.extract_features(raw_df, compute_targets=True)
    daily_df = fe.extract_daily_dataset(featured_df)

    processed_path = Path("data/processed/pv_soiling_features_processed.parquet")
    featured_df.to_parquet(processed_path, index=False)
    logger.info("Saved processed features to: %s", processed_path)

    # 3. Model Training & Purged Group Time-Series Cross-Validation
    logger.info("Step 3/6: Model training with Purged Group Walk-Forward Cross-Validation...")
    trainer = ModelTrainer(output_dir="models")
    feature_cols = fe.get_daily_feature_names()

    X = daily_df[feature_cols]
    y = daily_df[args.target]
    clean_df = daily_df.copy()

    if args.model_type == "auto":
        champion_name, champion_model, bench_table = trainer.run_full_comparative_benchmark_suite(
            X, y, clean_df, target_name=args.target
        )
        champion_metrics = trainer.metrics[f"{champion_name}_{args.target}"]
    else:
        champion_name = args.model_type
        champion_model, champion_metrics = trainer.train_and_validate(
            X, y, clean_df, target_name=args.target, model_type=champion_name
        )

    logger.info(
        "Model selected: %s | Holdout RMSE: %.6f | MAE: %.6f | R²: %.4f | Pearson r: %.4f",
        champion_name.upper(),
        champion_metrics.rmse,
        champion_metrics.mae,
        champion_metrics.r2,
        champion_metrics.pearson_r,
    )

    # 4. Explainable AI & Dual-Format Figure Generation
    logger.info("Step 4/6: Computing SHAP TreeExplainer attributions and 2D interaction tensor...")
    xai = XAIEngine(
        model=champion_model,
        feature_names=feature_cols,
        output_dirs=[Path(args.figures_dir), Path("paper/figures")],
    )
    xai.compute_shap_values(X, max_samples=args.shap_samples)
    beeswarm_paths = xai.plot_summary(save_name="fig1_global_shap_beeswarm.png")

    xai.compute_interaction_values(X, max_samples=min(args.shap_samples, 300))
    cementation_fig_paths = xai.plot_2d_cementation_cliff(
        feat_x="relative_humidity_mean",
        feat_y="pm10_mean",
        save_name="fig3_cementation_cliff_2d_interaction.png",
    )
    xai.plot_2d_cementation_cliff(
        feat_x="dew_deficit_mean",
        feat_y="pm2_5_mean",
        save_name="fig4_dew_deficit_pm25_interaction.png",
    )

    # 5. Programmatic Physical Threshold Mining & Bootstrapping
    logger.info("Step 5/6: Mining physical tipping points (v_crit and cementation boundaries)...")
    miner = ThresholdMiner(smoothing_window=15, n_bootstraps=400)

    wind_vals, wind_shap = xai.get_1d_dependence_curve("wind_speed_peak")
    dew_vals, _ = xai.get_1d_dependence_curve("dew_deficit_mean")
    rh_vals, rh_shap = xai.get_1d_dependence_curve("relative_humidity_mean")
    dew_inter_v, pm10_inter_v, inter_dew_pm = xai.extract_2d_interaction("dew_deficit_mean", "pm10_mean")

    tipping_points = miner.mine_all_thresholds(
        wind_speeds=wind_vals,
        wind_shap=wind_shap,
        dew_deficits=dew_inter_v,
        pm10_vals=pm10_inter_v,
        interaction_dew_pm=inter_dew_pm,
        rh_vals=rh_vals,
        rh_shap=rh_shap,
    )

    wind_plot_paths = xai.plot_wind_dependence(
        v_crit=tipping_points.wind_resuspension_threshold_mps,
        ci_lower=tipping_points.wind_v_crit_ci_lower_mps,
        ci_upper=tipping_points.wind_v_crit_ci_upper_mps,
        save_name="fig2_wind_resuspension_threshold.png",
    )

    tp_report_path = miner.export_report(
        tipping_points, output_path=Path(args.output_dir) / "tipping_points.json"
    )

    # 6. Operational Cleaning Engine & Techno-Economic Dispatch
    logger.info("Step 6/6: Running operational dispatch engine & 8,760h techno-economic simulation...")
    operational_engine = OperationalEngine(
        tipping_points=tipping_points,
        fe=fe,
        output_dir=args.output_dir,
    )
    forecast_df = operational_engine.generate_sample_forecast(hours=args.forecast_hours)
    decisions, op_summary = operational_engine.evaluate_forecast(forecast_df)
    op_report_path = operational_engine.export_operational_report(decisions, op_summary)

    sim = TechnoEconomicSimulator(plant_capacity_mwp=50.0, tipping_points=tipping_points)
    econ_summary, _ = sim.simulate_8760h(featured_df)
    sim.export_latex_table(econ_summary, Path("paper/tables/table4_techno_economic_roi.tex"))

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(" [OK] PIPELINE EXECUTION COMPLETE (in {:.1f}s)".format(elapsed))
    print("=" * 80)
    print("[*] DISCOVERED PHYSICAL TIPPING POINTS:")
    print(f"   - Critical Wind Resuspension (v_crit): {tipping_points.wind_resuspension_threshold_mps:.2f} m/s")
    print(f"     95% Bootstrapped CI: [{tipping_points.wind_v_crit_ci_lower_mps:.2f}, {tipping_points.wind_v_crit_ci_upper_mps:.2f}] m/s | Chow p={tipping_points.wind_structural_break_p_value:.4e}")
    print(f"   - Cementation Dew Deficit Threshold:  {tipping_points.cementation_dew_deficit_celsius:.2f} deg C")
    print(f"     95% Bootstrapped CI: [{tipping_points.dew_deficit_ci_lower_celsius:.2f}, {tipping_points.dew_deficit_ci_upper_celsius:.2f}] deg C")
    print(f"   - Critical Relative Humidity:         {tipping_points.critical_relative_humidity_pct:.1f} %")
    print("\n[*] HOLD-OUT MODEL PERFORMANCE:")
    print(f"   - Architecture: {champion_name.upper()}")
    print(f"   - Validation R2: {champion_metrics.r2:.4f}")
    print(f"   - Validation RMSE: {champion_metrics.rmse:.6f}")
    print(f"   - Pearson Correlation r: {champion_metrics.pearson_r:.4f}")
    print("\n[*] GENERATED ARTIFACTS & REPORTS:")
    print(f"   - SHAP Summary Beeswarm:      {beeswarm_paths[0]}")
    print(f"   - Wind Dependence Curve:      {wind_plot_paths[0]}")
    print(f"   - Cementation 2D Heatmap:     {cementation_fig_paths[0]}")
    print(f"   - Physical Boundaries JSON:   {tp_report_path}")
    print(f"   - Operational Dispatch JSON:  {op_report_path}")
    print(f"   - LaTeX Manuscript Paper:     paper/main.pdf")
    print("=" * 80)


if __name__ == "__main__":
    main()
