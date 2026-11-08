"""
Unit Tests for Mechanistic Physical Calculations and Feature Engineering.

Validates:
1. Magnus-Tetens formula against known meteorological benchmarks.
2. Dew Deficit & Nocturnal Condensation Logic.
3. Soiling Ratio (SR) and Delta_SR derivations.
4. Cumulative PM exposures and hours since rain counter.
5. Threshold Miner zero-crossing and change-point algorithms.
6. Operational Dispatch logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineer
from src.operational_engine import OperationalEngine
from src.threshold_miner import DiscoveredTippingPoints, ThresholdMiner


class TestFeatureEngineering:

    @pytest.fixture
    def feature_engineer(self):
        return FeatureEngineer()

    def test_magnus_dew_point_standard_benchmark(self, feature_engineer):
        """
        Validates Magnus formula against reference thermodynamic value:
        At T = 25.0 °C and RH = 50.0%, theoretical T_dew is ~13.84 - 13.90 °C.
        """
        temp = np.array([25.0])
        rh = np.array([50.0])

        t_dew, deficit = feature_engineer.compute_magnus_dew_point(temp, rh)

        assert len(t_dew) == 1
        assert 13.7 <= t_dew[0] <= 14.1, f"Expected T_dew ~13.85°C, got {t_dew[0]}"
        assert 10.9 <= deficit[0] <= 11.3, f"Expected Dew Deficit ~11.15°C, got {deficit[0]}"

    def test_magnus_dew_point_saturation_100_percent(self, feature_engineer):
        """
        At 100% RH (saturation), Dew-Point must equal Ambient Temperature,
        and Dew Deficit must equal 0.0 °C.
        """
        temp = np.array([18.5, 30.0, 5.0])
        rh = np.array([100.0, 100.0, 100.0])

        t_dew, deficit = feature_engineer.compute_magnus_dew_point(temp, rh)

        np.testing.assert_allclose(t_dew, temp, atol=1e-3)
        np.testing.assert_allclose(deficit, np.zeros(3), atol=1e-3)

    def test_soiling_ratio_derivation(self, feature_engineer):
        """
        Tests calculation of SR = P_soiled / P_clean, low-irradiance masking,
        and step-change Delta_SR = SR_t - SR_{t-1}.
        """
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2023-01-01 08:00:00", periods=5, freq="1h"),
                "irradiance": [200.0, 500.0, 800.0, 20.0, 0.0],  # last two are below threshold
                "power_clean": [10.0, 25.0, 40.0, 1.0, 0.0],
                "power_soiled": [9.0, 22.5, 34.0, 0.8, 0.0],  # ratios: 0.90, 0.90, 0.85
                "temp_cell": [30.0, 45.0, 55.0, 25.0, 20.0],
            }
        )

        result_df = feature_engineer.compute_soiling_ratio(df)

        assert "soiling_ratio" in result_df.columns
        assert "delta_soiling_ratio" in result_df.columns
        assert "is_daylight_observation" in result_df.columns

        # Verify daylight mask
        assert result_df["is_daylight_observation"].iloc[0] == True
        assert result_df["is_daylight_observation"].iloc[3] == False
        assert result_df["is_daylight_observation"].iloc[4] == False

        # First daylight observation should be around 0.90
        assert 0.88 <= result_df["soiling_ratio"].iloc[0] <= 0.92

        # Delta SR for row 2 (t=2 vs t=1): SR dropped from 0.90 to 0.85 -> delta is negative
        assert result_df["delta_soiling_ratio"].iloc[2] < 0.0

    def test_hours_since_rain_counter(self, feature_engineer):
        """
        Verifies that hours_since_rain resets to 0 during rain > 1.0 mm,
        and increments by dt otherwise.
        """
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2023-01-01 00:00:00", periods=6, freq="1h"),
                "rainfall": [0.0, 0.0, 3.5, 0.0, 0.0, 0.0],  # rain at t=2
                "temp_ambient": [20.0] * 6,
                "relative_humidity": [50.0] * 6,
                "wind_speed": [3.0] * 6,
                "pm10": [50.0] * 6,
                "pm2_5": [20.0] * 6,
                "irradiance": [0.0, 0.0, 100.0, 200.0, 300.0, 400.0],
            }
        )

        df = feature_engineer.compute_cumulative_and_rolling_features(df)

        rain_hours = df["hours_since_rain"].values
        assert rain_hours[2] == 0.0  # Reset upon rain
        assert rain_hours[3] == 1.0  # 1 hour after rain
        assert rain_hours[4] == 2.0  # 2 hours after rain
        assert rain_hours[5] == 3.0  # 3 hours after rain


class TestThresholdMiner:

    def test_find_zero_crossing_exact(self):
        """
        Tests linear zero-crossing detection on a known linear curve y = x - 7.2.
        Zero crossing must be exactly 7.2.
        """
        miner = ThresholdMiner(smoothing_window=5)
        x = np.linspace(3, 12, 100)
        y = x - 7.2

        crossing = miner.find_zero_crossing(x, y, search_min=3.0, search_max=12.0)
        assert crossing is not None
        assert abs(crossing - 7.2) < 0.15, f"Expected 7.2, got {crossing}"

    def test_find_knee_inflection_piecewise(self):
        """
        Tests knee / inflection finding on a piecewise function with corner at x=6.5.
        """
        miner = ThresholdMiner()
        x = np.linspace(2, 12, 100)
        y = np.where(x < 6.5, 0.0, 0.5 * (x - 6.5))

        knee = miner.find_knee_inflection(x, y)
        assert knee is not None
        assert 5.8 <= knee <= 7.2, f"Expected knee around 6.5, got {knee}"


class TestOperationalEngine:

    def test_operational_cementation_alert(self):
        """
        Ensures that low dew deficit (< 1.5°C) and high PM10 trigger HIGH/CRITICAL
        cementation alert and PREVENTIVE_WASH recommendation.
        """
        tp = DiscoveredTippingPoints(
            wind_resuspension_threshold_mps=6.8,
            wind_v_crit_ci_lower_mps=6.2,
            wind_v_crit_ci_upper_mps=7.4,
            wind_structural_break_p_value=0.001,
            critical_relative_humidity_pct=80.0,
            rh_crit_ci_lower_pct=75.0,
            rh_crit_ci_upper_pct=85.0,
            cementation_dew_deficit_celsius=1.5,
            dew_deficit_ci_lower_celsius=1.1,
            dew_deficit_ci_upper_celsius=1.9,
            wind_zero_crossing_mps=6.8,
            wind_knee_point_mps=6.5,
            confidence_score=0.95,
            notes={},
        )
        engine = OperationalEngine(tp)

        # Forecast with acute nocturnal cementation condition
        fc_df = pd.DataFrame(
            {
                "datetime": ["2026-10-28 04:00:00"],
                "temp_ambient": [14.0],
                "relative_humidity": [95.0],  # near dew!
                "wind_speed": [1.5],
                "rainfall": [0.0],
                "pm10": [120.0],  # high PM
                "pm2_5": [50.0],
            }
        )

        decisions, summary = engine.evaluate_forecast(fc_df)
        assert len(decisions) == 1
        assert decisions[0].cementation_risk_level in ["HIGH", "CRITICAL"]
        assert decisions[0].recommended_action == "PREVENTIVE_WASH_BEFORE_DEW"

    def test_free_cleaning_dispatch(self):
        """
        Ensures that dry high wind (> v_crit, RH < 65%) flags free cleaning
        and recommends deferring washing.
        """
        tp = DiscoveredTippingPoints(
            wind_resuspension_threshold_mps=6.8,
            wind_v_crit_ci_lower_mps=6.2,
            wind_v_crit_ci_upper_mps=7.4,
            wind_structural_break_p_value=0.001,
            critical_relative_humidity_pct=80.0,
            rh_crit_ci_lower_pct=75.0,
            rh_crit_ci_upper_pct=85.0,
            cementation_dew_deficit_celsius=1.5,
            dew_deficit_ci_lower_celsius=1.1,
            dew_deficit_ci_upper_celsius=1.9,
            wind_zero_crossing_mps=6.8,
            wind_knee_point_mps=6.5,
            confidence_score=0.95,
            notes={},
        )
        engine = OperationalEngine(tp)

        fc_df = pd.DataFrame(
            {
                "datetime": ["2026-10-28 14:00:00"],
                "temp_ambient": [32.0],
                "relative_humidity": [35.0],  # dry
                "wind_speed": [8.5],  # exceeds 6.8 m/s!
                "rainfall": [0.0],
                "pm10": [40.0],
                "pm2_5": [15.0],
            }
        )

        decisions, summary = engine.evaluate_forecast(fc_df)
        assert len(decisions) == 1
        assert decisions[0].free_cleaning_predicted == True
        assert decisions[0].recommended_action == "DEFER_WASH_FOR_WIND_CLEANING"
