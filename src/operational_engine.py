"""
Operational Decision Engine for Solar PV Soiling Management.

Ingests 24-48h forward weather forecasts and evaluates:
1. Cementation Risk Index (identifies imminent dew + PM cementation events).
2. Free Cleaning Dispatch (identifies upcoming high-wind dry cleaning events).
3. Automated cleaning schedule optimization (preventive cleaning vs wind-delay).
4. Full operational status reporting and JSON synthesis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.feature_engineering import FeatureEngineer
from src.threshold_miner import DiscoveredTippingPoints

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OperationalEngine")


@dataclass
class CleaningDispatchDecision:
    timestamp: str
    cementation_risk_level: str  # LOW, MODERATE, HIGH, CRITICAL
    cementation_risk_score: float  # 0 to 100
    free_cleaning_predicted: bool
    free_cleaning_hours: int
    recommended_action: str
    action_rationale: str
    cost_water_saving_potential: str
    expected_delta_soiling_ratio: float


class OperationalEngine:
    """
    Translates discovered physical tipping points and weather forecasts
    into actionable solar plant operational decisions.
    """

    def __init__(
        self,
        tipping_points: DiscoveredTippingPoints,
        fe: Optional[FeatureEngineer] = None,
        output_dir: Union[str, Path] = "reports",
    ):
        self.tipping_points = tipping_points
        self.fe = fe or FeatureEngineer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_single_hour(
        self,
        row: Union[pd.Series, Dict[str, Any]],
    ) -> CleaningDispatchDecision:
        """
        Evaluates a single observation or scenario row and returns a CleaningDispatchDecision.
        """
        if isinstance(row, pd.Series):
            df = pd.DataFrame([row.to_dict()])
        elif isinstance(row, dict):
            df = pd.DataFrame([row])
        else:
            df = pd.DataFrame([row])
        decisions, _ = self.evaluate_forecast(df)
        return decisions[0]

    def evaluate_forecast(
        self,
        forecast_df: pd.DataFrame,
    ) -> Tuple[List[CleaningDispatchDecision], Dict[str, Any]]:
        """
        Processes 24-48h weather forecast time series and generates hourly dispatch assessments
        and an aggregated plant operation summary.
        """
        df = forecast_df.copy()

        # Compute thermodynamic and aerodynamic indicators if not already present
        if "dew_deficit" not in df.columns:
            t_dew, dew_def = self.fe.compute_magnus_dew_point(
                df["temp_ambient"].values, df["relative_humidity"].values
            )
            df["temp_dew"] = t_dew
            df["dew_deficit"] = dew_def

        v_crit = self.tipping_points.wind_resuspension_threshold_mps
        dew_def_crit = self.tipping_points.cementation_dew_deficit_celsius
        rh_crit = self.tipping_points.critical_relative_humidity_pct

        hourly_decisions: List[CleaningDispatchDecision] = []
        total_cementation_alerts = 0
        total_wind_cleaning_hours = 0
        cumulative_sr_impact = 0.0

        for i in range(len(df)):
            row = df.iloc[i]
            time_str = str(row.get("datetime", f"T+{i}h"))
            ws = float(row.get("wind_speed", 0.0))
            rh = float(row.get("relative_humidity", 50.0))
            dew_def = float(row.get("dew_deficit", 5.0))
            pm10 = float(row.get("pm10", 40.0))
            rainfall = float(row.get("rainfall", 0.0))

            # 1. Cementation Risk Calculation
            # Triggered when dew_deficit < critical threshold and PM10 is elevated (> 50 ug/m3)
            risk_score = 0.0
            if dew_def < dew_def_crit:
                # Severity increases as dew deficit approaches 0 and PM surges
                deficit_factor = max(0.0, (dew_def_crit - dew_def) / max(0.1, dew_def_crit))
                pm_factor = min(2.0, pm10 / 50.0)
                risk_score = min(100.0, (deficit_factor * 60.0 + pm_factor * 40.0))
            elif rh > rh_crit:
                risk_score = min(65.0, (rh - rh_crit) * 3.0)

            if risk_score >= 75.0:
                risk_level = "CRITICAL"
                total_cementation_alerts += 1
            elif risk_score >= 50.0:
                risk_level = "HIGH"
                total_cementation_alerts += 1
            elif risk_score >= 25.0:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"

            # 2. Free Cleaning Dispatch Flag
            # Sustained wind > v_crit under dry conditions (RH < 65% and no rain)
            is_free_cleaning = bool(ws >= v_crit and rh < 65.0 and rainfall < 0.2)
            if is_free_cleaning:
                total_wind_cleaning_hours += 1

            # Estimate step impact on Soiling Ratio
            delta_sr = 0.0
            if rainfall > 1.0:
                delta_sr += 0.08  # rain wash recovery
            elif is_free_cleaning:
                delta_sr += 0.008 * (ws - v_crit + 0.5)  # wind resuspension recovery
            elif risk_level in ["CRITICAL", "HIGH"]:
                delta_sr -= 0.015 * (pm10 / 100.0)  # cementation loss
            else:
                delta_sr -= 0.0005 * (pm10 / 50.0)  # baseline dry settling

            cumulative_sr_impact += delta_sr

            # Operational dispatch recommendation logic
            if risk_level in ["CRITICAL", "HIGH"]:
                action = "PREVENTIVE_WASH_BEFORE_DEW"
                rationale = (
                    f"Near-dew condition (Deficit={dew_def:.1f}°C < {dew_def_crit:.1f}°C) with elevated PM10 "
                    f"({pm10:.0f} µg/m³) will cause irreversible dust cementation. Wash immediately before dusk."
                )
                savings = "Avoids costly chemical desoiling and permanent glass transmittance loss."
            elif is_free_cleaning:
                action = "DEFER_WASH_FOR_WIND_CLEANING"
                rationale = (
                    f"Dry wind speed ({ws:.1f} m/s) exceeds resuspension boundary ({v_crit:.1f} m/s). "
                    "Aerodynamic lift will naturally dislodge surface dust."
                )
                savings = "Direct water savings (~2.5 L/panel) and cancels planned robotic cleaning cycle."
            elif rainfall > 2.0:
                action = "CANCEL_WASH_NATURAL_RAIN"
                rationale = f"Natural precipitation ({rainfall:.1f} mm) will wash solar array."
                savings = "100% water and operational labor savings."
            else:
                action = "STANDARD_OPERATION"
                rationale = "Atmospheric conditions nominal; no acute tipping points detected."
                savings = "N/A"

            decision = CleaningDispatchDecision(
                timestamp=time_str,
                cementation_risk_level=risk_level,
                cementation_risk_score=round(risk_score, 1),
                free_cleaning_predicted=is_free_cleaning,
                free_cleaning_hours=1 if is_free_cleaning else 0,
                recommended_action=action,
                action_rationale=rationale,
                cost_water_saving_potential=savings,
                expected_delta_soiling_ratio=round(delta_sr, 5),
            )
            hourly_decisions.append(decision)

        # High-level aggregate summary
        summary = {
            "forecast_duration_hours": len(df),
            "discovered_physical_boundaries": {
                "wind_resuspension_threshold_mps": v_crit,
                "critical_relative_humidity_pct": rh_crit,
                "cementation_dew_deficit_celsius": dew_def_crit,
            },
            "operational_highlights": {
                "total_cementation_alerts_count": total_cementation_alerts,
                "free_cleaning_windows_count": total_wind_cleaning_hours,
                "net_forecasted_soiling_delta": round(cumulative_sr_impact, 4),
                "primary_recommendation": (
                    "DISPATCH_PREVENTIVE_CLEANING"
                    if total_cementation_alerts > 0
                    else ("DEFER_ROBOTIC_CLEANING" if total_wind_cleaning_hours >= 2 else "MAINTAIN_SCHEDULE")
                ),
            },
        }

        return hourly_decisions, summary

    def export_operational_report(
        self,
        hourly_decisions: List[CleaningDispatchDecision],
        summary: Dict[str, Any],
        output_file: Union[str, Path] = "reports/operational_decision_report.json",
    ) -> Path:
        """
        Saves the complete operational dispatch report to JSON.
        """
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "summary": summary,
            "hourly_schedule": [asdict(d) for d in hourly_decisions],
        }
        with open(path, "w") as f:
            json.dump(report_payload, f, indent=2)
        logger.info("Saved operational cleaning dispatch report to: %s", path)
        return path

    @staticmethod
    def generate_sample_forecast(hours: int = 48, seed: int = 101) -> pd.DataFrame:
        """
        Generates realistic 24-48h ahead hourly weather forecast time series for testing.
        Includes a nocturnal dew cementation event in night 1 and high-wind cleaning in day 2.
        """
        rng = np.random.RandomState(seed)
        times = pd.date_range("2026-10-28 00:00:00", periods=hours, freq="1h")
        h = times.hour.values

        # Temperature & RH diurnal cycle
        temp_amb = 22.0 + 8.0 * np.sin(2 * np.pi * (h - 8) / 24) + rng.normal(0, 0.5, size=hours)
        rh = 60.0 - 28.0 * np.sin(2 * np.pi * (h - 8) / 24)

        # Force near-dew conditions in Night 1 (hours 3 to 6)
        rh[3:7] = 92.0 + rng.uniform(0, 4, size=4)
        temp_amb[3:7] = 14.5 + rng.uniform(0, 0.5, size=4)

        # Wind speeds with strong afternoon gusts in Day 2 (hours 34 to 38)
        wind = rng.weibull(2.0, size=hours) * 3.5
        wind[34:39] = 8.5 + rng.uniform(0, 2.5, size=5)  # exceeds ~6.8 m/s v_crit!

        # Particulate matter with elevated dust
        pm10 = 45.0 + rng.gamma(2.0, 15.0, size=hours)
        pm10[2:8] = 110.0 + rng.uniform(10, 40, size=6)  # high PM coinciding with dew in Night 1
        pm2_5 = pm10 * 0.42

        rainfall = np.zeros(hours)

        return pd.DataFrame(
            {
                "datetime": times,
                "temp_ambient": np.round(temp_amb, 1),
                "relative_humidity": np.round(rh, 1),
                "wind_speed": np.round(wind, 1),
                "rainfall": rainfall,
                "pm10": np.round(pm10, 1),
                "pm2_5": np.round(pm2_5, 1),
            }
        )


if __name__ == "__main__":
    tp = DiscoveredTippingPoints(
        wind_resuspension_threshold_mps=6.8,
        critical_relative_humidity_pct=80.0,
        cementation_dew_deficit_celsius=1.5,
        wind_zero_crossing_mps=6.8,
        wind_knee_point_mps=6.5,
        confidence_score=0.95,
        notes={},
    )
    engine = OperationalEngine(tp)
    fc = engine.generate_sample_forecast(48)
    decisions, summary = engine.evaluate_forecast(fc)
    report_path = engine.export_operational_report(decisions, summary)
    print("Exported report:", report_path)
    print("Summary:", summary["operational_highlights"])
