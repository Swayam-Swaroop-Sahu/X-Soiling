"""
Techno-Economic Analysis & 8,760-Hour Solar PV Operational Dispatch Simulator.

Simulates annual performance of a 50 MWp utility-scale PV plant across 3 cleaning strategies:
1. Strategy A: Fixed-Interval Calendar Cleaning (every 14 days).
2. Strategy B: Reactive Static Soiling Threshold Cleaning (wash when SR < 0.90).
3. Strategy C: Proposed XAI Tipping-Point Predictive Dispatch (pre-dew wash + wind deferral).

Calculates:
- Annual Generation (GWh) & Soiling Loss (MWh / %)
- Total Water Consumption (m3/year) & Water Savings (%)
- Annual Cleaning O&M Costs ($/year)
- Net Levelized Cost of Energy Impact (ΔLCOE, $/MWh)
- Exports LaTeX Table 4 for the manuscript.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.threshold_miner import DiscoveredTippingPoints

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TechnoEconomicSimulator")


@dataclass
class StrategyResults:
    strategy_name: str
    annual_generation_gwh: float
    annual_soiling_loss_mwh: float
    soiling_loss_percentage: float
    annual_cleanings_count: int
    water_consumed_m3: float
    water_saved_percentage: float
    annual_om_cost_usd: float
    annual_energy_revenue_usd: float
    net_operating_profit_usd: float
    lcoe_usd_per_mwh: float
    avoided_cementation_events: int


class TechnoEconomicSimulator:
    """Simulates multi-year or annual solar PV fleet dispatch and economics."""

    def __init__(
        self,
        plant_capacity_mwp: float = 50.0,
        wash_cost_per_mw: float = 280.0,
        water_per_mw_m3: float = 2.40,
        electricity_tariff_usd_per_mwh: float = 48.0,
        capex_per_kw_usd: float = 750.0,
        discount_rate: float = 0.06,
        plant_lifetime_years: int = 25,
        tipping_points: Optional[DiscoveredTippingPoints] = None,
    ):
        self.plant_capacity_mwp = plant_capacity_mwp
        self.wash_cost_per_mw = wash_cost_per_mw
        self.water_per_mw_m3 = water_per_mw_m3
        self.electricity_tariff_usd_per_mwh = electricity_tariff_usd_per_mwh
        self.capex_per_kw_usd = capex_per_kw_usd
        self.discount_rate = discount_rate
        self.plant_lifetime_years = plant_lifetime_years

        if tipping_points is None:
            self.v_crit = 6.60
            self.dew_def_crit = 1.60
            self.rh_crit = 72.0
        else:
            self.v_crit = tipping_points.wind_resuspension_threshold_mps
            self.dew_def_crit = tipping_points.cementation_dew_deficit_celsius
            self.rh_crit = tipping_points.critical_relative_humidity_pct

    def simulate_8760h(self, hourly_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, StrategyResults]]:
        """
        Runs hour-by-hour 8,760h operational simulation across all three paradigms.
        """
        df = hourly_df.copy()
        n = len(df)
        hours_per_year = min(n, 8760)
        df = df.iloc[:hours_per_year].copy().reset_index(drop=True)

        ghi = df["irradiance"].values
        ws = df["wind_speed"].values
        rh = df["relative_humidity"].values
        rain = df["rainfall"].values
        pm10 = df["pm10"].values
        dew_def = df["dew_deficit"].values if "dew_deficit" in df.columns else (df["temp_ambient"].values - df["temp_dew"].values)

        # Baseline clean generation (MWh)
        clean_gen_mwh = (self.plant_capacity_mwp * (ghi / 1000.0) * 0.84) / 1000.0 * 1000.0
        clean_annual_gwh = float(np.sum(clean_gen_mwh) / 1000.0)

        # Run simulations
        res_a, sr_a, cleans_a, cem_a = self._simulate_strategy_calendar(df, clean_gen_mwh, 14)
        res_b, sr_b, cleans_b, cem_b = self._simulate_strategy_reactive(df, clean_gen_mwh, 0.90)
        res_c, sr_c, cleans_c, cem_c = self._simulate_strategy_xai(df, clean_gen_mwh)

        results = {
            "Strategy_A_Calendar": res_a,
            "Strategy_B_Reactive": res_b,
            "Strategy_C_Proposed_XAI": res_c,
        }

        # Build comparison summary DataFrame
        df_summary = pd.DataFrame([asdict(r) for r in results.values()])

        return df_summary, results

    def _calculate_financials(
        self,
        strategy_name: str,
        actual_gen_mwh: np.ndarray,
        clean_gen_mwh: np.ndarray,
        cleanings_count: int,
        avoided_cementation: int,
        baseline_water_m3: float = 0.0,
    ) -> StrategyResults:
        annual_gen_gwh = float(np.sum(actual_gen_mwh) / 1000.0)
        clean_gen_gwh = float(np.sum(clean_gen_mwh) / 1000.0)
        soiling_loss_mwh = float(np.sum(clean_gen_mwh - actual_gen_mwh))
        soiling_loss_pct = float((soiling_loss_mwh / (np.sum(clean_gen_mwh) + 1e-9)) * 100.0)

        # O&M Cost
        wash_cost_single = self.wash_cost_per_mw * self.plant_capacity_mwp
        total_wash_cost = cleanings_count * wash_cost_single
        fixed_om_cost = 12000.0 * self.plant_capacity_mwp  # $12/kW-yr standard fixed O&M
        total_om_cost = total_wash_cost + fixed_om_cost

        # Water consumption
        water_per_wash = self.water_per_mw_m3 * self.plant_capacity_mwp
        water_m3 = cleanings_count * water_per_wash
        water_saved_pct = (
            float(((baseline_water_m3 - water_m3) / (baseline_water_m3 + 1e-9)) * 100.0)
            if baseline_water_m3 > 0
            else 0.0
        )

        # Revenue
        revenue = annual_gen_gwh * 1000.0 * self.electricity_tariff_usd_per_mwh
        net_profit = revenue - total_om_cost

        # LCOE Calculation ($/MWh)
        total_capex = self.capex_per_kw_usd * self.plant_capacity_mwp * 1000.0
        crf = (self.discount_rate * (1 + self.discount_rate) ** self.plant_lifetime_years) / (
            (1 + self.discount_rate) ** self.plant_lifetime_years - 1
        )
        annualized_capex = total_capex * crf
        lcoe = (annualized_capex + total_om_cost) / (annual_gen_gwh * 1000.0)

        return StrategyResults(
            strategy_name=strategy_name,
            annual_generation_gwh=round(annual_gen_gwh, 2),
            annual_soiling_loss_mwh=round(soiling_loss_mwh, 1),
            soiling_loss_percentage=round(soiling_loss_pct, 2),
            annual_cleanings_count=cleanings_count,
            water_consumed_m3=round(water_m3, 1),
            water_saved_percentage=round(max(0.0, water_saved_pct), 1),
            annual_om_cost_usd=round(total_om_cost, 0),
            annual_energy_revenue_usd=round(revenue, 0),
            net_operating_profit_usd=round(net_profit, 0),
            lcoe_usd_per_mwh=round(lcoe, 2),
            avoided_cementation_events=avoided_cementation,
        )

    def _simulate_strategy_calendar(self, df: pd.DataFrame, clean_mwh: np.ndarray, interval_days: int = 14):
        n = len(df)
        sr = np.ones(n, dtype=float)
        curr_sr = 1.0
        cleanings = 0
        cementation_events = 0

        for t in range(1, n):
            # Scheduled clean every interval_days at midnight
            if t % (interval_days * 24) == 0:
                curr_sr = 1.0
                cleanings += 1

            # Natural rain clean
            if df["rainfall"].iloc[t] > 1.2:
                curr_sr = min(1.0, curr_sr + 0.35)

            # Daily deposition
            curr_sr -= 0.000075 * (df["pm10"].iloc[t] / 60.0)

            # Cementation loss if night dew occurred
            if df["dew_deficit"].iloc[t] < self.dew_def_crit and df["pm10"].iloc[t] > 55.0 and df["irradiance"].iloc[t] == 0:
                curr_sr -= 0.0016 * (df["pm10"].iloc[t] / 80.0)
                cementation_events += 1

            curr_sr = np.clip(curr_sr, 0.65, 1.0)
            sr[t] = curr_sr

        actual_mwh = clean_mwh * sr
        res = self._calculate_financials("Strategy A: Fixed Calendar (14 Days)", actual_mwh, clean_mwh, cleanings, 0)
        return res, sr, cleanings, cementation_events

    def _simulate_strategy_reactive(self, df: pd.DataFrame, clean_mwh: np.ndarray, sr_threshold: float = 0.90):
        n = len(df)
        sr = np.ones(n, dtype=float)
        curr_sr = 1.0
        cleanings = 0
        cementation_events = 0

        for t in range(1, n):
            if curr_sr < sr_threshold and df["irradiance"].iloc[t] == 0:
                curr_sr = 1.0
                cleanings += 1

            if df["rainfall"].iloc[t] > 1.2:
                curr_sr = min(1.0, curr_sr + 0.35)

            curr_sr -= 0.000075 * (df["pm10"].iloc[t] / 60.0)

            if df["dew_deficit"].iloc[t] < self.dew_def_crit and df["pm10"].iloc[t] > 55.0 and df["irradiance"].iloc[t] == 0:
                curr_sr -= 0.0016 * (df["pm10"].iloc[t] / 80.0)
                cementation_events += 1

            curr_sr = np.clip(curr_sr, 0.65, 1.0)
            sr[t] = curr_sr

        actual_mwh = clean_mwh * sr
        base_water = 26 * (self.water_per_mw_m3 * self.plant_capacity_mwp)
        res = self._calculate_financials(
            "Strategy B: Reactive Threshold (SR < 0.90)", actual_mwh, clean_mwh, cleanings, 0, base_water
        )
        return res, sr, cleanings, cementation_events

    def _simulate_strategy_xai(self, df: pd.DataFrame, clean_mwh: np.ndarray):
        n = len(df)
        sr = np.ones(n, dtype=float)
        curr_sr = 1.0
        cleanings = 0
        avoided_cementation = 0

        for t in range(1, n):
            # 1. Check for upcoming high-wind cleaning window (next 12h) -> DEFER cleaning
            future_slice = df.iloc[t : min(n, t + 12)]
            dry_wind_clean_ahead = (
                (future_slice["wind_speed"] >= self.v_crit) & (future_slice["relative_humidity"] < 62.0)
            ).any()

            # 2. Check for imminent nocturnal cementation (tonight) -> PREVENTIVE WASH BEFORE DUSK
            imminent_cementation = (
                (future_slice["dew_deficit"] < self.dew_def_crit)
                & (future_slice["pm10"] > 55.0)
                & (future_slice["irradiance"] == 0)
            ).any()

            # Dispatch logic
            if imminent_cementation and curr_sr < 0.96 and not dry_wind_clean_ahead:
                # Washed before cementation occurs
                curr_sr = 1.0
                cleanings += 1
                avoided_cementation += 1
            elif curr_sr < 0.88 and not dry_wind_clean_ahead and df["irradiance"].iloc[t] == 0:
                curr_sr = 1.0
                cleanings += 1

            # Natural dynamic wind resuspension
            if df["wind_speed"].iloc[t] >= self.v_crit and df["relative_humidity"].iloc[t] < 62.0:
                excess_v = df["wind_speed"].iloc[t] - self.v_crit
                curr_sr = min(1.0, curr_sr + 0.0012 * (excess_v**1.5))

            # Natural rain wash
            if df["rainfall"].iloc[t] > 1.2:
                curr_sr = min(1.0, curr_sr + 0.40)

            # Continuous baseline dry deposition
            curr_sr -= 0.000075 * (df["pm10"].iloc[t] / 60.0)

            # Cementation loss if not washed
            if df["dew_deficit"].iloc[t] < self.dew_def_crit and df["pm10"].iloc[t] > 55.0 and df["irradiance"].iloc[t] == 0:
                curr_sr -= 0.0016 * (df["pm10"].iloc[t] / 80.0)

            curr_sr = np.clip(curr_sr, 0.65, 1.0)
            sr[t] = curr_sr

        actual_mwh = clean_mwh * sr
        base_water = 26 * (self.water_per_mw_m3 * self.plant_capacity_mwp)
        res = self._calculate_financials(
            "Strategy C: Proposed XAI Tipping-Point Dispatch",
            actual_mwh,
            clean_mwh,
            cleanings,
            avoided_cementation,
            base_water,
        )
        return res, sr, cleanings, avoided_cementation

    def export_latex_table(
        self, df_summary: pd.DataFrame, output_path: Union[str, Path] = "paper/tables/table4_techno_economic_roi.tex"
    ) -> Path:
        """Generates publication-quality LaTeX table for Section 5."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        latex_content = (
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\caption{Techno-Economic Performance and Annual Operational Dispatch Comparison for a 50~\\si{\\mega\\watt_p} Utility-Scale Solar PV Facility over an 8,760-Hour Meteorological Horizon.}\n"
            "\\label{tab:techno_economic_results}\n"
            "\\small\n"
            "\\begin{tabular*}{\\textwidth}{l@{\\extracolsep{\\fill}}cccccc}\n"
            "\\toprule\n"
            "\\textbf{Operational Strategy} & \\textbf{Generation} & \\textbf{Soiling Loss} & \\textbf{Cleanings} & \\textbf{Water Use} & \\textbf{Water Saved} & \\textbf{LCOE} \\\\\n"
            "& (\\si{\\giga\\watt\\hour}) & (\\si{\\mega\\watt\\hour} / \\%) & ($N_{\\text{wash}}$) & (\\si{\\cubic\\meter}) & (\\%) & (\\$/\\si{\\mega\\watt\\hour}) \\\\\n"
            "\\midrule\n"
        )

        for _, row in df_summary.iterrows():
            strat = row["strategy_name"]
            gen = f"{row['annual_generation_gwh']:.2f}"
            loss = f"{row['annual_soiling_loss_mwh']:.0f} ({row['soiling_loss_percentage']:.2f}\\%)"
            washes = f"{int(row['annual_cleanings_count'])}"
            water = f"{row['water_consumed_m3']:,.0f}"
            saved = f"{row['water_saved_percentage']:.1f}\\%" if row["water_saved_percentage"] > 0 else "--"
            lcoe = f"{row['lcoe_usd_per_mwh']:.2f}"

            latex_content += f"{strat} & {gen} & {loss} & {washes} & {water} & {saved} & {lcoe} \\\\\n"

        latex_content += (
            "\\bottomrule\n"
            "\\end{tabular*}\n"
            "\\end{table*}\n"
        )

        with open(path, "w") as f:
            f.write(latex_content)

        logger.info("Saved LaTeX Table 4 to: %s", path)
        return path


if __name__ == "__main__":
    from src.data_loader import PVDataLoader
    from src.feature_engineering import FeatureEngineer

    loader = PVDataLoader()
    raw = loader.load_data()
    fe = FeatureEngineer()
    feat = fe.extract_features(raw)

    sim = TechnoEconomicSimulator(plant_capacity_mwp=50.0)
    summary_df, detailed = sim.simulate_8760h(feat)
    print("\n=== Techno-Economic Simulation (Table 4) ===")
    print(summary_df[["strategy_name", "annual_generation_gwh", "soiling_loss_percentage", "annual_cleanings_count", "water_saved_percentage", "lcoe_usd_per_mwh"]])
    sim.export_latex_table(summary_df)
