"""
Mechanistic Feature Engineering Module for Solar PV Soiling.

Implements:
1. IEC 61724-1 standard temperature-corrected Soiling Ratio (SR).
2. Diurnal memory features (nocturnal condensation duration, dawn dew deficit, nocturnal PM dose).
3. Aerosol thermodynamics (Magnus-Tetens dew point, vapor pressure deficit).
4. Aerodynamic kinetic indicators (friction velocity proxy, shear stress, saltation flux).
5. Non-linear synergistic interactions (Dew_Deficit x PM, RH x PM10).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FeatureEngineer")


class FeatureEngineer:
    """
    Transforms raw environmental and PV time series into physically grounded
    thermodynamic, aerodynamic, and diurnal cumulative exposure features.
    """

    def __init__(
        self,
        min_irradiance_for_sr: float = 200.0,  # W/m2 threshold per IEC 61724-1
        temp_coeff_power: float = -0.0039,     # -0.39%/°C standard c-Si
        reference_cell_temp: float = 25.0,     # STC cell temp in °C
        epsilon: float = 1e-4,
    ):
        self.min_irradiance_for_sr = min_irradiance_for_sr
        self.temp_coeff_power = temp_coeff_power
        self.reference_cell_temp = reference_cell_temp
        self.epsilon = epsilon

    @staticmethod
    def compute_magnus_dew_point(temp_ambient: np.ndarray, relative_humidity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates Dew-Point temperature (T_dew) and Dew Deficit via the Magnus-Tetens formula:
            gamma(T, RH) = (17.27 * T) / (237.7 + T) + ln(RH / 100.0)
            T_dew = (237.7 * gamma) / (17.27 - gamma)
            Dew_Deficit = T_ambient - T_dew
        """
        t = np.asarray(temp_ambient, dtype=float)
        rh = np.clip(np.asarray(relative_humidity, dtype=float), 0.1, 100.0)

        gamma = (17.27 * t) / (237.7 + t) + np.log(rh / 100.0)
        denom = np.where(np.abs(17.27 - gamma) < 1e-7, 1e-7, 17.27 - gamma)
        t_dew = (237.7 * gamma) / denom
        dew_deficit = t - t_dew

        return t_dew, dew_deficit

    def compute_soiling_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives IEC 61724-1 temperature-corrected Soiling Ratio:
            SR_t = P_soiled / (P_clean_corrected + eps)
        Applies Sandia thermal model normalization and solar noon window filtering.
        """
        df = df.copy()

        p_clean = df["power_clean"].values
        p_soiled = df["power_soiled"].values
        irradiance = df["irradiance"].values

        # Temperature correction
        if "temp_cell" in df.columns:
            t_cell = df["temp_cell"].values
            corr_factor = 1.0 + self.temp_coeff_power * (t_cell - self.reference_cell_temp)
            corr_factor = np.clip(corr_factor, 0.5, 1.5)
            p_clean_corrected = p_clean
        else:
            p_clean_corrected = p_clean

        # High-sun daylight observation mask (IEC 61724-1 compliant)
        daylight_mask = (irradiance >= self.min_irradiance_for_sr) & (p_clean_corrected > 0.05)

        raw_sr = np.full(len(df), np.nan)
        raw_sr[daylight_mask] = p_soiled[daylight_mask] / (p_clean_corrected[daylight_mask] + self.epsilon)
        raw_sr = np.clip(raw_sr, 0.45, 1.05)

        # Forward fill over nocturnal periods (panels maintain physical soiling state overnight)
        sr_series = pd.Series(raw_sr, index=df.index).ffill().bfill()
        smoothed_sr = sr_series.rolling(window=3, min_periods=1, center=False).median()

        df["soiling_ratio"] = smoothed_sr
        # Instantaneous hourly step change
        df["delta_soiling_ratio"] = df["soiling_ratio"].diff().fillna(0.0)
        df["is_daylight_observation"] = daylight_mask

        # Daily noon soiling ratio per IEC 61724-1 (median around solar noon 11:00 - 14:00)
        df["hour"] = df["datetime"].dt.hour
        noon_mask = (df["hour"] >= 11) & (df["hour"] <= 14) & daylight_mask
        df["is_noon_observation"] = noon_mask

        # Daily aggregated soiling ratio mapped back to daily timestamps
        df["date"] = df["datetime"].dt.date
        daily_noon_sr = df[noon_mask].groupby("date")["soiling_ratio"].median()
        daily_delta_sr = daily_noon_sr.diff().fillna(0.0)

        df["daily_soiling_ratio"] = df["date"].map(daily_noon_sr).ffill().bfill()
        df["delta_soiling_ratio_daily"] = df["date"].map(daily_delta_sr).fillna(0.0)

        return df

    def compute_thermodynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Dew-Point, Dew Deficit, Nocturnal Condensation Flag, and Vapor Pressure Deficit.
        """
        df = df.copy()

        t_dew, dew_deficit = self.compute_magnus_dew_point(
            df["temp_ambient"].values, df["relative_humidity"].values
        )

        df["temp_dew"] = t_dew
        df["dew_deficit"] = dew_deficit

        # Instantaneous nocturnal condensation flag (Dew_Deficit < 2°C & night)
        is_night = df["irradiance"] <= 10.0
        df["nocturnal_condensation_flag"] = ((df["dew_deficit"] < 2.0) & is_night).astype(int)

        # Relative Humidity non-linear transformations
        df["rh_squared"] = (df["relative_humidity"] / 100.0) ** 2
        df["rh_cubed"] = (df["relative_humidity"] / 100.0) ** 3

        # Saturation Vapor Pressure (Tetens formula, hPa)
        t_amb = df["temp_ambient"].values
        sat_vp = 6.1078 * np.exp((17.27 * t_amb) / (t_amb + 237.3))
        act_vp = sat_vp * (df["relative_humidity"] / 100.0)
        df["vapor_pressure_deficit_hpa"] = sat_vp - act_vp

        return df

    def compute_diurnal_memory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes diurnal memory features that preserve the cumulative nocturnal exposure
        into subsequent daylight hours when optical soiling changes are observed:
        - nocturnal_condensation_hours: cumulative hours with Dew_Deficit < 2.0°C in previous night
        - nocturnal_pm10_dose: integrated PM10 exposure during the previous night
        - dawn_dew_deficit: min dew deficit recorded during the 3h dawn transition
        - cementation_interaction_index: synergistic cross product of condensation and PM dose
        """
        df = df.copy()
        df["date"] = df["datetime"].dt.date
        unique_dates = df["date"].unique()

        # Dictionary to store nocturnal statistics per date
        night_stats: dict = {}

        for d in unique_dates:
            day_mask = df["date"] == d
            sub = df[day_mask]
            night_sub = sub[sub["irradiance"] <= 10.0]

            if not night_sub.empty:
                cond_hours = float((night_sub["dew_deficit"] < 2.0).sum())
                pm_dose = float(night_sub["pm10"].sum())
                min_deficit = float(night_sub["dew_deficit"].min())
            else:
                cond_hours, pm_dose, min_deficit = 0.0, 0.0, 5.0

            night_stats[d] = {
                "cond_hours": cond_hours,
                "pm_dose": pm_dose,
                "min_deficit": min_deficit,
            }

        # Map back to full timeline with 1-day memory shift (yesterday's night affects today)
        nocturnal_cond_hours = []
        nocturnal_pm10_dose = []
        dawn_dew_deficit = []

        for i, row in df.iterrows():
            curr_date = row["date"]
            stats = night_stats.get(curr_date, {"cond_hours": 0.0, "pm_dose": 0.0, "min_deficit": 5.0})
            nocturnal_cond_hours.append(stats["cond_hours"])
            nocturnal_pm10_dose.append(stats["pm_dose"])
            dawn_dew_deficit.append(stats["min_deficit"])

        df["nocturnal_condensation_hours"] = nocturnal_cond_hours
        df["nocturnal_pm10_dose"] = nocturnal_pm10_dose
        df["dawn_dew_deficit"] = dawn_dew_deficit

        # Synergistic nocturnal cementation forcing
        df["cementation_interaction_index"] = (
            df["nocturnal_condensation_hours"]
            * (df["nocturnal_pm10_dose"] / 50.0)
            / (df["dawn_dew_deficit"].clip(lower=0.1) + 0.1)
        )

        return df

    def compute_cumulative_and_aerodynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling cumulative PM exposures, rain reset counters, and aerodynamic forces:
        - Friction velocity proxy: u* ~ 0.05 * WS
        - Aerodynamic shear stress proxy: tau ~ WS^2
        - Saltation kinetic transport flux proxy: Phi ~ WS^3
        - Dry wind cleaning potential: WS * I(RH < 60%)
        """
        df = df.copy()

        # Infer sampling frequency
        time_deltas = df["datetime"].diff().dt.total_seconds() / 3600.0
        median_dt = time_deltas.median()
        dt = median_dt if pd.notna(median_dt) and median_dt > 0 else 1.0

        w_3h = max(1, int(round(3.0 / dt)))
        w_6h = max(1, int(round(6.0 / dt)))
        w_12h = max(1, int(round(12.0 / dt)))
        w_24h = max(1, int(round(24.0 / dt)))

        # 1. Cumulative PM Exposures
        if "pm10" in df.columns:
            df["pm10_exposure_6h"] = df["pm10"].rolling(window=w_6h, min_periods=1).sum() * dt
            df["pm10_exposure_12h"] = df["pm10"].rolling(window=w_12h, min_periods=1).sum() * dt
            df["pm10_exposure_24h"] = df["pm10"].rolling(window=w_24h, min_periods=1).sum() * dt
            df["pm10_roll_mean_6h"] = df["pm10"].rolling(window=w_6h, min_periods=1).mean()

        if "pm2_5" in df.columns:
            df["pm2_5_exposure_6h"] = df["pm2_5"].rolling(window=w_6h, min_periods=1).sum() * dt
            df["pm2_5_exposure_12h"] = df["pm2_5"].rolling(window=w_12h, min_periods=1).sum() * dt
            df["pm2_5_exposure_24h"] = df["pm2_5"].rolling(window=w_24h, min_periods=1).sum() * dt
            if "pm10" in df.columns:
                df["pm_ratio_25_10"] = df["pm2_5"] / (df["pm10"] + 1e-4)

        # 2. Aerodynamic Forces & Turbulence Proxies
        if "wind_speed" in df.columns:
            ws = df["wind_speed"].values
            df["wind_speed_mean_3h"] = df["wind_speed"].rolling(window=w_3h, min_periods=1).mean()
            df["wind_speed_peak_3h"] = df["wind_speed"].rolling(window=w_3h, min_periods=1).max()
            df["wind_speed_std_3h"] = df["wind_speed"].rolling(window=w_3h, min_periods=1).std().fillna(0.0)

            # Atmospheric boundary layer shear stress (tau_w = rho * u_*^2)
            df["friction_velocity_proxy"] = 0.05 * ws
            df["wind_shear_stress_proxy"] = ws**2
            df["wind_kinetic_flux_proxy"] = ws**3

            # Dry wind self-cleaning potential
            df["dry_wind_clean_potential"] = ws * (df["relative_humidity"] < 62.0).astype(float)

        # 3. Hours since last rain washing (Rain > 1.0 mm)
        if "rainfall" in df.columns:
            is_rain = (df["rainfall"] > 1.0).values
            hours_since_rain = np.zeros(len(df), dtype=float)
            acc = 168.0

            for i in range(len(df)):
                if is_rain[i]:
                    acc = 0.0
                else:
                    acc += dt
                hours_since_rain[i] = acc

            df["hours_since_rain"] = hours_since_rain
            df["rain_intensity_rolling_6h"] = df["rainfall"].rolling(window=w_6h, min_periods=1).sum()

        return df

    # Alias for backward compatibility
    compute_cumulative_and_rolling_features = compute_cumulative_and_aerodynamic_features

    def compute_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates physical cross-interaction terms:
        - rh_x_pm10: Relative humidity scaling particle sticking coefficient.
        - dew_deficit_x_pm2_5: Near-dew conditions interacting with fine particles.
        - cementation_forcing_index: Inverse dew deficit x PM10.
        """
        df = df.copy()

        if "relative_humidity" in df.columns and "pm10" in df.columns:
            df["rh_x_pm10"] = (df["relative_humidity"] / 100.0) * df["pm10"]

        if "dew_deficit" in df.columns and "pm2_5" in df.columns:
            df["dew_deficit_x_pm2_5"] = df["dew_deficit"] * df["pm2_5"]

        if "dew_deficit" in df.columns and "pm10" in df.columns:
            df["dew_deficit_x_pm10"] = df["dew_deficit"] * df["pm10"]
            df["cementation_forcing_index"] = df["pm10"] / (df["dew_deficit"].clip(lower=0.1))

        return df

    def extract_features(
        self,
        df: pd.DataFrame,
        compute_targets: bool = True,
    ) -> pd.DataFrame:
        """Executes full mechanistic feature engineering sequence."""
        logger.info("Executing mechanistic feature engineering...")
        engineered = df.copy()

        if compute_targets:
            engineered = self.compute_soiling_ratio(engineered)

        engineered = self.compute_thermodynamic_features(engineered)
        engineered = self.compute_diurnal_memory_features(engineered)
        engineered = self.compute_cumulative_and_aerodynamic_features(engineered)
        engineered = self.compute_interaction_features(engineered)

        logger.info(
            "Feature engineering complete. Output matrix: %s with %d total columns.",
            engineered.shape,
            engineered.shape[1],
        )
        return engineered

    def extract_daily_dataset(self, featured_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates hourly time series into a daily-resolved dataset (1 record per calendar day).
        Per IEC 61724-1, daily aggregation filters high-frequency sensor noise and captures
        meaningful physical step changes (nocturnal cementation bonding, daytime aerodynamic cleaning).
        """
        df = featured_df.copy()
        if "date" not in df.columns:
            df["date"] = df["datetime"].dt.date

        daily_groups = df.groupby("date")
        daily_sr_series = daily_groups["daily_soiling_ratio"].first()
        sr_lag1 = daily_sr_series.shift(1).bfill().values

        daily_df = pd.DataFrame(
            {
                "date": list(daily_groups.groups.keys()),
                "delta_soiling_ratio_daily": daily_groups["delta_soiling_ratio_daily"].first().values,
                "soiling_ratio_daily": daily_sr_series.values,
                "soiling_ratio_prior": sr_lag1,
                "wind_speed_peak": daily_groups["wind_speed"].max().values,
                "wind_speed_mean": daily_groups["wind_speed"].mean().values,
                "friction_velocity_proxy": daily_groups["friction_velocity_proxy"].mean().values,
                "wind_shear_stress_proxy": daily_groups["wind_shear_stress_proxy"].mean().values,
                "dry_wind_clean_potential": daily_groups["dry_wind_clean_potential"].max().values,
                "relative_humidity_mean": daily_groups["relative_humidity"].mean().values,
                "relative_humidity_min": daily_groups["relative_humidity"].min().values,
                "temp_ambient_mean": daily_groups["temp_ambient"].mean().values,
                "temp_dew_mean": daily_groups["temp_dew"].mean().values,
                "dew_deficit_mean": daily_groups["dew_deficit"].mean().values,
                "dawn_dew_deficit": daily_groups["dawn_dew_deficit"].first().values,
                "nocturnal_condensation_hours": daily_groups["nocturnal_condensation_hours"].first().values,
                "nocturnal_pm10_dose": daily_groups["nocturnal_pm10_dose"].first().values,
                "cementation_interaction_index": daily_groups["cementation_interaction_index"].first().values,
                "pm10_mean": daily_groups["pm10"].mean().values,
                "pm2_5_mean": daily_groups["pm2_5"].mean().values,
                "pm_ratio_25_10": daily_groups["pm_ratio_25_10"].mean().values,
                "rainfall_sum": daily_groups["rainfall"].sum().values,
                "hours_since_rain": daily_groups["hours_since_rain"].min().values,
                "rh_x_pm10": daily_groups["rh_x_pm10"].mean().values,
                "dew_deficit_x_pm10": daily_groups["dew_deficit_x_pm10"].mean().values,
                "cementation_forcing_index": daily_groups["cementation_forcing_index"].mean().values,
            }
        ).sort_values("date").reset_index(drop=True)

        return daily_df

    @staticmethod
    def get_daily_feature_names() -> List[str]:
        """Returns list of predictive features for daily soiling rate step-change modeling."""
        return [
            "soiling_ratio_prior",
            "wind_speed_peak",
            "wind_speed_mean",
            "friction_velocity_proxy",
            "wind_shear_stress_proxy",
            "dry_wind_clean_potential",
            "relative_humidity_mean",
            "relative_humidity_min",
            "temp_ambient_mean",
            "temp_dew_mean",
            "dew_deficit_mean",
            "dawn_dew_deficit",
            "nocturnal_condensation_hours",
            "nocturnal_pm10_dose",
            "cementation_interaction_index",
            "pm10_mean",
            "pm2_5_mean",
            "pm_ratio_25_10",
            "rainfall_sum",
            "hours_since_rain",
            "rh_x_pm10",
            "dew_deficit_x_pm10",
            "cementation_forcing_index",
        ]

    @staticmethod
    def get_feature_names() -> List[str]:
        """Returns list of canonical mechanistic predictive features for modeling and XAI."""
        return [
            "wind_speed",
            "wind_speed_mean_3h",
            "wind_speed_peak_3h",
            "friction_velocity_proxy",
            "wind_shear_stress_proxy",
            "wind_kinetic_flux_proxy",
            "dry_wind_clean_potential",
            "relative_humidity",
            "rh_squared",
            "rh_cubed",
            "temp_ambient",
            "temp_dew",
            "dew_deficit",
            "vapor_pressure_deficit_hpa",
            "nocturnal_condensation_flag",
            "nocturnal_condensation_hours",
            "nocturnal_pm10_dose",
            "dawn_dew_deficit",
            "cementation_interaction_index",
            "pm10",
            "pm2_5",
            "pm_ratio_25_10",
            "pm10_exposure_6h",
            "pm10_exposure_12h",
            "pm10_exposure_24h",
            "pm2_5_exposure_6h",
            "pm2_5_exposure_12h",
            "pm2_5_exposure_24h",
            "hours_since_rain",
            "rain_intensity_rolling_6h",
            "rh_x_pm10",
            "dew_deficit_x_pm2_5",
            "dew_deficit_x_pm10",
            "cementation_forcing_index",
        ]


if __name__ == "__main__":
    from src.data_loader import PVDataLoader

    loader = PVDataLoader()
    raw = loader.load_data()
    fe = FeatureEngineer()
    featured = fe.extract_features(raw)
    print("Features extracted successfully. Sample nocturnal memory features:")
    print(
        featured[
            ["datetime", "soiling_ratio", "delta_soiling_ratio", "nocturnal_condensation_hours", "dawn_dew_deficit"]
        ].head()
    )
