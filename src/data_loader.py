"""
Data Ingestion and Preprocessing Module for PV Soiling & Weather Data.

Handles multi-station dataset loading, canonical schema harmonization,
temporal alignment, physical boundary sanity filtering, and empirical calibration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PVDataLoader")

CANONICAL_COLUMNS: Dict[str, str] = {
    "datetime": "datetime",
    "power_clean": "power_clean",
    "power_soiled": "power_soiled",
    "irradiance": "irradiance",
    "temp_ambient": "temp_ambient",
    "temp_cell": "temp_cell",
    "relative_humidity": "relative_humidity",
    "wind_speed": "wind_speed",
    "rainfall": "rainfall",
    "pm2_5": "pm2_5",
    "pm10": "pm10",
}

COLUMN_ALIASES: Dict[str, List[str]] = {
    "datetime": ["timestamp", "time", "date", "datetime", "date_time", "dt"],
    "power_clean": [
        "clean pv power",
        "power_clean",
        "clean_power",
        "p_clean",
        "pclean",
        "clean power (w)",
        "clean power (kw)",
        "pv_clean_w",
        "clean_pv_power",
    ],
    "power_soiled": [
        "soiled pv power",
        "power_soiled",
        "soiled_power",
        "p_soiled",
        "psoiled",
        "soiled power (w)",
        "soiled power (kw)",
        "pv_soiled_w",
        "soiled_pv_power",
    ],
    "irradiance": [
        "solar irradiance",
        "irradiance",
        "ghi",
        "poa",
        "solar_irradiance",
        "glob_irradiance",
        "irradiance (w/m2)",
        "ghi (w/m2)",
        "poa_irradiance",
    ],
    "temp_ambient": [
        "ambient temperature",
        "temp_ambient",
        "t_ambient",
        "tamb",
        "temp_amb",
        "ambient_temp",
        "t_amb",
        "ambient temperature (c)",
    ],
    "temp_cell": [
        "module temperature",
        "cell temperature",
        "temp_cell",
        "temp_module",
        "t_module",
        "tcell",
        "t_cell",
        "module_temp",
        "cell_temperature (c)",
    ],
    "relative_humidity": [
        "relative humidity",
        "relative_humidity",
        "rh",
        "humidity",
        "rh (%)",
        "rel_humidity",
        "relative_humidity (%)",
    ],
    "wind_speed": [
        "wind speed",
        "wind_speed",
        "ws",
        "v_wind",
        "windspeed",
        "wind speed (m/s)",
        "wind_vel",
    ],
    "rainfall": [
        "rainfall",
        "rain",
        "precipitation",
        "rain_mm",
        "rainfall (mm)",
        "precip",
    ],
    "pm2_5": ["pm2.5", "pm25", "pm2_5", "particulate_matter_2_5", "pm2.5 (ug/m3)"],
    "pm10": ["pm10", "pm_10", "particulate_matter_10", "pm10 (ug/m3)"],
}


class PVDataLoader:
    """Ingests, cleans, harmonizes, and validates multi-station PV soiling time series."""

    def __init__(self, raw_dir: Union[str, Path] = "data/raw", processed_dir: Union[str, Path] = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def harmonize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps raw column names to canonical internal names using fuzzy alias matching."""
        df = df.copy()
        raw_col_lower = {str(col).strip().lower(): col for col in df.columns}
        rename_map: Dict[str, str] = {}

        for canonical, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if alias.lower() in raw_col_lower:
                    orig_col = raw_col_lower[alias.lower()]
                    rename_map[orig_col] = canonical
                    break

        df = df.rename(columns=rename_map)
        missing_cols = [c for c in CANONICAL_COLUMNS if c not in df.columns]

        if missing_cols:
            logger.warning("Columns missing from canonical schema: %s", missing_cols)
            if "temp_cell" in missing_cols and "temp_ambient" in df.columns and "irradiance" in df.columns:
                logger.info("Approximating missing 'temp_cell' using Sandia PV thermal model.")
                ws = df["wind_speed"].values if "wind_speed" in df.columns else 2.0
                delta_t = df["irradiance"].clip(lower=0.0) * np.exp(-3.47 + 0.0594 * ws)
                df["temp_cell"] = df["temp_ambient"] + delta_t
            if "rainfall" in missing_cols:
                df["rainfall"] = 0.0
            if "pm2_5" in missing_cols and "pm10" in df.columns:
                df["pm2_5"] = df["pm10"] * 0.42

        return df

    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses timestamps, enforces strictly monotonic ordering, applies physical boundaries,
        and interpolates minor missing values.
        """
        df = df.copy()

        if "datetime" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df["datetime"] = df.index
            else:
                for col in df.columns:
                    try:
                        df["datetime"] = pd.to_datetime(df[col])
                        break
                    except Exception:
                        continue

        if "datetime" not in df.columns:
            raise ValueError("No datetime column found or convertible in DataFrame.")

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

        bounds = {
            "irradiance": (0.0, 1400.0),
            "relative_humidity": (0.0, 100.0),
            "wind_speed": (0.0, 50.0),
            "temp_ambient": (-40.0, 65.0),
            "temp_cell": (-40.0, 95.0),
            "rainfall": (0.0, 250.0),
            "pm2_5": (0.0, 1200.0),
            "pm10": (0.0, 2500.0),
            "power_clean": (0.0, None),
            "power_soiled": (0.0, None),
        }

        for col, (low, high) in bounds.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if low is not None:
                    df[col] = df[col].clip(lower=low)
                if high is not None:
                    df[col] = df[col].clip(upper=high)

        if "rainfall" in df.columns:
            df["rainfall"] = df["rainfall"].fillna(0.0)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=3).bfill().ffill()

        return df

    def list_available_datasets(self) -> List[Path]:
        """Returns list of CSV/Parquet files present in data/raw."""
        return sorted(list(self.raw_dir.glob("*.csv")) + list(self.raw_dir.glob("*.parquet")))

    def load_data(
        self,
        file_path: Optional[Union[str, Path]] = None,
        station_name: Optional[str] = None,
        auto_generate_if_empty: bool = True,
    ) -> pd.DataFrame:
        """
        Loads dataset from file_path, station_name, or default primary cohort in raw_dir.
        """
        target_path: Optional[Path] = None

        if file_path is not None:
            target_path = Path(file_path)
        elif station_name is not None:
            candidates = list(self.raw_dir.glob(f"*{station_name.lower()}*"))
            if candidates:
                target_path = candidates[0]
        else:
            # Look for primary arid station first, then any mendeley file
            arid_files = list(self.raw_dir.glob("*station_a_arid_desert*.csv"))
            if arid_files:
                target_path = arid_files[0]
            else:
                all_files = self.list_available_datasets()
                if all_files:
                    target_path = all_files[0]

        if target_path is not None and target_path.exists():
            logger.info("Loading dataset from: %s", target_path)
            if target_path.suffix.lower() == ".parquet":
                raw_df = pd.read_parquet(target_path)
            else:
                raw_df = pd.read_csv(target_path)

            harmonized = self.harmonize_columns(raw_df)
            cleaned = self.clean_and_validate(harmonized)
            return cleaned

        if auto_generate_if_empty:
            logger.warning("No dataset found. Invoking DataDownloader cohort generation...")
            from src.data_download import DataDownloader

            downloader = DataDownloader(raw_dir=self.raw_dir)
            primary_path = downloader.download_mendeley_dataset()
            return self.load_data(file_path=primary_path, auto_generate_if_empty=False)

        raise FileNotFoundError(f"No valid dataset found in '{self.raw_dir}'.")


if __name__ == "__main__":
    loader = PVDataLoader()
    df = loader.load_data()
    print("Loaded canonical shape:", df.shape)
    print("Columns:", list(df.columns))
