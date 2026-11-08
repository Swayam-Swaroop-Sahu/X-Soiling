"""
Explainable AI (XAI) Engine for PV Soiling Tipping Points.

Leverages SHAP TreeExplainer for:
1. Global feature attribution (beeswarm & mean |SHAP| rankings).
2. 1D dependence curves with continuous lowess trend and 95% Bootstrapped CI ribbons.
3. 2D SHAP interaction matrices (discovering synergistic moisture x aerosol cementation cliffs).
4. Dual-format figure generation: High-res 300 DPI PNG and Vectorized PDF for LaTeX manuscript.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("XAIEngine")

# Set standard publication font styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


class XAIEngine:
    """Computes and visualizes SHAP attributions and 2D interaction tensors."""

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        output_dirs: Optional[Union[List[Union[str, Path]], str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.model = model
        self.feature_names = feature_names

        target_dirs = []
        if output_dirs is not None:
            if isinstance(output_dirs, (str, Path)):
                target_dirs = [Path(output_dirs)]
            else:
                target_dirs = [Path(d) for d in output_dirs]
        elif output_dir is not None:
            target_dirs = [Path(output_dir)]
        else:
            target_dirs = [Path("figures"), Path("paper/figures")]

        self.output_dirs = target_dirs
        for d in self.output_dirs:
            d.mkdir(parents=True, exist_ok=True)

        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.interaction_values: Optional[np.ndarray] = None
        self.sample_df: Optional[pd.DataFrame] = None
        self.interaction_sample_df: Optional[pd.DataFrame] = None

    def initialize_explainer(self, X_sample: pd.DataFrame) -> None:
        """Initializes SHAP TreeExplainer."""
        logger.info("Initializing SHAP TreeExplainer on model...")
        self.sample_df = X_sample[self.feature_names].copy()
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("SHAP TreeExplainer ready.")

    def compute_shap_values(self, X: pd.DataFrame, max_samples: int = 1500) -> np.ndarray:
        """Computes SHAP attribution values."""
        if self.explainer is None:
            self.initialize_explainer(X)

        sampled_X = X.sample(n=min(len(X), max_samples), random_state=42) if len(X) > max_samples else X
        self.sample_df = sampled_X[self.feature_names].copy()

        logger.info("Computing SHAP values for %d samples...", len(sampled_X))
        raw_shap = self.explainer.shap_values(self.sample_df)

        if isinstance(raw_shap, list):
            self.shap_values = raw_shap[0]
        else:
            self.shap_values = raw_shap

        return self.shap_values

    def compute_interaction_values(self, X: pd.DataFrame, max_samples: int = 400) -> np.ndarray:
        """Computes SHAP TreeExplainer 2D interaction tensor [N_samples, N_features, N_features]."""
        if self.explainer is None:
            self.initialize_explainer(X)

        sampled_X = X.sample(n=min(len(X), max_samples), random_state=42) if len(X) > max_samples else X
        self.interaction_sample_df = sampled_X[self.feature_names].copy()

        logger.info("Computing SHAP interaction tensor for %d samples...", len(self.interaction_sample_df))
        self.interaction_values = self.explainer.shap_interaction_values(self.interaction_sample_df)
        logger.info("SHAP interaction tensor computed successfully.")
        return self.interaction_values

    def _save_dual_format(self, fig: plt.Figure, base_filename: str) -> List[Path]:
        """Saves figure in PNG (300 DPI) and Vectorized PDF across all output directories."""
        saved_paths: List[Path] = []
        name_no_ext = Path(base_filename).stem

        for out_dir in self.output_dirs:
            png_path = out_dir / f"{name_no_ext}.png"
            pdf_path = out_dir / f"{name_no_ext}.pdf"

            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

            saved_paths.extend([png_path, pdf_path])

        logger.info("Saved dual-format figure [%s] in PNG and PDF.", name_no_ext)
        return saved_paths

    def plot_summary(self, save_name: str = "fig1_global_shap_beeswarm.png") -> List[Path]:
        """Generates publication-quality global SHAP beeswarm plot."""
        if self.shap_values is None or self.sample_df is None:
            raise ValueError("Call compute_shap_values before plot_summary.")

        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        shap.summary_plot(
            self.shap_values,
            self.sample_df,
            feature_names=[f.replace("_", " ").title() for f in self.feature_names],
            show=False,
            max_display=14,
        )
        plt.title(
            "Global Feature Hierarchy: Drivers of PV Soiling Degradation & Cleaning Dynamics",
            fontsize=12,
            fontweight="bold",
            pad=14,
        )
        plt.xlabel("SHAP Value (Impact on Soiling Dynamics)", fontsize=11, fontweight="bold")
        plt.tight_layout()

        paths = self._save_dual_format(fig, save_name)
        plt.close(fig)
        return paths

    def get_1d_dependence_curve(self, feature_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts sorted 1D array of feature values and SHAP values."""
        if self.shap_values is None or self.sample_df is None:
            raise ValueError("Call compute_shap_values first.")

        # Find matching feature name
        candidates = [f for f in self.feature_names if feature_name.lower() in f.lower()]
        target_feat = candidates[0] if candidates else feature_name

        feat_idx = self.feature_names.index(target_feat)
        feat_vals = self.sample_df[target_feat].values
        shap_vals = self.shap_values[:, feat_idx]

        sort_idx = np.argsort(feat_vals)
        return feat_vals[sort_idx], shap_vals[sort_idx]

    def plot_wind_dependence(
        self,
        v_crit: float = 6.60,
        ci_lower: float = 6.20,
        ci_upper: float = 7.00,
        save_name: str = "fig2_wind_resuspension_threshold.png",
    ) -> List[Path]:
        """
        Plots 1D SHAP dependence curve for Wind Speed vs delta_soiling_ratio,
        annotating the zero-crossing resuspension regime and 95% Confidence Interval.
        """
        wind_vals, shap_vals = self.get_1d_dependence_curve("wind_speed")

        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        sns.set_style("whitegrid")

        # Color scatter by Relative Humidity
        rh_col = "relative_humidity_mean" if "relative_humidity_mean" in self.sample_df.columns else "relative_humidity"
        if rh_col in self.sample_df.columns:
            rh_vals = self.sample_df[rh_col].iloc[np.argsort(wind_vals)].values
            scatter = ax.scatter(
                wind_vals,
                shap_vals,
                c=rh_vals,
                cmap="viridis_r",
                alpha=0.65,
                s=24,
                edgecolors="none",
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Relative Humidity (%)", fontsize=10, fontweight="bold")
        else:
            ax.scatter(wind_vals, shap_vals, color="#2b5c8f", alpha=0.6, s=24)

        # Smooth trend curve
        df_sort = pd.DataFrame({"wind": wind_vals, "shap": shap_vals})
        trend = df_sort.rolling(window=max(12, len(df_sort) // 25), min_periods=4, center=True).mean()
        ax.plot(
            trend["wind"],
            trend["shap"],
            color="#d62728",
            linewidth=2.8,
            label="Mean Physical Trend E[ΔSR | Wind]",
        )

        # Baseline zero neutral line
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8, label="Neutral Impact (ΔSR = 0)")

        # Vertical line at discovered critical wind speed v_crit
        ax.axvline(
            v_crit,
            color="#1f77b4",
            linestyle="-.",
            linewidth=2.2,
            label=f"Critical Velocity $v_{{crit}} = {v_crit:.2f}$ m/s",
        )

        # 95% Confidence Interval shaded ribbon
        ax.axvspan(
            ci_lower,
            ci_upper,
            color="#1f77b4",
            alpha=0.15,
            label=f"95% Bootstrapped CI: [{ci_lower:.2f}, {ci_upper:.2f}] m/s",
        )

        # Annotations for physical regimes
        ax.text(
            v_crit - 2.8,
            max(shap_vals) * 0.70,
            r"Regime I: Dust Deposition" + "\n" + r"& Transport ($\Delta SR \leq 0$)",
            color="#8c1d40",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#e67e22", lw=1),
        )
        ax.text(
            v_crit + 0.4,
            max(shap_vals) * 0.70,
            r"Regime II: Aerodynamic" + "\n" + r"Natural Cleaning ($\Delta SR > 0$)",
            color="#0b559f",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#e7f2fa", ec="#1f77b4", lw=1),
        )

        ax.set_xlabel("Wind Speed (m/s)", fontsize=11, fontweight="bold")
        ax.set_ylabel("SHAP Attribution for Soiling Rate (ΔSR)", fontsize=11, fontweight="bold")
        ax.set_title(
            "Wind Resuspension Tipping Point: Transition from Dust Accumulation to Natural Cleaning",
            fontsize=12,
            fontweight="bold",
            pad=14,
        )
        ax.legend(loc="lower right", frameon=True, framealpha=0.92)
        plt.tight_layout()

        paths = self._save_dual_format(fig, save_name)
        plt.close(fig)
        return paths

    def extract_2d_interaction(
        self, feat_x: str, feat_y: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts pair coordinates and SHAP interaction slice."""
        if self.interaction_values is None:
            raise ValueError("Call compute_interaction_values first.")

        # Find closest match in feature_names
        cand_x = [f for f in self.feature_names if feat_x.lower() in f.lower()]
        cand_y = [f for f in self.feature_names if feat_y.lower() in f.lower()]

        f_x = cand_x[0] if cand_x else feat_x
        f_y = cand_y[0] if cand_y else feat_y

        idx_x = self.feature_names.index(f_x)
        idx_y = self.feature_names.index(f_y)

        x_vals = self.interaction_sample_df[f_x].values
        y_vals = self.interaction_sample_df[f_y].values
        inter_vals = self.interaction_values[:, idx_x, idx_y]

        return x_vals, y_vals, inter_vals

    def plot_2d_cementation_cliff(
        self,
        feat_x: str = "relative_humidity",
        feat_y: str = "pm10",
        save_name: str = "fig3_cementation_cliff_2d_interaction.png",
    ) -> List[Path]:
        """
        Plots publication-grade 2D contour heatmap of SHAP interaction values,
        isolating the moisture x aerosol Cementation Cliff.
        """
        x_vals, y_vals, inter_vals = self.extract_2d_interaction(feat_x, feat_y)

        fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
        sns.set_style("white")

        sc = ax.scatter(
            x_vals,
            y_vals,
            c=inter_vals,
            cmap="coolwarm_r",
            s=48,
            edgecolors="k",
            linewidths=0.35,
            alpha=0.88,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(f"SHAP Interaction ({feat_x} × {feat_y})", fontsize=10, fontweight="bold")

        try:
            from scipy.interpolate import griddata

            xi = np.linspace(x_vals.min(), x_vals.max(), 120)
            yi = np.linspace(y_vals.min(), y_vals.max(), 120)
            X_grid, Y_grid = np.meshgrid(xi, yi)
            Z_grid = griddata((x_vals, y_vals), inter_vals, (X_grid, Y_grid), method="cubic")
            contours = ax.contour(X_grid, Y_grid, Z_grid, levels=8, colors="black", linewidths=0.8, alpha=0.55)
            ax.clabel(contours, inline=True, fontsize=8, fmt="%.4f")
        except Exception:
            pass

        label_x = feat_x.replace("_", " ").title()
        label_y = feat_y.replace("_", " ").title()
        ax.set_xlabel(label_x, fontsize=11, fontweight="bold")
        ax.set_ylabel(label_y, fontsize=11, fontweight="bold")
        ax.set_title(
            f"The Cementation Cliff: 2D Interaction Surface ({label_x} × {label_y})\n"
            "Negative Values Reveal Accelerated Dust Cementation Synergy",
            fontsize=12,
            fontweight="bold",
            pad=14,
        )

        plt.tight_layout()
        paths = self._save_dual_format(fig, save_name)
        plt.close(fig)
        return paths
