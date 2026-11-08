"""
PV-Soiling-XAI: Physics-Informed Explainable AI for Solar PV Dust Tipping Points.

Modules:
    data_loader: Ingestion, schema normalization, and physics-driven sample simulation.
    feature_engineering: Thermodynamic and atmospheric indicators (dew point, deficit, PM exposure).
    train: Purged walk-forward CV, LightGBM/XGBoost model training.
    xai_engine: SHAP TreeExplainer attribution and 2D interaction extraction.
    threshold_miner: Change-point detection for wind resuspension (v_crit) & cementation boundaries.
    operational_engine: Real-time decision dispatch for smart-cleaning and cementation alert systems.
"""

__version__ = "1.0.0"
__author__ = "Applied ML & PV Physics Research Team"

from src.data_loader import PVDataLoader
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.xai_engine import XAIEngine
from src.threshold_miner import ThresholdMiner
from src.operational_engine import OperationalEngine

__all__ = [
    "PVDataLoader",
    "FeatureEngineer",
    "ModelTrainer",
    "XAIEngine",
    "ThresholdMiner",
    "OperationalEngine",
]
