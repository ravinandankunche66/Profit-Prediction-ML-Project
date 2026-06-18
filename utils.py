from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

from dataset import DATASET_PATH


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.pkl"


def ensure_directories() -> None:
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def currency(value: float) -> str:
    return f"${value:,.0f}"


def compute_profit_thresholds(profits: np.ndarray) -> Dict[str, float]:
    low, high = np.quantile(profits, [0.33, 0.66])
    return {"low": float(low), "high": float(high)}


def classify_profit(value: float, thresholds: Dict[str, float]) -> str:
    if value >= thresholds["high"]:
        return "High Profit"
    if value >= thresholds["low"]:
        return "Moderate Profit"
    return "Low / Negative Profit"


def profit_color(value: float, thresholds: Dict[str, float]) -> str:
    if value >= thresholds["high"]:
        return "#1B8A5A"
    if value >= thresholds["low"]:
        return "#D9A404"
    return "#C0392B"


def save_artifact(artifact: Dict[str, Any], path: Path = MODEL_PATH) -> None:
    ensure_directories()
    joblib.dump(artifact, path)


def load_artifact(path: Path = MODEL_PATH) -> Dict[str, Any]:
    return joblib.load(path)


def dataset_ready(path: Path = DATASET_PATH) -> bool:
    return path.exists()


def model_ready(path: Path = MODEL_PATH) -> bool:
    return path.exists()


def ensure_dataset() -> None:
    ensure_directories()
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
