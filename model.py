from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset import (
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    get_model_feature_columns,
    load_and_prepare_dataset,
)
from utils import MODEL_PATH, compute_profit_thresholds, save_artifact

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - depends on environment setup
    XGBRegressor = None


def build_preprocessor() -> ColumnTransformer:
    numeric_columns, categorical_columns = get_model_feature_columns()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    one_hot_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical_one_hot", one_hot_pipeline, categorical_columns),
        ]
    )


def build_models(random_state: int = 42) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=random_state),
    }
    if XGBRegressor is not None:
        models["XGBoost Regressor"] = XGBRegressor(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=1,
            objective="reg:squarederror",
        )
    return models


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    results: List[Dict[str, float | str]] = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, estimator in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        results.append({"Model": name, "R2 Score": r2, "MAE": mae, "RMSE": rmse})
        fitted_pipelines[name] = pipeline

    metrics_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    return metrics_df, fitted_pipelines


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    feature_names: List[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue

        if hasattr(transformer, "named_steps"):
            if "encoder" in transformer.named_steps:
                encoder = transformer.named_steps["encoder"]
                if hasattr(encoder, "get_feature_names_out"):
                    names = list(encoder.get_feature_names_out(columns))
                else:
                    names = list(columns)
            else:
                names = list(columns)
        elif hasattr(transformer, "get_feature_names_out"):
            names = list(transformer.get_feature_names_out(columns))
        else:
            names = list(columns)

        feature_names.extend([str(feature_name) for feature_name in names])

    return feature_names


def extract_feature_importance(best_pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]
    feature_names = get_transformed_feature_names(preprocessor)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        importances = np.abs(coefficients).ravel()
    else:
        importances = np.zeros(len(feature_names), dtype=float)

    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    return importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)


def correlation_frame(df: pd.DataFrame) -> pd.DataFrame:
    corr_columns = NUMERIC_COLUMNS + ["Country Encoded", "Business Category Encoded", "Company Name Encoded", TARGET_COLUMN]
    return df[corr_columns].corr()


def train_and_save_model(
    dataset_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
) -> Dict[str, Any]:
    dataset_artifacts = load_and_prepare_dataset(path=dataset_path)
    dataframe = dataset_artifacts.dataframe

    X = dataframe[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    metrics_df, fitted_pipelines = evaluate_models(X_train, X_test, y_train, y_test)

    best_model_name = str(metrics_df.iloc[0]["Model"])
    best_pipeline = fitted_pipelines[best_model_name]
    predictions = best_pipeline.predict(X_test)

    thresholds = compute_profit_thresholds(dataframe[TARGET_COLUMN].to_numpy())
    feature_importance_df = extract_feature_importance(best_pipeline)
    artifact: Dict[str, Any] = {
        "model": best_pipeline,
        "best_model_name": best_model_name,
        "sklearn_version": sklearn.__version__,
        "metrics": metrics_df,
        "dataset": dataframe,
        "feature_importance": feature_importance_df,
        "correlation": correlation_frame(dataframe),
        "thresholds": thresholds,
        "x_test": X_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "predictions": pd.Series(predictions).reset_index(drop=True),
        "company_classes": list(dataset_artifacts.company_encoder.classes_),
        "country_classes": list(dataset_artifacts.country_encoder.classes_),
        "category_classes": list(dataset_artifacts.category_encoder.classes_),
    }
    save_artifact(artifact, path=model_path)
    return artifact
