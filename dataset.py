from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "business_profit.csv"

NUMERIC_COLUMNS: List[str] = [
    "R&D Spend",
    "Administration Cost",
    "Marketing Spend",
    "Logistics Cost",
    "Employee Count",
    "Year",
]
CATEGORICAL_COLUMNS: List[str] = [
    "Country",
    "Business Category",
    "Company Name",
]
TARGET_COLUMN = "Profit"


@dataclass
class DatasetArtifacts:
    dataframe: pd.DataFrame
    company_encoder: LabelEncoder
    country_encoder: LabelEncoder
    category_encoder: LabelEncoder


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]

    required_columns = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(cleaned.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].fillna("Unknown").astype(str).str.strip()

    cleaned["Year"] = cleaned["Year"].round().astype(int)
    cleaned["Employee Count"] = cleaned["Employee Count"].round().astype(int)
    return cleaned[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]].copy()


def add_encoded_columns(df: pd.DataFrame) -> DatasetArtifacts:
    encoded = df.copy()

    company_encoder = LabelEncoder()
    country_encoder = LabelEncoder()
    category_encoder = LabelEncoder()

    encoded["Company Name Encoded"] = company_encoder.fit_transform(encoded["Company Name"])
    encoded["Country Encoded"] = country_encoder.fit_transform(encoded["Country"])
    encoded["Business Category Encoded"] = category_encoder.fit_transform(encoded["Business Category"])

    return DatasetArtifacts(
        dataframe=encoded,
        company_encoder=company_encoder,
        country_encoder=country_encoder,
        category_encoder=category_encoder,
    )


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    dataset_path = path or DATASET_PATH
    return pd.read_csv(dataset_path)


def load_and_prepare_dataset(path: Path | None = None) -> DatasetArtifacts:
    raw_df = load_dataset(path=path)
    cleaned_df = clean_dataset(raw_df)
    return add_encoded_columns(cleaned_df)


def get_model_feature_columns() -> Tuple[List[str], List[str]]:
    return NUMERIC_COLUMNS.copy(), CATEGORICAL_COLUMNS.copy()
