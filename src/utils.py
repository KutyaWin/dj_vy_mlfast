from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models import DatasetInfoChurn, DatasetRowChurn, DatasetSplitInfoChurn, TrainModelResponseChurn


DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "churn_dataset.csv"
NUMERIC_FEATURE_COLUMNS = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]
CATEGORICAL_FEATURE_COLUMNS = ["region", "device_type", "payment_method"]
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
TARGET_COLUMN = "churn"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
MODEL_NAME = "LogisticRegression"

trained_churn_model: Optional[Pipeline] = None


def load_churn_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    try:
        dataframe = pd.read_csv(DATASET_PATH)
    except pd.errors.EmptyDataError as error:
        raise HTTPException(status_code=400, detail="Dataset file is empty") from error

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset is missing required columns: {', '.join(missing_columns)}",
        )

    return dataframe[REQUIRED_COLUMNS]


def get_validated_churn_dataframe() -> pd.DataFrame:
    dataframe = load_churn_dataset()
    dataset_rows = dataframe_to_dataset_rows(dataframe)
    records = [row.model_dump() for row in dataset_rows]
    return pd.DataFrame(records, columns=REQUIRED_COLUMNS)


def prepare_churn_data() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    dataframe = load_churn_dataset().copy()
    if dataframe.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    feature_dataframe = dataframe[FEATURE_COLUMNS].copy()
    target_series = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    if target_series.isna().any():
        raise HTTPException(status_code=500, detail="Target column churn contains missing or invalid values")
    target_series = target_series.astype(int)

    feature_dataframe.loc[:, NUMERIC_FEATURE_COLUMNS] = feature_dataframe[NUMERIC_FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    numeric_fill_values = feature_dataframe[NUMERIC_FEATURE_COLUMNS].median()
    feature_dataframe.loc[:, NUMERIC_FEATURE_COLUMNS] = feature_dataframe[NUMERIC_FEATURE_COLUMNS].fillna(
        numeric_fill_values,
    )

    categorical_fill_values = {}
    for column in CATEGORICAL_FEATURE_COLUMNS:
        mode = feature_dataframe[column].mode(dropna=True)
        categorical_fill_values[column] = mode.iloc[0] if not mode.empty else ""
    feature_dataframe.loc[:, CATEGORICAL_FEATURE_COLUMNS] = feature_dataframe[
        CATEGORICAL_FEATURE_COLUMNS
    ].fillna(categorical_fill_values)

    return feature_dataframe, target_series, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS


def get_class_distribution(target_series: pd.Series) -> dict[str, int]:
    if target_series.empty:
        return {}

    return {
        str(label): int(count)
        for label, count in target_series.value_counts().sort_index().items()
    }


def dataframe_to_dataset_rows(dataframe: pd.DataFrame) -> list[DatasetRowChurn]:
    dataset_rows: list[DatasetRowChurn] = []
    records = dataframe.to_dict(orient="records")

    for index, record in enumerate(records, start=2):
        try:
            dataset_rows.append(DatasetRowChurn.model_validate(record))
        except ValidationError as error:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid dataset row at CSV line {index}: {error.errors()}",
            ) from error

    return dataset_rows


def get_dataset_preview(limit: int = 5) -> list[dict[str, object]]:
    dataframe = get_validated_churn_dataframe()
    dataset_rows = dataframe_to_dataset_rows(dataframe.head(limit))
    return [row.model_dump() for row in dataset_rows]


def get_dataset_info() -> DatasetInfoChurn:
    validated_dataframe = get_validated_churn_dataframe()
    churn_distribution = get_class_distribution(validated_dataframe[TARGET_COLUMN])

    return DatasetInfoChurn(
        row_count=int(validated_dataframe.shape[0]),
        column_count=int(validated_dataframe.shape[1]),
        feature_names=FEATURE_COLUMNS,
        target_name=TARGET_COLUMN,
        churn_distribution=churn_distribution,
    )

def get_dataset_split_info(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratified: bool = True,
) -> DatasetSplitInfoChurn:
    feature_dataframe, target_series, numeric_columns, categorical_columns = prepare_churn_data()

    try:
        _, _, target_train, target_test = train_test_split(
            feature_dataframe,
            target_series,
            test_size=test_size,
            random_state=random_state,
            stratify=target_series if stratified else None,
        )
    except ValueError as error:
        raise HTTPException(status_code=500, detail=f"Unable to split dataset: {error}") from error

    return DatasetSplitInfoChurn(
        train_size=int(target_train.shape[0]),
        test_size=int(target_test.shape[0]),
        test_size_ratio=float(test_size),
        random_state=random_state,
        stratified=stratified,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        train_churn_distribution=get_class_distribution(target_train),
        test_churn_distribution=get_class_distribution(target_test),
    )


def build_churn_model_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURE_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURE_COLUMNS),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE, solver="liblinear")),
        ]
    )


def train_churn_model(feature_dataframe: pd.DataFrame, target_series: pd.Series) -> Pipeline:
    if feature_dataframe.empty or target_series.empty:
        raise HTTPException(status_code=400, detail="Training dataset is empty")

    pipeline = build_churn_model_pipeline()
    pipeline.fit(feature_dataframe, target_series)
    return pipeline


def run_churn_model_training(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratified: bool = True,
) -> TrainModelResponseChurn:
    global trained_churn_model

    feature_dataframe, target_series, numeric_columns, categorical_columns = prepare_churn_data()
    if len(feature_dataframe) < 2:
        raise HTTPException(status_code=400, detail="Dataset does not contain enough rows for training")

    try:
        features_train, features_test, target_train, target_test = train_test_split(
            feature_dataframe,
            target_series,
            test_size=test_size,
            random_state=random_state,
            stratify=target_series if stratified else None,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"Unable to split dataset for training: {error}") from error

    trained_churn_model = train_churn_model(features_train, target_train)

    try:
        predictions = trained_churn_model.predict(features_test)
    except ValueError as error:
        raise HTTPException(status_code=500, detail=f"Unable to generate predictions: {error}") from error

    return TrainModelResponseChurn(
        model_name=MODEL_NAME,
        train_size=int(features_train.shape[0]),
        test_size=int(features_test.shape[0]),
        accuracy=float(accuracy_score(target_test, predictions)),
        f1=float(f1_score(target_test, predictions)),
        random_state=random_state,
        test_size_ratio=float(test_size),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
