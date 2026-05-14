from pathlib import Path
from typing import Any, Optional, Union
from datetime import datetime, timezone
import json

import joblib
import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.models import DatasetInfoChurn, DatasetRowChurn, DatasetSplitInfoChurn, FeatureSchemaItemChurn, FeatureVectorChurn, ModelMetricsResponseChurn, ModelSchemaChurn, ModelStatusChurn, PredictionResponseChurn, TrainingConfigChurn, TrainModelResponseChurn, TrainingHistoryEntryChurn


DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "churn_dataset.csv"
MODELS_PATH = Path(__file__).resolve().parent.parent / "models"
CHURN_MODEL_PATH = MODELS_PATH / "churn_model.joblib"
CHURN_MODEL_METADATA_PATH = MODELS_PATH / "churn_model_metadata.json"
CHURN_TRAINING_HISTORY_PATH = MODELS_PATH / "churn_training_history.json"
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
DEFAULT_MODEL_TYPE = "logreg"
MODEL_NAMES = {
    "logreg": "LogisticRegression",
    "random_forest": "RandomForestClassifier",
}
FEATURE_VALUE_TYPES = {
    "monthly_fee": "float",
    "usage_hours": "float",
    "support_requests": "int",
    "account_age_months": "int",
    "failed_payments": "int",
    "autopay_enabled": "int",
    "region": "str",
    "device_type": "str",
    "payment_method": "str",
}

trained_churn_model: Optional[Pipeline] = None
trained_churn_model_metadata: Optional[dict[str, object]] = None


def build_churn_http_exception(
    status_code: int,
    code: str,
    message: str,
    details: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
            "details": details,
        },
    )


def save_churn_model(model: Pipeline, metrics: TrainModelResponseChurn) -> None:
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(CHURN_MODEL_PATH),
        "model_type": metrics.model_type,
        "hyperparameters": metrics.hyperparameters,
        "metrics": metrics.model_dump(),
    }

    try:
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, CHURN_MODEL_PATH)
        CHURN_MODEL_METADATA_PATH.write_text(json.dumps(metadata), encoding="utf-8")
    except Exception as error:
        raise build_churn_http_exception(
            status_code=500,
            code="model_save_failed",
            message="Unable to save the trained churn model.",
            details={"reason": str(error)},
        ) from error

    global trained_churn_model_metadata
    trained_churn_model_metadata = metadata


def load_churn_training_history() -> list[TrainingHistoryEntryChurn]:
    if not CHURN_TRAINING_HISTORY_PATH.exists():
        return []

    try:
        raw_history = json.loads(CHURN_TRAINING_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception as error:
        raise build_churn_http_exception(
            status_code=500,
            code="training_history_load_failed",
            message="Unable to load churn training history.",
            details={"reason": str(error)},
        ) from error

    if not isinstance(raw_history, list):
        raise build_churn_http_exception(
            status_code=500,
            code="training_history_invalid",
            message="Churn training history file has an invalid format.",
        )

    history: list[TrainingHistoryEntryChurn] = []
    for entry in raw_history:
        try:
            history.append(TrainingHistoryEntryChurn.model_validate(entry))
        except ValidationError as error:
            raise build_churn_http_exception(
                status_code=500,
                code="training_history_invalid",
                message="Churn training history file contains an invalid record.",
                details={"errors": error.errors()},
            ) from error

    return history


def save_churn_training_history(history: list[TrainingHistoryEntryChurn]) -> None:
    try:
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        CHURN_TRAINING_HISTORY_PATH.write_text(
            json.dumps([entry.model_dump() for entry in history]),
            encoding="utf-8",
        )
    except Exception as error:
        raise build_churn_http_exception(
            status_code=500,
            code="training_history_save_failed",
            message="Unable to save churn training history.",
            details={"reason": str(error)},
        ) from error


def append_churn_training_history(entry: TrainingHistoryEntryChurn) -> None:
    history = load_churn_training_history()
    history.append(entry)
    save_churn_training_history(history)


def get_churn_model_metrics(limit: int = 5, model_type: Optional[str] = None) -> ModelMetricsResponseChurn:
    normalized_model_type = model_type.strip().lower() if model_type is not None else None
    history = load_churn_training_history()
    if normalized_model_type is not None:
        history = [entry for entry in history if entry.model_type == normalized_model_type]

    ordered_history = list(reversed(history))
    limited_history = ordered_history[:limit]

    return ModelMetricsResponseChurn(
        latest=limited_history[0] if limited_history else None,
        history=limited_history,
    )


def load_churn_model() -> Optional[Pipeline]:
    global trained_churn_model
    global trained_churn_model_metadata

    if not CHURN_MODEL_PATH.exists():
        trained_churn_model = None
        trained_churn_model_metadata = None
        return None

    try:
        loaded_model = joblib.load(CHURN_MODEL_PATH)
    except Exception:
        trained_churn_model = None
        trained_churn_model_metadata = None
        return None

    metadata: Optional[dict[str, object]] = None
    if CHURN_MODEL_METADATA_PATH.exists():
        try:
            metadata = json.loads(CHURN_MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            metadata = None

    trained_churn_model = loaded_model
    trained_churn_model_metadata = metadata or {"model_path": str(CHURN_MODEL_PATH)}
    return loaded_model


def initialize_churn_model_state() -> None:
    load_churn_model()


def get_churn_model_status() -> ModelStatusChurn:
    metrics = None
    trained_at = None
    model_path = None
    model_type = None
    hyperparameters = None

    if trained_churn_model_metadata is not None:
        trained_at_value = trained_churn_model_metadata.get("trained_at")
        model_path_value = trained_churn_model_metadata.get("model_path")
        model_type_value = trained_churn_model_metadata.get("model_type")
        hyperparameters_value = trained_churn_model_metadata.get("hyperparameters")
        metrics_value = trained_churn_model_metadata.get("metrics")

        trained_at = trained_at_value if isinstance(trained_at_value, str) else None
        model_path = model_path_value if isinstance(model_path_value, str) else None
        model_type = model_type_value if isinstance(model_type_value, str) else None
        hyperparameters = hyperparameters_value if isinstance(hyperparameters_value, dict) else None
        if isinstance(metrics_value, dict):
            try:
                metrics = TrainModelResponseChurn.model_validate(metrics_value)
            except ValidationError:
                metrics = None

        if model_type is None and metrics is not None:
            model_type = metrics.model_type
        if hyperparameters is None and metrics is not None:
            hyperparameters = metrics.hyperparameters

    return ModelStatusChurn(
        is_trained=trained_churn_model is not None,
        trained_at=trained_at,
        model_path=model_path,
        model_type=model_type,
        hyperparameters=hyperparameters,
        metrics=metrics,
    )


def get_active_churn_model() -> Pipeline:
    if trained_churn_model is not None:
        return trained_churn_model

    loaded_model = load_churn_model()
    if loaded_model is None:
        raise build_churn_http_exception(
            status_code=409,
            code="model_not_trained",
            message="Churn model is not trained. Train the model via POST /model/train first.",
        )

    return loaded_model


def get_churn_model_schema() -> ModelSchemaChurn:
    features: list[FeatureSchemaItemChurn] = []

    for feature_name in FEATURE_COLUMNS:
        feature_kind = "numeric" if feature_name in NUMERIC_FEATURE_COLUMNS else "categorical"
        features.append(
            FeatureSchemaItemChurn(
                name=feature_name,
                data_type=FEATURE_VALUE_TYPES[feature_name],
                feature_kind=feature_kind,
            )
        )

    return ModelSchemaChurn(
        features=features,
        numeric_features=NUMERIC_FEATURE_COLUMNS,
        categorical_features=CATEGORICAL_FEATURE_COLUMNS,
    )


def build_churn_feature_dataframe_from_records(records: list[dict[str, object]]) -> pd.DataFrame:
    feature_dataframe = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    if feature_dataframe.empty:
        return feature_dataframe

    feature_dataframe = feature_dataframe.copy()
    feature_dataframe.loc[:, NUMERIC_FEATURE_COLUMNS] = feature_dataframe[NUMERIC_FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    feature_dataframe.loc[:, CATEGORICAL_FEATURE_COLUMNS] = feature_dataframe[CATEGORICAL_FEATURE_COLUMNS].where(
        feature_dataframe[CATEGORICAL_FEATURE_COLUMNS].notna(),
        None,
    )
    return feature_dataframe


def build_churn_feature_dataframe(payloads: list[FeatureVectorChurn]) -> pd.DataFrame:
    records = [payload.model_dump() for payload in payloads]
    return build_churn_feature_dataframe_from_records(records)


def predict_churn(payloads: list[FeatureVectorChurn]) -> list[PredictionResponseChurn]:
    if not payloads:
        raise build_churn_http_exception(
            status_code=400,
            code="empty_prediction_request",
            message="Prediction request must contain at least one client.",
        )

    model = get_active_churn_model()
    feature_dataframe = build_churn_feature_dataframe(payloads)

    try:
        predicted_classes = model.predict(feature_dataframe)
        predicted_probabilities = model.predict_proba(feature_dataframe)
    except ValueError as error:
        raise build_churn_http_exception(
            status_code=500,
            code="prediction_failed",
            message="Unable to generate churn predictions.",
            details={"reason": str(error)},
        ) from error

    class_labels = list(model.classes_)
    churn_index = class_labels.index(1)
    non_churn_index = class_labels.index(0)

    responses: list[PredictionResponseChurn] = []
    for predicted_class, probabilities in zip(predicted_classes, predicted_probabilities):
        responses.append(
            PredictionResponseChurn(
                predicted_class=int(predicted_class),
                churn_probability=float(probabilities[churn_index]),
                non_churn_probability=float(probabilities[non_churn_index]),
            )
        )

    return responses


def load_churn_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise build_churn_http_exception(
            status_code=404,
            code="dataset_not_found",
            message="Dataset file not found.",
        )

    try:
        dataframe = pd.read_csv(DATASET_PATH)
    except pd.errors.EmptyDataError as error:
        raise build_churn_http_exception(
            status_code=400,
            code="dataset_file_empty",
            message="Dataset file is empty.",
        ) from error

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise build_churn_http_exception(
            status_code=500,
            code="dataset_schema_invalid",
            message="Dataset is missing required columns.",
            details={"missing_columns": missing_columns},
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
        raise build_churn_http_exception(
            status_code=400,
            code="dataset_empty",
            message="Dataset is empty.",
        )

    feature_dataframe = build_churn_feature_dataframe_from_records(
        dataframe[FEATURE_COLUMNS].to_dict(orient="records")
    )
    target_series = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    if target_series.isna().any():
        raise build_churn_http_exception(
            status_code=500,
            code="target_invalid",
            message="Target column churn contains missing or invalid values.",
        )
    target_series = target_series.astype(int)

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
            raise build_churn_http_exception(
                status_code=500,
                code="dataset_row_invalid",
                message="Dataset contains an invalid row.",
                details={"csv_line": index, "errors": error.errors()},
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
        raise build_churn_http_exception(
            status_code=500,
            code="dataset_split_failed",
            message="Unable to split the dataset.",
            details={"reason": str(error)},
        ) from error

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


def normalize_training_config(config: Optional[TrainingConfigChurn]) -> TrainingConfigChurn:
    if config is None:
        return TrainingConfigChurn()

    normalized_model_type = config.model_type.strip().lower()
    return TrainingConfigChurn(
        model_type=normalized_model_type,
        hyperparameters=dict(config.hyperparameters),
    )


def build_churn_estimator(config: TrainingConfigChurn):
    model_type = config.model_type
    hyperparameters = dict(config.hyperparameters)

    if model_type == "logreg":
        estimator_params: dict[str, Any] = {
            "max_iter": 1000,
            "random_state": DEFAULT_RANDOM_STATE,
            "solver": "liblinear",
        }
        estimator_params.update(hyperparameters)
        try:
            return LogisticRegression(**estimator_params)
        except (TypeError, ValueError) as error:
            raise build_churn_http_exception(
                status_code=400,
                code="invalid_hyperparameters",
                message="Invalid hyperparameters for logreg.",
                details={"model_type": "logreg", "reason": str(error)},
            ) from error

    if model_type == "random_forest":
        estimator_params = {
            "random_state": DEFAULT_RANDOM_STATE,
        }
        estimator_params.update(hyperparameters)
        try:
            return RandomForestClassifier(**estimator_params)
        except (TypeError, ValueError) as error:
            raise build_churn_http_exception(
                status_code=400,
                code="invalid_hyperparameters",
                message="Invalid hyperparameters for random_forest.",
                details={"model_type": "random_forest", "reason": str(error)},
            ) from error

    raise build_churn_http_exception(
        status_code=400,
        code="unsupported_model_type",
        message="Unsupported model_type. Supported values: logreg, random_forest.",
        details={"supported_values": list(MODEL_NAMES.keys())},
    )


def build_churn_model_pipeline(config: TrainingConfigChurn) -> Pipeline:
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
    estimator = build_churn_estimator(config)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def train_churn_model(
    feature_dataframe: pd.DataFrame,
    target_series: pd.Series,
    config: TrainingConfigChurn,
) -> Pipeline:
    if feature_dataframe.empty or target_series.empty:
        raise build_churn_http_exception(
            status_code=400,
            code="dataset_empty",
            message="Training dataset is empty.",
        )

    pipeline = build_churn_model_pipeline(config)
    try:
        pipeline.fit(feature_dataframe, target_series)
    except ValueError as error:
        raise build_churn_http_exception(
            status_code=400,
            code="training_failed",
            message="Unable to train the churn model.",
            details={"reason": str(error)},
        ) from error
    return pipeline


def run_churn_model_training(
    config: Optional[TrainingConfigChurn] = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratified: bool = True,
) -> TrainModelResponseChurn:
    global trained_churn_model

    normalized_config = normalize_training_config(config)
    feature_dataframe, target_series, numeric_columns, categorical_columns = prepare_churn_data()
    if len(feature_dataframe) < 2:
        raise build_churn_http_exception(
            status_code=400,
            code="dataset_too_small",
            message="Dataset does not contain enough rows for training.",
        )

    try:
        features_train, features_test, target_train, target_test = train_test_split(
            feature_dataframe,
            target_series,
            test_size=test_size,
            random_state=random_state,
            stratify=target_series if stratified else None,
        )
    except ValueError as error:
        raise build_churn_http_exception(
            status_code=400,
            code="dataset_split_failed",
            message="Unable to split the dataset for training.",
            details={"reason": str(error)},
        ) from error

    trained_churn_model = train_churn_model(features_train, target_train, normalized_config)

    try:
        predictions = trained_churn_model.predict(features_test)
    except ValueError as error:
        raise build_churn_http_exception(
            status_code=500,
            code="training_evaluation_failed",
            message="Unable to evaluate the trained churn model.",
            details={"reason": str(error)},
        ) from error

    roc_auc: Optional[float] = None
    try:
        prediction_probabilities = trained_churn_model.predict_proba(features_test)
        class_labels = list(trained_churn_model.classes_)
        churn_index = class_labels.index(1)
        roc_auc = float(roc_auc_score(target_test, prediction_probabilities[:, churn_index]))
    except (AttributeError, ValueError):
        roc_auc = None

    training_result = TrainModelResponseChurn(
        model_name=MODEL_NAMES[normalized_config.model_type],
        model_type=normalized_config.model_type,
        hyperparameters=normalized_config.hyperparameters,
        train_size=int(features_train.shape[0]),
        test_size=int(features_test.shape[0]),
        accuracy=float(accuracy_score(target_test, predictions)),
        f1=float(f1_score(target_test, predictions)),
        roc_auc=roc_auc,
        random_state=random_state,
        test_size_ratio=float(test_size),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    save_churn_model(trained_churn_model, training_result)
    append_churn_training_history(
        TrainingHistoryEntryChurn(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=training_result.model_name,
            model_type=training_result.model_type,
            hyperparameters=training_result.hyperparameters,
            accuracy=training_result.accuracy,
            f1=training_result.f1,
            roc_auc=training_result.roc_auc,
        )
    )
    return training_result
