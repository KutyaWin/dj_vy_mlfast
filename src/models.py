from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class DatasetInfoChurn(BaseModel):
    row_count: int
    column_count: int
    feature_names: list[str]
    target_name: str
    churn_distribution: dict[str, int]


class DatasetSplitInfoChurn(BaseModel):
    train_size: int
    test_size: int
    test_size_ratio: float
    random_state: int
    stratified: bool
    numeric_columns: list[str]
    categorical_columns: list[str]
    train_churn_distribution: dict[str, int]
    test_churn_distribution: dict[str, int]


class TrainingConfigChurn(BaseModel):
    model_type: str = "logreg"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class TrainModelResponseChurn(BaseModel):
    model_name: str
    model_type: str = "logreg"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    train_size: int
    test_size: int
    accuracy: float
    f1: float
    random_state: int
    test_size_ratio: float
    numeric_columns: list[str]
    categorical_columns: list[str]


class ModelStatusChurn(BaseModel):
    is_trained: bool
    trained_at: Optional[str]
    model_path: Optional[str]
    model_type: Optional[str]
    hyperparameters: Optional[dict[str, Any]]
    metrics: Optional[TrainModelResponseChurn]


class PredictionResponseChurn(BaseModel):
    predicted_class: int
    churn_probability: float
    non_churn_probability: float


class FeatureSchemaItemChurn(BaseModel):
    name: str
    data_type: str
    feature_kind: str


class ModelSchemaChurn(BaseModel):
    features: list[FeatureSchemaItemChurn]
    numeric_features: list[str]
    categorical_features: list[str]


class ErrorDetailChurn(BaseModel):
    field: Optional[str] = None
    issue: str
    input_value: Optional[Any] = None


class ErrorResponseChurn(BaseModel):
    code: str
    message: str
    details: Optional[Union[list[ErrorDetailChurn], dict[str, Any]]] = None
