from pydantic import BaseModel
from typing import Optional


class FeatureVectorChurn(BaseModel):
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


class TrainModelResponseChurn(BaseModel):
    model_name: str
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
    metrics: Optional[TrainModelResponseChurn]


class PredictionResponseChurn(BaseModel):
    predicted_class: int
    churn_probability: float
    non_churn_probability: float
