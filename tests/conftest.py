from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
import src.utils as utils


@pytest.fixture()
def synthetic_churn_dataframe() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(24):
        rows.append(
            {
                "monthly_fee": 25.0 + index,
                "usage_hours": 8.0 + (index % 6) * 3.5,
                "support_requests": index % 4,
                "account_age_months": 6 + index,
                "failed_payments": 1 if index % 5 == 0 else 0,
                "autopay_enabled": 0 if index % 4 == 0 else 1,
                "region": ["North", "South", "East"][index % 3],
                "device_type": ["Mobile", "Desktop"][index % 2],
                "payment_method": ["Card", "Bank Transfer", "PayPal"][index % 3],
                "churn": 1 if index % 4 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def isolated_churn_environment(tmp_path: Path, synthetic_churn_dataframe: pd.DataFrame, monkeypatch: pytest.MonkeyPatch):
    dataset_path = tmp_path / "churn_dataset.csv"
    models_path = tmp_path / "models"
    model_path = models_path / "churn_model.joblib"
    metadata_path = models_path / "churn_model_metadata.json"
    history_path = models_path / "churn_training_history.json"

    synthetic_churn_dataframe.to_csv(dataset_path, index=False)

    monkeypatch.setattr(utils, "DATASET_PATH", dataset_path)
    monkeypatch.setattr(utils, "MODELS_PATH", models_path)
    monkeypatch.setattr(utils, "CHURN_MODEL_PATH", model_path)
    monkeypatch.setattr(utils, "CHURN_MODEL_METADATA_PATH", metadata_path)
    monkeypatch.setattr(utils, "CHURN_TRAINING_HISTORY_PATH", history_path)
    monkeypatch.setattr(utils, "trained_churn_model", None)
    monkeypatch.setattr(utils, "trained_churn_model_metadata", None)

    yield {
        "dataset_path": dataset_path,
        "models_path": models_path,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "history_path": history_path,
    }

    utils.trained_churn_model = None
    utils.trained_churn_model_metadata = None


@pytest.fixture()
def client(isolated_churn_environment):
    with TestClient(main.app) as test_client:
        yield test_client
