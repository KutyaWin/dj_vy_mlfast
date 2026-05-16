import json

import pytest
from fastapi import HTTPException

import src.utils as utils
from src.models import FeatureVectorChurn, TrainingConfigChurn


def build_sample_payload() -> dict[str, object]:
    return {
        "monthly_fee": 39.99,
        "usage_hours": 64.5,
        "support_requests": 1,
        "account_age_months": 18,
        "failed_payments": 0,
        "region": "North",
        "device_type": "Mobile",
        "payment_method": "Card",
        "autopay_enabled": 1,
    }


def test_build_churn_feature_dataframe_from_records_preserves_expected_columns(isolated_churn_environment):
    records = [build_sample_payload()]

    feature_dataframe = utils.build_churn_feature_dataframe_from_records(records)

    assert list(feature_dataframe.columns) == utils.FEATURE_COLUMNS
    assert feature_dataframe.shape == (1, len(utils.FEATURE_COLUMNS))
    assert feature_dataframe.loc[0, "monthly_fee"] == pytest.approx(39.99)
    assert feature_dataframe.loc[0, "region"] == "North"


def test_prepare_churn_data_returns_expected_feature_groups(isolated_churn_environment):
    feature_dataframe, target_series, numeric_columns, categorical_columns = utils.prepare_churn_data()

    assert list(feature_dataframe.columns) == utils.FEATURE_COLUMNS
    assert len(feature_dataframe) == len(target_series)
    assert numeric_columns == utils.NUMERIC_FEATURE_COLUMNS
    assert categorical_columns == utils.CATEGORICAL_FEATURE_COLUMNS
    assert set(target_series.unique()) == {0, 1}


def test_run_churn_model_training_saves_model_and_history(isolated_churn_environment):
    training_result = utils.run_churn_model_training(
        config=TrainingConfigChurn(model_type="logreg", hyperparameters={})
    )

    history = utils.load_churn_training_history()

    assert training_result.model_type == "logreg"
    assert training_result.accuracy >= 0.0
    assert training_result.f1 >= 0.0
    assert isolated_churn_environment["model_path"].exists()
    assert isolated_churn_environment["metadata_path"].exists()
    assert isolated_churn_environment["history_path"].exists()
    assert len(history) == 1
    assert history[0].model_type == "logreg"
    assert history[0].accuracy == pytest.approx(training_result.accuracy)


def test_get_churn_model_metrics_filters_by_model_type(isolated_churn_environment):
    utils.run_churn_model_training(config=TrainingConfigChurn(model_type="logreg", hyperparameters={}))
    utils.run_churn_model_training(
        config=TrainingConfigChurn(
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 3},
        )
    )

    metrics = utils.get_churn_model_metrics(limit=5, model_type="random_forest")

    assert metrics.latest is not None
    assert metrics.latest.model_type == "random_forest"
    assert len(metrics.history) == 1
    assert metrics.history[0].hyperparameters == {"n_estimators": 10, "max_depth": 3}


def test_predict_churn_without_trained_model_raises_structured_http_error(isolated_churn_environment):
    payload = [FeatureVectorChurn.model_validate(build_sample_payload())]

    with pytest.raises(HTTPException) as error_info:
        utils.predict_churn(payload)

    assert error_info.value.status_code == 409
    assert error_info.value.detail["code"] == "model_not_trained"
    assert "Train the model" in error_info.value.detail["message"]


def test_load_churn_training_history_with_invalid_json_raises_structured_error(isolated_churn_environment):
    isolated_churn_environment["history_path"].parent.mkdir(parents=True, exist_ok=True)
    isolated_churn_environment["history_path"].write_text("{broken", encoding="utf-8")

    with pytest.raises(HTTPException) as error_info:
        utils.load_churn_training_history()

    assert error_info.value.status_code == 500
    assert error_info.value.detail["code"] == "training_history_load_failed"


def test_api_train_status_predict_and_metrics_flow(client, isolated_churn_environment):
    dataset_info_response = client.get("/dataset/info")
    assert dataset_info_response.status_code == 200
    dataset_info = dataset_info_response.json()
    assert dataset_info["row_count"] == 24
    assert dataset_info["target_name"] == utils.TARGET_COLUMN

    train_response = client.post(
        "/model/train",
        json={"model_type": "logreg", "hyperparameters": {}},
    )
    assert train_response.status_code == 200
    train_payload = train_response.json()
    assert train_payload["model_type"] == "logreg"
    assert "roc_auc" in train_payload

    status_response = client.get("/model/status")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["is_trained"] is True
    assert status_payload["model_type"] == "logreg"
    assert status_payload["metrics"]["accuracy"] == pytest.approx(train_payload["accuracy"])

    predict_response = client.post("/predict", json=build_sample_payload())
    assert predict_response.status_code == 200
    prediction_payload = predict_response.json()
    assert prediction_payload["predicted_class"] in {0, 1}
    assert 0.0 <= prediction_payload["churn_probability"] <= 1.0
    assert 0.0 <= prediction_payload["non_churn_probability"] <= 1.0

    metrics_response = client.get("/model/metrics", params={"limit": 3})
    assert metrics_response.status_code == 200
    metrics_payload = metrics_response.json()
    assert metrics_payload["latest"]["model_type"] == "logreg"
    assert len(metrics_payload["history"]) == 1

    history_on_disk = json.loads(isolated_churn_environment["history_path"].read_text(encoding="utf-8"))
    assert len(history_on_disk) == 1
    assert history_on_disk[0]["model_type"] == "logreg"


def test_api_metrics_supports_limit_and_model_type_filter(client):
    client.post("/model/train", json={"model_type": "logreg", "hyperparameters": {}})
    client.post(
        "/model/train",
        json={
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 12, "max_depth": 4},
        },
    )

    filtered_response = client.get(
        "/model/metrics",
        params={"limit": 1, "model_type": "random_forest"},
    )

    assert filtered_response.status_code == 200
    filtered_payload = filtered_response.json()
    assert filtered_payload["latest"]["model_type"] == "random_forest"
    assert len(filtered_payload["history"]) == 1
    assert filtered_payload["history"][0]["hyperparameters"] == {"n_estimators": 12, "max_depth": 4}


def test_api_predict_without_trained_model_returns_structured_error(client):
    response = client.post("/predict", json=build_sample_payload())

    assert response.status_code == 409
    payload = response.json()
    assert payload["code"] == "model_not_trained"
    assert payload["details"] is None


def test_api_predict_validation_error_returns_structured_details(client):
    invalid_payload = build_sample_payload()
    invalid_payload["usage_hours"] = "invalid-number"

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "invalid_feature_type"
    assert payload["details"][0]["field"] == "usage_hours"


def test_api_train_with_unsupported_model_type_returns_structured_error(client):
    response = client.post(
        "/model/train",
        json={"model_type": "svm", "hyperparameters": {}},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == "unsupported_model_type"
    assert payload["details"]["supported_values"] == ["logreg", "random_forest"]
