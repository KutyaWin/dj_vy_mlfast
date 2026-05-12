from contextlib import asynccontextmanager
from typing import Annotated, Optional, Union

from fastapi import Body, FastAPI, Query

from src.models import DatasetInfoChurn, DatasetRowChurn, DatasetSplitInfoChurn, FeatureVectorChurn, ModelSchemaChurn, ModelStatusChurn, PredictionResponseChurn, TrainingConfigChurn, TrainModelResponseChurn
from src.utils import get_churn_model_schema, get_churn_model_status, get_dataset_info, get_dataset_preview, get_dataset_split_info, initialize_churn_model_state, predict_churn, run_churn_model_training


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_churn_model_state()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "ml churn service is running"}


@app.post(
    "/predict",
    response_model=Union[PredictionResponseChurn, list[PredictionResponseChurn]],
    responses={
        200: {
            "description": "Churn predictions for one or multiple clients",
            "content": {
                "application/json": {
                    "examples": {
                        "single_prediction": {
                            "summary": "Prediction for one client",
                            "value": {
                                "predicted_class": 0,
                                "churn_probability": 0.18,
                                "non_churn_probability": 0.82,
                            },
                        },
                        "batch_prediction": {
                            "summary": "Prediction for multiple clients",
                            "value": [
                                {
                                    "predicted_class": 0,
                                    "churn_probability": 0.18,
                                    "non_churn_probability": 0.82,
                                },
                                {
                                    "predicted_class": 1,
                                    "churn_probability": 0.71,
                                    "non_churn_probability": 0.29,
                                },
                            ],
                        },
                    }
                }
            },
        },
        409: {
            "description": "Churn model is not trained yet",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Churn model is not trained. Train the model via POST /model/train first."
                    }
                }
            },
        },
    },
)
def predict(
    payload: Union[FeatureVectorChurn, list[FeatureVectorChurn]] = Body(
        ...,
        openapi_examples={
            "single_client": {
                "summary": "Single client payload",
                "value": {
                    "monthly_fee": 39.99,
                    "usage_hours": 87.5,
                    "support_requests": 1,
                    "account_age_months": 24,
                    "failed_payments": 0,
                    "region": "North",
                    "device_type": "Mobile",
                    "payment_method": "Card",
                    "autopay_enabled": 1,
                },
            },
            "batch_clients": {
                "summary": "Batch payload",
                "value": [
                    {
                        "monthly_fee": 39.99,
                        "usage_hours": 87.5,
                        "support_requests": 1,
                        "account_age_months": 24,
                        "failed_payments": 0,
                        "region": "North",
                        "device_type": "Mobile",
                        "payment_method": "Card",
                        "autopay_enabled": 1,
                    },
                    {
                        "monthly_fee": 79.99,
                        "usage_hours": 12.0,
                        "support_requests": 6,
                        "account_age_months": 3,
                        "failed_payments": 2,
                        "region": "West",
                        "device_type": "Desktop",
                        "payment_method": "Bank Transfer",
                        "autopay_enabled": 0,
                    },
                ],
            },
        },
    )
) -> Union[PredictionResponseChurn, list[PredictionResponseChurn]]:
    payloads = payload if isinstance(payload, list) else [payload]
    predictions = predict_churn(payloads)
    return predictions if isinstance(payload, list) else predictions[0]


@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
def dataset_preview(limit: int = Query(default=5, ge=1, le=100)) -> list[DatasetRowChurn]:
    rows = get_dataset_preview(limit=limit)
    return [DatasetRowChurn.model_validate(row) for row in rows]


@app.get("/dataset/info", response_model=DatasetInfoChurn)
def dataset_info() -> DatasetInfoChurn:
    return get_dataset_info()


@app.get("/dataset/split-info", response_model=DatasetSplitInfoChurn)
def dataset_split_info() -> DatasetSplitInfoChurn:
    return get_dataset_split_info()


@app.post("/model/train", response_model=TrainModelResponseChurn)
def train_model(
    config: Annotated[
        Optional[TrainingConfigChurn],
        Body(
            openapi_examples={
                "default_logreg": {
                    "summary": "Default logistic regression training",
                    "value": {
                        "model_type": "logreg",
                        "hyperparameters": {},
                    },
                },
                "random_forest": {
                    "summary": "Random forest with custom hyperparameters",
                    "value": {
                        "model_type": "random_forest",
                        "hyperparameters": {
                            "n_estimators": 200,
                            "max_depth": 8,
                            "min_samples_split": 4,
                        },
                    },
                },
            },
        ),
    ] = None
) -> TrainModelResponseChurn:
    return run_churn_model_training(config=config)


@app.get("/model/status", response_model=ModelStatusChurn)
def model_status() -> ModelStatusChurn:
    return get_churn_model_status()


@app.get("/model/schema", response_model=ModelSchemaChurn)
def model_schema() -> ModelSchemaChurn:
    return get_churn_model_schema()
