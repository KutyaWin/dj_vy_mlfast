from contextlib import asynccontextmanager
import logging
from typing import Annotated, Optional, Union

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.models import DatasetInfoChurn, DatasetRowChurn, DatasetSplitInfoChurn, ErrorDetailChurn, ErrorResponseChurn, FeatureVectorChurn, HealthResponseChurn, ModelMetricsResponseChurn, ModelSchemaChurn, ModelStatusChurn, PredictionResponseChurn, TrainingConfigChurn, TrainModelResponseChurn
from src.utils import get_churn_health_status, get_churn_model_metrics, get_churn_model_schema, get_churn_model_status, get_dataset_info, get_dataset_preview, get_dataset_split_info, initialize_churn_model_state, predict_churn, run_churn_model_training


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("churn_service")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting churn FastAPI lifespan")
    initialize_churn_model_state()
    yield


app = FastAPI(lifespan=lifespan)


def build_error_response(
    code: str,
    message: str,
    details: Optional[Union[list[ErrorDetailChurn], dict[str, object]]] = None,
) -> dict[str, object]:
    return ErrorResponseChurn(code=code, message=message, details=details).model_dump()


def normalize_http_exception(exc: HTTPException) -> ErrorResponseChurn:
    if isinstance(exc.detail, dict):
        code = exc.detail.get("code")
        message = exc.detail.get("message")
        details = exc.detail.get("details")
        if isinstance(code, str) and isinstance(message, str):
            return ErrorResponseChurn(code=code, message=message, details=details)

    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return ErrorResponseChurn(code="http_error", message=message, details=None)


def build_validation_error_response(exc: RequestValidationError) -> ErrorResponseChurn:
    details: list[ErrorDetailChurn] = []
    error_types = set()

    for error in exc.errors():
        error_type = error.get("type", "validation_error")
        error_types.add(error_type)
        raw_location = [str(part) for part in error.get("loc", []) if part != "body"]
        if raw_location and raw_location[0].endswith("Churn"):
            raw_location = raw_location[1:]
        field = ".".join(raw_location) if raw_location else None
        details.append(
            ErrorDetailChurn(
                field=field,
                issue=error.get("msg", "Invalid request value."),
                input_value=error.get("input"),
            )
        )

    if any(error_type in {"missing", "extra_forbidden"} for error_type in error_types):
        return ErrorResponseChurn(
            code="invalid_feature_count",
            message="Request body contains missing or unexpected fields.",
            details=details,
        )

    if any(
        error_type.startswith("float")
        or error_type.startswith("int")
        or error_type.startswith("string")
        or error_type.startswith("list")
        or error_type.startswith("dict")
        for error_type in error_types
    ):
        return ErrorResponseChurn(
            code="invalid_feature_type",
            message="Request body contains values of invalid types.",
            details=details,
        )

    return ErrorResponseChurn(
        code="validation_error",
        message="Request validation failed.",
        details=details,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    logger.warning("Handled HTTP exception", extra={"status_code": exc.status_code, "detail": exc.detail})
    response = normalize_http_exception(exc)
    return JSONResponse(status_code=exc.status_code, content=response.model_dump())


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Handled request validation error", extra={"errors": exc.errors()})
    response = build_validation_error_response(exc)
    return JSONResponse(status_code=422, content=response.model_dump())


@app.exception_handler(Exception)
async def unexpected_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled internal server error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content=build_error_response(
            code="internal_server_error",
            message="Internal server error.",
            details=None,
        ),
    )


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "ml churn service is running"}


@app.post(
    "/predict",
    response_model=Union[PredictionResponseChurn, list[PredictionResponseChurn]],
    responses={
        400: {
            "model": ErrorResponseChurn,
            "description": "Prediction request is semantically invalid",
            "content": {
                "application/json": {
                    "examples": {
                        "empty_prediction_request": {
                            "summary": "Empty batch",
                            "value": {
                                "code": "empty_prediction_request",
                                "message": "Prediction request must contain at least one client.",
                                "details": None,
                            },
                        }
                    }
                }
            },
        },
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
        422: {
            "model": ErrorResponseChurn,
            "description": "Request body validation failed",
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_feature_type": {
                            "summary": "Wrong value type",
                            "value": {
                                "code": "invalid_feature_type",
                                "message": "Request body contains values of invalid types.",
                                "details": [
                                    {
                                        "field": "usage_hours",
                                        "issue": "Input should be a valid number, unable to parse string as a number",
                                        "input_value": "abc",
                                    }
                                ],
                            },
                        },
                        "invalid_feature_count": {
                            "summary": "Missing or extra fields",
                            "value": {
                                "code": "invalid_feature_count",
                                "message": "Request body contains missing or unexpected fields.",
                                "details": [
                                    {
                                        "field": "payment_method",
                                        "issue": "Field required",
                                        "input_value": None,
                                    }
                                ],
                            },
                        },
                    }
                }
            },
        },
        409: {
            "model": ErrorResponseChurn,
            "description": "Churn model is not trained yet",
            "content": {
                "application/json": {
                    "example": {
                        "code": "model_not_trained",
                        "message": "Churn model is not trained. Train the model via POST /model/train first.",
                        "details": None,
                    }
                }
            },
        },
        500: {
            "model": ErrorResponseChurn,
            "description": "Prediction failed due to internal model or preprocessing error",
            "content": {
                "application/json": {
                    "example": {
                        "code": "prediction_failed",
                        "message": "Unable to generate churn predictions.",
                        "details": {
                            "reason": "Model prediction failed for the provided data."
                        },
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
    logger.info("Received /predict request", extra={"batch_size": len(payloads)})
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


@app.post(
    "/model/train",
    response_model=TrainModelResponseChurn,
    responses={
        400: {
            "model": ErrorResponseChurn,
            "description": "Training configuration or dataset is invalid",
            "content": {
                "application/json": {
                    "examples": {
                        "unsupported_model_type": {
                            "summary": "Unsupported model type",
                            "value": {
                                "code": "unsupported_model_type",
                                "message": "Unsupported model_type. Supported values: logreg, random_forest.",
                                "details": {
                                    "supported_values": ["logreg", "random_forest"]
                                },
                            },
                        },
                        "invalid_hyperparameters": {
                            "summary": "Invalid hyperparameters",
                            "value": {
                                "code": "invalid_hyperparameters",
                                "message": "Invalid hyperparameters for random_forest.",
                                "details": {
                                    "model_type": "random_forest",
                                    "reason": "max_depth must be greater than 0"
                                },
                            },
                        },
                        "dataset_empty": {
                            "summary": "Empty dataset",
                            "value": {
                                "code": "dataset_empty",
                                "message": "Dataset is empty.",
                                "details": None,
                            },
                        },
                    }
                }
            },
        },
        404: {
            "model": ErrorResponseChurn,
            "description": "Training dataset file was not found",
            "content": {
                "application/json": {
                    "example": {
                        "code": "dataset_not_found",
                        "message": "Dataset file not found.",
                        "details": None,
                    }
                }
            },
        },
        422: {
            "model": ErrorResponseChurn,
            "description": "Training request body validation failed",
            "content": {
                "application/json": {
                    "example": {
                        "code": "invalid_feature_type",
                        "message": "Request body contains values of invalid types.",
                        "details": [
                            {
                                "field": "hyperparameters",
                                "issue": "Input should be a valid dictionary",
                                "input_value": ["not", "a", "dict"],
                            }
                        ],
                    }
                }
            },
        },
        500: {
            "model": ErrorResponseChurn,
            "description": "Unexpected failure during training or model persistence",
            "content": {
                "application/json": {
                    "examples": {
                        "training_failed": {
                            "summary": "Estimator training failed",
                            "value": {
                                "code": "training_failed",
                                "message": "Unable to train the churn model.",
                                "details": {
                                    "reason": "Training data could not be processed by the estimator."
                                },
                            },
                        },
                        "model_save_failed": {
                            "summary": "Saving model failed",
                            "value": {
                                "code": "model_save_failed",
                                "message": "Unable to save the trained churn model.",
                                "details": {
                                    "reason": "Model artifact could not be written to disk."
                                },
                            },
                        },
                    }
                }
            },
        },
    },
)
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
    logger.info("Received /model/train request")
    return run_churn_model_training(config=config)


@app.get("/model/status", response_model=ModelStatusChurn)
def model_status() -> ModelStatusChurn:
    return get_churn_model_status()


@app.get("/health", response_model=HealthResponseChurn)
def health() -> JSONResponse:
    health_status = get_churn_health_status()
    status_code = 200 if health_status.status == "ok" else 503
    return JSONResponse(status_code=status_code, content=health_status.model_dump())


@app.get("/model/metrics", response_model=ModelMetricsResponseChurn)
def model_metrics(
    limit: Annotated[int, Query(ge=1, le=100)] = 5,
    model_type: Annotated[Optional[str], Query()] = None,
) -> ModelMetricsResponseChurn:
    return get_churn_model_metrics(limit=limit, model_type=model_type)


@app.get("/model/schema", response_model=ModelSchemaChurn)
def model_schema() -> ModelSchemaChurn:
    return get_churn_model_schema()
