from contextlib import asynccontextmanager

from fastapi import FastAPI, Query

from src.models import DatasetInfoChurn, DatasetRowChurn, DatasetSplitInfoChurn, FeatureVectorChurn, ModelStatusChurn, TrainModelResponseChurn
from src.utils import get_churn_model_status, get_dataset_info, get_dataset_preview, get_dataset_split_info, initialize_churn_model_state, run_churn_model_training


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_churn_model_state()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "ml churn service is running"}

@app.post("/predict", response_model=FeatureVectorChurn)
def predict(payload: FeatureVectorChurn) -> FeatureVectorChurn:
    return payload


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
def train_model() -> TrainModelResponseChurn:
    return run_churn_model_training()


@app.get("/model/status", response_model=ModelStatusChurn)
def model_status() -> ModelStatusChurn:
    return get_churn_model_status()
