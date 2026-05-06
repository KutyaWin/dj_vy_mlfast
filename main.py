from fastapi import FastAPI, Query

from src.models import DatasetInfoChurn, DatasetRowChurn, FeatureVectorChurn
from src.utils import get_dataset_info, get_dataset_preview

app = FastAPI()

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
