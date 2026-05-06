from pathlib import Path

import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError

from src.models import DatasetInfoChurn, DatasetRowChurn


DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "churn_dataset.csv"
REQUIRED_COLUMNS = list(DatasetRowChurn.model_fields.keys())
FEATURE_COLUMNS = [column for column in REQUIRED_COLUMNS if column != "churn"]


def load_churn_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    dataframe = pd.read_csv(DATASET_PATH)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset is missing required columns: {', '.join(missing_columns)}",
        )

    return dataframe[REQUIRED_COLUMNS]


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
    dataframe = load_churn_dataset()
    dataset_rows = dataframe_to_dataset_rows(dataframe.head(limit))
    return [row.model_dump() for row in dataset_rows]


def get_dataset_info() -> DatasetInfoChurn:
    dataframe = load_churn_dataset()
    dataset_rows = dataframe_to_dataset_rows(dataframe)
    records = [row.model_dump() for row in dataset_rows]
    validated_dataframe = pd.DataFrame(records, columns=REQUIRED_COLUMNS)
    churn_distribution = (
        {
            str(label): int(count)
            for label, count in validated_dataframe["churn"].value_counts().sort_index().items()
        }
        if not validated_dataframe.empty
        else {}
    )

    return DatasetInfoChurn(
        row_count=int(validated_dataframe.shape[0]),
        column_count=int(validated_dataframe.shape[1]),
        feature_names=FEATURE_COLUMNS,
        churn_distribution=churn_distribution,
    )
