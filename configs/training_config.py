from pathlib import Path

RAW_DATA_PATH = Path("data/raw/yellow_tripdata_2025-01.parquet")

PROCESSED_DATA_PATH = Path("data/processed/hourly_demand_2025-01.parquet")

FEATURE_COLUMNS = [
    "PULocationID",
    "hour",
    "day_of_week",
    "is_weekend",
]

TARGET_COLUMN = "pickups"

TIMESTAMP_COLUMN = "tpep_pickup_datetime"

TEST_SIZE = 0.2

EXPERIMENT_NAME = "nyc-taxi-demand-forecasting"
