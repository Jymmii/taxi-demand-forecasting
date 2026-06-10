from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("data/raw/yellow_tripdata_2025-01.parquet")
PROCESSED_DATA_PATH = Path("data/processed/hourly_demand_2025-01.parquet")


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw taxi trip data from a parquet file"""
    return pd.read_parquet(path)


def create_hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw taxi trips into hourly pickup counts by pickup zone."""
    hourly_demand = (
        df.groupby(
            [
                pd.Grouper(key="tpep_pickup_datetime", freq="h"),
                "PULocationID",
            ]
        )
        .size()
        .to_frame("pickups")
        .reset_index()
    )

    return hourly_demand


def add_time_features(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features to the hourly demand dataset"""
    hourly_demand = hourly_demand.copy()

    hourly_demand["hour"] = hourly_demand["tpep_pickup_datetime"].dt.hour
    hourly_demand["day_of_week"] = hourly_demand["tpep_pickup_datetime"].dt.dayofweek
    hourly_demand["month"] = hourly_demand["tpep_pickup_datetime"].dt.month
    hourly_demand["is_weekend"] = hourly_demand["day_of_week"].isin([5, 6])

    return hourly_demand


def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    """Save processed feature dataset to a parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    raw_df = load_raw_data(RAW_DATA_PATH)
    hourly_demand = create_hourly_demand(raw_df)
    features = add_time_features(hourly_demand)
    save_processed_data(features, PROCESSED_DATA_PATH)

    print(f"Saved processed data to {PROCESSED_DATA_PATH}")
    print(f"Shape: {features.shape}")


if __name__ == "__main__":
    main()
