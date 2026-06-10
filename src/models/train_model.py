from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.evaluation.metrics import calculate_mae
from configs.training_config import (
    FEATURE_COLUMNS,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)


def load_processed_data(path: Path) -> pd.DataFrame:
    """Load the processed hourly taxi deman dataset."""
    return pd.read_parquet(path)


def time_based_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train and test sets."""
    df = df.sort_values(TIMESTAMP_COLUMN)

    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    return train_df, test_df


def train_linear_regression_baseline(train_df: pd.DataFrame) -> LinearRegression:
    """Train a Linear Regression baseline model"""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def main() -> None:
    df = load_processed_data(PROCESSED_DATA_PATH)

    train_df, test_df = time_based_train_test_split(df)

    print(test_df["pickups"].describe())

    model = train_linear_regression_baseline(train_df)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    predictions = model.predict(X_test)

    print("Predictions:")
    print(predictions[:10])

    print("Actual:")
    print(y_test.iloc[:10].values)

    print(f"Min Prediction: {predictions.min()}")
    print(f"Max Prediction: {predictions.max()}")
    print(f"Mean Prediction: {predictions.mean()}")

    mae = calculate_mae(y_test, predictions)

    print(f"Linear Regression baseline MAE: {mae:.2f}")


if __name__ == "__main__":
    main()
