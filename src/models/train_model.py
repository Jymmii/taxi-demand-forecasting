from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from configs.training_config import (
    FEATURE_COLUMNS,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    TEST_SIZE,
    EXPERIMENT_NAME,
)
from src.evaluation.metrics import calculate_mae


def load_processed_data(path: Path) -> pd.DataFrame:
    """Load the processed hourly taxi demand dataset."""
    return pd.read_parquet(path)


def split_data_by_time(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data chronologically to prevent future data leakage."""
    df = df.sort_values(TIMESTAMP_COLUMN)

    split_index = int(len(df) * (1 - TEST_SIZE))

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def run_experiment(
    model,
    model_type: str,
    run_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Train one model and track its experiment with MLflow."""
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae = calculate_mae(y_test, predictions)

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("split_type", "time_based")

        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("prediction_min", float(predictions.min()))
        mlflow.log_metric("prediction_max", float(predictions.max()))
        mlflow.log_metric("prediction_mean", float(predictions.mean()))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(5),
        )

        print(f"{model_type} completed. MAE: {mae:.2f}")


def train_model(data_path: Path) -> None:
    """Train baseline models and track experiments with MLflow."""
    df = load_processed_data(data_path)

    X_train, X_test, y_train, y_test = split_data_by_time(df)

    mlflow.set_experiment(EXPERIMENT_NAME)

    run_experiment(
        model=LinearRegression(),
        model_type="LinearRegression",
        run_name="linear_regression_baseline",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    run_experiment(
        model=DecisionTreeRegressor(random_state=42),
        model_type="DecisionTreeRegressor",
        run_name="decision_tree_baseline",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def main() -> None:
    """Run the model training pipeline."""
    train_model(PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
