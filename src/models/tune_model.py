import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from configs.training_config import (
    EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    TIMESTAMP_COLUMN,
)
from src.evaluation.metrics import calculate_mae
from src.models.train_model import load_processed_data, split_data_by_time


def tune_model(model, param_grid, X_train, y_train):
    """Tune a model using GridSearchCV with time-series cross-validation."""
    time_series_cv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=time_series_cv,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def tune_decision_tree() -> None:
    """Tune DecisionTreeRegressor and track the best result with MLflow."""
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data_by_time(df)

    param_grid = {
        "max_depth": [5, 10, 20, None],
        "min_samples_leaf": [1, 5, 10],
        "min_samples_split": [2, 5, 10],
    }

    grid_search = tune_model(
        model=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
    )

    best_model = grid_search.best_estimator_

    train_predictions = best_model.predict(X_train)
    test_predictions = best_model.predict(X_test)

    train_mae = calculate_mae(y_train, train_predictions)
    test_mae = calculate_mae(y_test, test_predictions)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="tuned_decision_tree"):
        mlflow.log_param("model_type", type(best_model).__name__)
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("cv_strategy", "TimeSeriesSplit")
        mlflow.log_param("n_splits", 3)

        for param_name, param_value in grid_search.best_params_.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_metric("best_cv_mae", float(-grid_search.best_score_))
        mlflow.log_metric("train_mae", float(train_mae))
        mlflow.log_metric("test_mae", float(test_mae))
        mlflow.log_metric("prediction_min", float(test_predictions.min()))
        mlflow.log_metric("prediction_max", float(test_predictions.max()))
        mlflow.log_metric("prediction_mean", float(test_predictions.mean()))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_train.head(5),
        )

    print("Tuned Decision Tree completed.")
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.2f}")
    print(f"Train MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")


def main() -> None:
    """Run the model tuning pipeline."""
    tune_decision_tree()


if __name__ == "__main__":
    main()
