import great_expectations as gx
import pandas as pd

from configs.training_config import RAW_DATA_PATH


def get_raw_data_batch(df: pd.DataFrame):
    """Create a Great Expectations batch from a Pandas DataFrame."""

    context = gx.get_context()

    data_source = context.data_sources.add_pandas(name="taxi_raw_pandas")

    data_asset = data_source.add_dataframe_asset(name="yellow_tripdata_raw")

    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name="yellow_tripdata_raw_batch"
    )

    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    return batch


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate raw NYC taxi trip data before feature engineering."""

    batch = get_raw_data_batch(df)

    expectations = [
        gx.expectations.ExpectColumnToExist(column="tpep_pickup_datetime"),
        gx.expectations.ExpectColumnToExist(column="PULocationID"),
        gx.expectations.ExpectColumnToExist(column="DOLocationID"),
        gx.expectations.ExpectColumnToExist(column="trip_distance"),
        gx.expectations.ExpectColumnToExist(column="passenger_count"),
        gx.expectations.ExpectColumnToExist(column="total_amount"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="tpep_pickup_datetime"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="PULocationID"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="DOLocationID"),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="trip_distance",
            min_value=0,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="passenger_count",
            min_value=0,
        ),
    ]

    failures = []

    for expectation in expectations:
        result = batch.validate(expectation)

        if not result.success:
            failures.append(result)

    if failures:
        print("\nRaw data validation failed:\n")

        for failure in failures:
            expectation_type = failure.expectation_config.type
            column = failure.expectation_config.kwargs.get("column")
            unexpected_count = failure.result.get("unexpected_count")

            print(
                f"- {expectation_type} | "
                f"column={column} | "
                f"unexpected_count={unexpected_count}"
            )

        raise ValueError("Raw data validation failed. Stopping pipeline.")

    print("Raw data validation passed.")


def main() -> None:
    df = pd.read_parquet(RAW_DATA_PATH)
    validate_raw_data(df)


if __name__ == "__main__":
    main()
