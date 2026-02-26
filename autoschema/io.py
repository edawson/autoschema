from typing import Any, Optional
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None


def to_arrow_table(data: Any) -> "pa.Table":
    """Converts various dataframe types to a PyArrow Table."""
    if isinstance(data, pa.Table):
        return data

    # Check for pandas
    if "pandas" in str(type(data)):
        if pd is None:
            raise ImportError(
                "pandas is not installed but a pandas object was passed. "
                "Please install pandas to use this functionality."
            )
        if isinstance(data, pd.DataFrame):
            return pa.Table.from_pandas(data)

    # Check for polars
    if "polars" in str(type(data)):
        if pl is None:
            raise ImportError(
                "polars is not installed but a polars object was passed. "
                "Please install polars to use this functionality."
            )
        if isinstance(data, pl.DataFrame):
            return data.to_arrow()

    # Check for cudf or other objects with to_arrow
    if hasattr(data, "to_arrow"):
        return data.to_arrow()

    raise ValueError(f"Unsupported data type: {type(data)}")


def write_parquet(
    data: Any, path: str, header: Optional[dict[str, str]] = None, **kwargs: Any
) -> None:
    """
    Writes data to a Parquet file with an optimized schema and custom header.
    """
    from .schema import infer_schema

    table = to_arrow_table(data)
    optimized_schema = infer_schema(table)

    # Cast table to optimized schema
    table = table.cast(optimized_schema)

    # Add header metadata
    if header:
        existing_metadata = optimized_schema.metadata or {}
        # Parquet metadata keys and values must be bytes
        new_metadata = {**existing_metadata}
        for k, v in header.items():
            new_metadata[k.encode("utf-8")] = str(v).encode("utf-8")

        table = table.replace_schema_metadata(new_metadata)

    pq.write_table(table, path, **kwargs)


def read_parquet(path: str) -> tuple["pd.DataFrame", dict[str, str]]:
    """
    Reads a Parquet file and returns the data and the custom header metadata.
    """
    if pd is None:
        raise ImportError(
            "pandas is not installed. pandas is required for read_parquet "
            "to return a DataFrame."
        )

    table = pq.read_table(path)
