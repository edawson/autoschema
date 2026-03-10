import pathlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autoschema import read_parquet, write_parquet


def test_basic_io(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    path = str(tmp_path / "test.parquet")
    header = {"key": "value"}

    write_parquet(df, path, header=header)
    df_read, header_read = read_parquet(path)

    pd.testing.assert_frame_equal(
        df, df_read, check_dtype=False, check_categorical=False
    )
    assert header_read["key"] == "value"


def test_integer_downcasting(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame(
        {"small": [0, 255], "medium": [-32768, 32767], "large": [0, 4294967295]}
    )
    path = str(tmp_path / "downcast.parquet")
    write_parquet(df, path)

    table = pq.read_table(path)
    assert table.schema.field("small").type == pa.uint8()
    assert table.schema.field("medium").type == pa.int16()
    assert table.schema.field("large").type == pa.uint32()


def test_dictionary_encoding(tmp_path: pathlib.Path) -> None:
    # High cardinality - should not be encoded
    df_high = pd.DataFrame({"col": [str(i) for i in range(1000)]})
    # Low cardinality - should be encoded
    df_low = pd.DataFrame({"col": ["A", "B"] * 500})

    path_high = str(tmp_path / "high.parquet")
    path_low = str(tmp_path / "low.parquet")

    write_parquet(df_high, path_high)
    write_parquet(df_low, path_low)

    table_high = pq.read_table(path_high)
    table_low = pq.read_table(path_low)

    assert not pa.types.is_dictionary(table_high.schema.field("col").type)
    assert pa.types.is_dictionary(table_low.schema.field("col").type)


def test_float_downcasting(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame({"f": [1.1, 2.2]}, dtype="float64")
    path = str(tmp_path / "float.parquet")
    write_parquet(df, path)

    table = pq.read_table(path)
    assert table.schema.field("f").type == pa.float32()

def test_io_with_polars(tmp_path: pathlib.Path) -> None:
    try:
        import polars as pl
    except ImportError:
        pytest.skip("polars not installed")

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    path = str(tmp_path / "polars.parquet")
    write_parquet(df, path)

    df_read, _ = read_parquet(path)
    # read_parquet returns pandas by default
    assert isinstance(df_read, pd.DataFrame)
    assert len(df_read) == 3

def test_io_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported data type"):
        write_parquet([1, 2, 3], "test.parquet")

def test_io_no_pandas(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate pandas not being installed
    import autoschema.io
    monkeypatch.setattr(autoschema.io, "pd", None)

    df = pa.table({"a": [1, 2, 3]})
    path = str(tmp_path / "no_pandas.parquet")
    write_parquet(df, path)

    with pytest.raises(ImportError, match="pandas is required for read_parquet"):
        read_parquet(path)
