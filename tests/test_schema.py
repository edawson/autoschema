import pandas as pd
import pyarrow as pa
import pytest

from autoschema.io import to_arrow_table
from autoschema.schema import (
    cast_to_fixed_binary,
    infer_schema,
    map_to_vocabulary,
    strings_to_fixed_size_binary,
)


def test_map_to_vocabulary():
    df = pd.DataFrame({"kmer": ["AAAA", "CCCC", "GGGG", "TTTT", "Unknown"]})
    table = to_arrow_table(df)
    vocabulary = ["AAAA", "CCCC", "GGGG", "TTTT"]

    # Map to vocabulary
    mapped_table = map_to_vocabulary(table, "kmer", vocabulary)

    # Check type is dictionary
    field = mapped_table.schema.field("kmer")
    assert pa.types.is_dictionary(field.type)

    # Check index type is uint8 (since vocabulary size is 4)
    assert field.type.index_type == pa.uint8()

    # Check values
    # "Unknown" should be null
    result = mapped_table.column("kmer").to_pylist()
    assert result == ["AAAA", "CCCC", "GGGG", "TTTT", None]

def test_map_to_vocabulary_large():
    # Vocabulary size > 255 should use uint16
    vocabulary = [str(i) for i in range(300)]
    df = pd.DataFrame({"col": ["0", "299"]})
    table = to_arrow_table(df)

    mapped_table = map_to_vocabulary(table, "col", vocabulary)
    field = mapped_table.schema.field("col")
    assert field.type.index_type == pa.uint16()

def test_cast_to_fixed_binary():
    df = pd.DataFrame({"kmer": ["AAAA", "CCCC", "GGGG"]})
    table = to_arrow_table(df)

    # Cast to fixed binary
    cast_table = cast_to_fixed_binary(table, "kmer")

    # Check type
    field = cast_table.schema.field("kmer")
    assert pa.types.is_fixed_size_binary(field.type)
    assert field.type.byte_width == 4

def test_cast_to_fixed_binary_error():
    # Non-uniform length should raise ValueError
    df = pd.DataFrame({"kmer": ["AAAA", "CCC"]})
    table = to_arrow_table(df)

    with pytest.raises(ValueError, match="requires uniform length"):
        cast_to_fixed_binary(table, "kmer")

def test_strings_to_fixed_size_binary():
    df = pd.DataFrame({
        "kmer": ["AAAA", "CCCC"],
        "other": ["X", "Y"],
        "mixed": ["A", "BB"]
    })
    table = to_arrow_table(df)

    optimized_table = strings_to_fixed_size_binary(table)

    assert pa.types.is_fixed_size_binary(optimized_table.schema.field("kmer").type)
    assert pa.types.is_fixed_size_binary(optimized_table.schema.field("other").type)
    assert not pa.types.is_fixed_size_binary(optimized_table.schema.field("mixed").type)

def test_infer_schema_dictionary_downcast():
    # Test that infer_schema downcasts dictionary indices
    # Create a dictionary array with int32 indices but small dictionary
    indices = pa.array([0, 1], type=pa.int32())
    dictionary = pa.array(["A", "B"], type=pa.string())
    dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
    table = pa.table({"col": dict_array})

    schema = infer_schema(table)
    assert schema.field("col").type.index_type == pa.uint8()
