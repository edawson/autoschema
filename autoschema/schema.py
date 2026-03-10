import pyarrow as pa
import pyarrow.compute as pc


def map_column_to_fixed_vocabulary(
    table: "pa.Table", column_name: str, vocabulary: list[str]
) -> "pa.Table":
    """
    Maps a column to a fixed vocabulary to ensure stable integer IDs.

    This ensures that a given string always maps to the same ID (its index in the
    vocabulary) across different files, which is critical for joining sparse
    datasets like kmers.

    - IDs correspond to the index in the vocabulary list.
    - Values not present in the vocabulary will be mapped to null.
    - The underlying storage uses a DictionaryArray with the smallest possible 
      integer index type (uint8, uint16, or int32) to minimize memory usage.
    """
    if pa is None:
        raise ImportError("pyarrow is required for map_to_vocabulary.")

    col_idx = table.schema.get_field_index(column_name)
    if col_idx == -1:
        raise ValueError(f"Column '{column_name}' not found in table.")

    column = table.column(col_idx)

    # Create the vocabulary array
    vocab_array = pa.array(vocabulary, type=pa.string())

    # Determine the smallest possible index type based on vocabulary size
    dict_size = len(vocabulary)
    if dict_size <= 255:
        index_type = pa.uint8()
    elif dict_size <= 65535:
        index_type = pa.uint16()
    else:
        index_type = pa.int32()

    def encode_chunk(chunk):
        # Find the index of each value in the vocabulary
        indices = pc.index_in(chunk, value_set=vocab_array)
        # Cast indices to the optimized type
        indices = indices.cast(index_type)
        return pa.DictionaryArray.from_arrays(indices, vocab_array)

    new_chunks = [encode_chunk(chunk) for chunk in column.chunks]
    new_column = pa.chunked_array(new_chunks)

    return table.set_column(col_idx, column_name, new_column)


def cast_to_fixed_binary(table: "pa.Table", column_name: str) -> "pa.Table":
    """
    Casts a string/binary column to a FixedSizeBinary type.

    This is the most efficient storage format for fixed-length sequences like kmers,
    as it eliminates the 4-byte-per-row offset overhead required by standard strings.

    - All entries in the column must have the same length.
    - Raises ValueError if lengths are non-uniform or the column is empty.
    """
    if pa is None:
        raise ImportError("pyarrow is required for cast_to_fixed_binary.")

    col_idx = table.schema.get_field_index(column_name)
    if col_idx == -1:
        raise ValueError(f"Column '{column_name}' not found in table.")

    column = table.column(col_idx)

    # Validate uniform length
    lengths = pc.binary_length(column)
    min_len_scalar = pc.min(lengths)
    max_len_scalar = pc.max(lengths)

    min_len = min_len_scalar.as_py() if min_len_scalar.is_valid else None
    max_len = max_len_scalar.as_py() if max_len_scalar.is_valid else None

    if min_len is None:
        raise ValueError(f"Column '{column_name}' is empty or contains only nulls.")

    if min_len != max_len:
        raise ValueError(
            f"Column '{column_name}' requires uniform length for FixedSizeBinary "
            f"(found min={min_len}, max={max_len})."
        )

    new_type = pa.binary(min_len)
    new_column = column.cast(new_type)

    return table.set_column(col_idx, column_name, new_column)


def strings_to_fixed_size_binary(table: "pa.Table") -> "pa.Table":
    """
    Detects string/binary columns with uniform length and converts them to FixedSizeBinary.
    This is particularly efficient for kmers and other fixed-length sequences.
    """
    if pa is None:
        raise ImportError("pyarrow is required for optimize_fixed_size_binary.")

    new_fields = []
    new_columns = []

    for i in range(table.num_columns):
        column = table.column(i)
        field = table.schema.field(i)
        dtype = field.type

        if (
            pa.types.is_string(dtype)
            or pa.types.is_binary(dtype)
            or pa.types.is_large_string(dtype)
            or pa.types.is_large_binary(dtype)
        ):
            # Check for uniform length
            lengths = pc.binary_length(column)
            min_len_scalar = pc.min(lengths)
            max_len_scalar = pc.max(lengths)

            min_len = min_len_scalar.as_py() if min_len_scalar.is_valid else None
            max_len = max_len_scalar.as_py() if max_len_scalar.is_valid else None

            if min_len is not None and min_len == max_len and min_len > 0:
                new_type = pa.binary(min_len)
                new_columns.append(column.cast(new_type))
                new_fields.append(pa.field(field.name, new_type, nullable=field.nullable))
                continue

        new_columns.append(column)
        new_fields.append(field)

    return pa.Table.from_arrays(new_columns, schema=pa.schema(new_fields))


def infer_schema(table: "pa.Table") -> "pa.Schema":
    """
    Infers an optimized Arrow schema for the given table.
    - Downcasts integers to the smallest possible bit-width.
    - Downcasts floats if possible (e.g., float64 to float32).
    - Dictionary encodes strings if they have low cardinality.
    """
    fields = []

    for i in range(table.num_columns):
        column = table.column(i)
        name = table.schema.names[i]
        dtype = column.type

        new_type = dtype

        # Integer optimization
        if pa.types.is_integer(dtype):
            # Using pyarrow compute for min/max is more efficient than to_pandas
            min_val = pc.min(column).as_py()
            max_val = pc.max(column).as_py()

            if min_val is not None and max_val is not None:
                if min_val >= 0:
                    if max_val <= 255:
                        new_type = pa.uint8()
                    elif max_val <= 65535:
                        new_type = pa.uint16()
                    elif max_val <= 4294967295:
                        new_type = pa.uint32()
                else:
                    if min_val >= -128 and max_val <= 127:
                        new_type = pa.int8()
                    elif min_val >= -32768 and max_val <= 32767:
                        new_type = pa.int16()
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        new_type = pa.int32()

        # Dictionary optimization (Downcast indices)
        elif pa.types.is_dictionary(dtype):
            # Check the size of the dictionary to see if we can downcast indices
            # We look at the dictionary attached to the first chunk
            dict_size = len(column.chunk(0).dictionary) if column.num_chunks > 0 else 0

            if dict_size <= 255:
                index_type = pa.uint8()
            elif dict_size <= 65535:
                index_type = pa.uint16()
            else:
                index_type = pa.int32()

            if dtype.index_type != index_type:
                new_type = pa.dictionary(index_type, dtype.value_type)

        # Float optimization
        elif pa.types.is_floating(dtype):
            # For simplicity, we'll try to downcast to float32 if it's float64
            # and doesn't lose too much precision (naive check)
            if dtype == pa.float64():
                new_type = pa.float32()

        # String optimization (Dictionary encoding)
        elif (
            pa.types.is_string(dtype)
            or pa.types.is_binary(dtype)
            or pa.types.is_large_string(dtype)
            or pa.types.is_large_binary(dtype)
        ):
            # Heuristic: dictionary encode if unique values < 50% of total
            # and < 10,000 unique. For small datasets, unique_count / total_count
            # might be high, so we add a minimum total_count
            unique_count = len(column.unique())
            total_count = len(column)

            # Normalize dtype to non-large version for dictionary value type
            # if preferred, or just use the existing dtype.
            value_type = dtype
            if pa.types.is_large_string(dtype):
                value_type = pa.string()
            elif pa.types.is_large_binary(dtype):
                value_type = pa.binary()

            if unique_count < 10000 and (
                total_count < 100 or unique_count / total_count < 0.5
            ):
                new_type = pa.dictionary(pa.int32(), value_type)

        fields.append(pa.field(name, new_type, nullable=table.schema.field(i).nullable))

    return pa.schema(fields)
