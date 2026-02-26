import pyarrow as pa
import pyarrow.compute as pc


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
