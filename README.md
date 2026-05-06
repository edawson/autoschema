# (ARCHIVED) AutoSchema


**NOTE: THIS REPOSITORY HAS BEEN ARCHIVED DUE TO A NAME ISSUE IN PYPI. DEVELOPMENT CONTINUES AT https://github.com/edawson/autoparquet**

AutoSchema is a Python package that wraps Parquet/Arrow to automatically generate optimized schemas for your data. It focuses on better compression through automatic bit-packing, int-packing, and dictionary encoding, while providing a convenient "header" system for storing custom metadata.

## AI-Generated Code with Human Review Note

```text
This repository was generated using Gemini 3 Flash, based on a specification written by the author. The code was reviewed by the author for correctness and tested locally.

Signed: Eric T. Dawson
```

## Features

- **Automatic Schema Inference**: Automatically detects the smallest possible integer and float types to save space.
- **Optimized Compression**: Uses bit-packing and dictionary encoding where appropriate.
- **Custom Headers**: Easily add and retrieve custom metadata (versioning, key-value pairs) in Parquet files.
- **Multi-Framework Support**: Works with Pandas, Polars, and cuDF.

## Installation

```bash
git clone https://github.com/edawson/autoschema
cd autoschema
pip install .
```


Coming soon: Pypi

```bash
pip install autoschema
```

## Usage

### Basic Example

```python
import pandas as pd
import autoschema

# Create a dataframe with mixed types
df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "category": ["A", "B", "A", "B", "A"],
    "value": [1.1, 2.2, 3.3, 4.4, 5.5]
})

# Write with automatic schema optimization and a custom header
autoschema.write_parquet(
    df, 
    "data.parquet", 
    header={"version": "1.0", "author": "Eric T. Dawson"}
)

# Read back the data and the header
df_read, header = autoschema.read_parquet("data.parquet")
print(header)  # {'version': '1.0', 'author': 'Eric T. Dawson'}
```

### Automatic Schema Optimization

AutoSchema analyzes your data to find the most efficient storage format:

- **Integers**: Downcasts to the smallest possible bit-width (`uint8`, `int16`, `uint32`, etc.) based on the actual range of values in your dataset.
- **Floats**: Automatically converts `float64` to `float32` to save 50% space when extreme precision isn't required.
- **Strings/Binary**: Uses a heuristic to apply **Dictionary Encoding** to columns with low cardinality, significantly reducing file size for repetitive text data.

### Custom Headers (Metadata)

AutoSchema makes it easy to attach "headers" (key-value metadata) to your files. This is perfect for:
- Data versioning
- Tracking data lineage (source, timestamp, author)
- Storing processing parameters

```python
header = {
    "schema_version": "2.4.1",
    "captured_at": "2026-02-25",
    "environment": "production"
}
autoschema.write_parquet(df, "prod_data.parquet", header=header)
```

### Kmer Counting & Stable IDs

For bioinformatics and large-scale kmer analysis, AutoSchema provides tools to ensure **stable IDs** across different files and **optimized storage** for fixed-length sequences.

#### 1. Fixed-Size Binary Optimization
If your kmers have a uniform length, you can manually cast a specific column to `FixedSizeBinary`. This removes the 4-byte-per-row offset overhead of standard strings, saving significant space and improving scan speed in R and Python.

```python
from autoschema import cast_to_fixed_binary

table = autoschema.to_arrow_table(df)
table = cast_to_fixed_binary(table, "kmer")
```

#### 2. Stable IDs with Global Vocabularies
When working with sparse kmer data across multiple files, you can use `map_to_vocabulary` to ensure a kmer always maps to the same integer ID.

```python
import pandas as pd
import autoschema
import itertools
from autoschema import map_to_vocabulary

# 1. Generate a stable vocabulary of all possible 4-mers (256 total)
# This ensures "AAAA" is always ID 0, "AAAC" is always ID 1, etc.
vocabulary = ["".join(p) for p in itertools.product("ACGT", repeat=4)]

# 2. Your kmer count data (sparse)
df = pd.DataFrame({
    "kmer": ["AAAC", "AAGT", "TTTT"],
    "count": [10, 5, 100]
})

# 3. Convert to Arrow and map to the stable 4-mer vocabulary
# AutoSchema will automatically use uint8 (1 byte) for the IDs since 256 <= 256
table = autoschema.to_arrow_table(df)
table = map_to_vocabulary(table, "kmer", vocabulary)

# 4. Store the vocabulary in the metadata for future reference
header = {
    "kmer_k": 4,
    "vocabulary": ",".join(vocabulary)
}

autoschema.write_parquet(table, "kmers_4mer_stable.parquet", header=header)
```

### Multi-Framework Support

AutoSchema is designed to be a drop-in wrapper for the most popular Python data frameworks.

#### Polars
```python
import polars as pl
import autoschema

lf = pl.DataFrame({"a": [1, 2, 3], "b": ["low", "low", "high"]})
autoschema.write_parquet(lf, "polars_data.parquet")
```

#### cuDF (NVIDIA GPU Dataframes)
```python
import cudf
import autoschema

gdf = cudf.DataFrame({"a": [1, 2, 3]})
autoschema.write_parquet(gdf, "gpu_data.parquet")
```

### Advanced Writing

You can pass any standard `pyarrow.parquet.write_table` arguments through `write_parquet`:

```python
autoschema.write_parquet(
    df, 
    "compressed.parquet", 
    compression="zstd", 
    compression_level=10,
    row_group_size=100_000
)
```
