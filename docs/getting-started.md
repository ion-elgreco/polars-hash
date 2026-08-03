# Getting Started

## Installation

```bash
pip install polars-hash
```

polars-hash supplies prebuilt wheels for Linux, macOS, and Windows, on x86, x86-64 and
aarch64. It requires `polars >= 1.36.1` and Python 3.10 or later. `plh.__version__`
gives the installed version.

Releases up to 0.7.0 also carried `linux-armv7` and `linux-ppc64le` wheels. Those
targets have no AES instructions, which the [GxHash
expressions](api-reference/non-cryptographic.md#gxhash32) need, so 0.8.0 dropped them.
Pip falls back to the source distribution on any platform without a wheel, and that
build needs a Rust toolchain and fails on those two targets.

## Your first hash

Import `polars_hash`, then select a namespace on a string expression:

```python
import polars as pl
import polars_hash as plh

df = pl.DataFrame({"foo": ["hello_world"]})

df.select(plh.col("foo").chash.sha2_256())
```

```text
┌──────────────────────────────────────────────────────────────────┐
│ foo                                                              │
│ ---                                                              │
│ str                                                              │
╞══════════════════════════════════════════════════════════════════╡
│ 35072c1ae546350e0bfa7ab11d49dc6f129e72ccd57ec7eb671225bbd197c8f1 │
└──────────────────────────────────────────────────────────────────┘
```

The output column has the same name as the input column. To keep both columns, give the
output a different name with `alias()`:

```python
df.with_columns(plh.col("foo").chash.sha2_256().alias("foo_sha256"))
```

## The six namespaces

The `import polars_hash` statement registers all six namespaces on `pl.Expr`:

| Namespace | Contents | Reference |
|-----------|----------|-----------|
| `chash` | Cryptographic hash functions: SHA-2, SHA-3, SHAKE128, BLAKE3, HMAC | [chash](api-reference/cryptographic.md) |
| `nchash` | Non-cryptographic hash functions: wyhash, xxHash, Murmur, FarmHash, CityHash, MD5, SHA-1 | [nchash](api-reference/non-cryptographic.md) |
| `geohash` | Geohash encode, decode, and neighbors | [geohash](api-reference/geohash.md) |
| `h3` | H3 hexagonal cell index | [h3](api-reference/h3.md) |
| `timehash` | Time-bucket encode, decode, and neighbors | [timehash](api-reference/timehash.md) |
| `uuidhash` | Deterministic UUID v5 | [uuidhash](api-reference/uuid.md) |

## `plh.col` and `pl.col` { #plh-col-vs-pl-col }

Both functions work at run time. `plh.col` is a typed wrapper around `pl.col` that
declares the namespaces. mypy and Pyright then accept `.chash.sha2_256()`. With
`pl.col`, they report an error:

```python
plh.col("foo").chash.sha2_256()
pl.col("foo").chash.sha2_256()  # type: ignore
```

`plh.concat_str` is the equivalent wrapper around `pl.concat_str`. Both wrappers return
`plh.HExpr`. This class is a subclass of `pl.Expr` that holds the five namespace
properties. Use it as the type annotation when you pass these expressions between
functions.

## Hash of more than one column

First concatenate the columns, then hash the result. Give a `separator` value. Without
a separator, `("ab", "c")` and `("a", "bc")` give the same hash:

```python
df = pl.DataFrame({"foo": ["hello_world"], "bar": ["today"]})

df.select(plh.concat_str("foo", "bar", separator="|").chash.sha2_256())
```

To make a key from two columns, you can also use
[`uuid5_concat`](api-reference/uuid.md#uuid5_concat). That expression concatenates the
two columns and controls what a null value becomes.

## Null values

Each expression gives null for a null input. It does not hash a substitute value. An
empty string is a value, and the expression hashes it:

```python
df = pl.DataFrame({"literal": ["hello_world", None, ""]})

df.select(plh.col("literal").nchash.murmur32())
```

```text
┌────────────┐
│ literal    │
│ ---        │
│ u32        │
╞════════════╡
│ 3531928679 │
│ null       │
│ 0          │
└────────────┘
```

The default value of `ignore_nulls` in `pl.concat_str` is `False`. One null input then
makes the concatenation null, and the hash is also null. To get a value instead, set
`ignore_nulls=True` or replace the null values first.

## Selection of a hash function

- **Digest size.** Select the `chash` expression with the output width that you need.
  `sha2_256()` and `blake3()` give 256 bits. `sha2_512()` gives 512 bits.
- **Speed.** `nchash.xxh3_64()` and `nchash.wyhash()` are the fastest expressions.
  `chash.blake3()` is the fastest expression in `chash`.
- **Keyed output.** Use `chash.hmac_sha256(key=...)`.
- **Compatibility with a different system.** Use the same algorithm and the same seed
  as that system.

## Seeds

Some `nchash` expressions accept a `seed` argument. Each seed gives a different hash,
and each of these hashes is stable. The default seed is `0`. The same seed always gives
the same output:

```python
df.select(plh.col("literal").nchash.xxhash64(seed=42))
```

## Binary input

Each hash expression accepts a `Binary` column and a Utf8 column. It hashes the bytes
of both:

```python
pl.select(pl.lit(b"my_bytes").nchash.wyhash())  # type: ignore
```

A digest reads bytes. Therefore the data type does not change it, and a Utf8 column
gives the same digest as a `Binary` column with the same UTF-8 bytes. Bytes that are
not correct UTF-8 have no Utf8 column, and this is why the `Binary` type is necessary.
A different data type causes an error. The `geohash`, `h3` and `timehash` expressions
are not digests. They accept the input types that their pages give.

## Geospatial indexes

The `geohash` and `h3` namespaces accept a struct with a `latitude` field and a
`longitude` field. Both fields must be float:

```python
df = pl.DataFrame(
    {"coord": [{"longitude": -120.6623, "latitude": 35.3003}]},
    schema={
        "coord": pl.Struct(
            [pl.Field("longitude", pl.Float64), pl.Field("latitude", pl.Float64)]
        ),
    },
)

df.with_columns(
    geohash=plh.col("coord").geohash.from_coords(5),
    h3=plh.col("coord").h3.from_coords(5),  # type: ignore
)
```

```text
┌─────────────────────┬─────────┬─────────────────┐
│ coord               ┆ geohash ┆ h3              │
│ ---                 ┆ ---     ┆ ---             │
│ struct[2]           ┆ str     ┆ str             │
╞═════════════════════╪═════════╪═════════════════╡
│ {-120.6623,35.3003} ┆ 9q60y   ┆ 8529adc7fffffff │
└─────────────────────┴─────────┴─────────────────┘
```

[`geohash.to_coords()`](api-reference/geohash.md#to_coords) decodes a geohash to
coordinates. [`geohash.neighbors()`](api-reference/geohash.md#neighbors) gives the eight
adjacent cells.

## Next steps

- [API Reference](api-reference/index.md). All expressions in one table.
- [chash](api-reference/cryptographic.md). The cryptographic hash functions.
- [nchash](api-reference/non-cryptographic.md). The non-cryptographic hash functions.
