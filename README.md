This plugin provides stable hashing functionality across different polars versions.

📖 **[Documentation](https://ion-elgreco.github.io/polars-hash/)** — every expression,
its input and output types, and its arguments.

## Examples
### Cryptographic Hashers

```python
import polars as pl
import polars_hash as plh

df = pl.DataFrame({
    "foo":["hello_world"]
})

result = df.select(plh.col('foo').chash.sha2_256())

print(result)

┌──────────────────────────────────────────────────────────────────┐
│ foo                                                              │
│ ---                                                              │
│ str                                                              │
╞══════════════════════════════════════════════════════════════════╡
│ 35072c1ae546350e0bfa7ab11d49dc6f129e72ccd57ec7eb671225bbd197c8f1 │
└──────────────────────────────────────────────────────────────────┘
```

### Non-cryptographic Hashers
```python
df = pl.DataFrame({
    "foo":["hello_world"]
})

result = df.select(plh.col('foo').nchash.wyhash())
print(result)
┌──────────────────────┐
│ foo                  │
│ ---                  │
│ u64                  │
╞══════════════════════╡
│ 16737367591072095403 │
└──────────────────────┘

result = df.select(plh.col('foo').nchash.farmhash64())
print(result)
┌──────────────────────┐
│ foo                  │
│ ---                  │
│ u64                  │
╞══════════════════════╡
│ 15605398435621216523 │
└──────────────────────┘

result = df.select(plh.col('foo').nchash.farmhash32())
print(result)
┌────────────┐
│ foo        │
│ ---        │
│ u32        │
╞════════════╡
│ 1719156559 │
└────────────┘

result = df.select(plh.col('foo').nchash.cityhash128())
print(result)
┌─────────────────────────────────────────┐
│ foo                                     │
│ ---                                     │
│ u128                                    │
╞═════════════════════════════════════════╡
│ 133423608296839006301901834072762183026 │
└─────────────────────────────────────────┘

result = df.select(plh.col('foo').nchash.gxhash64())
print(result)
┌─────────────────────┐
│ foo                 │
│ ---                 │
│ u64                 │
╞═════════════════════╡
│ 2180020304351407825 │
└─────────────────────┘
```

`cityhash32()` and `cityhash64()` return the values printed above for `farmhash32()`
and `farmhash64()`. That is expected: FarmHash reuses CityHash for short input, and
`hello_world` is 11 bytes. See
[the CityHash reference](https://ion-elgreco.github.io/polars-hash/latest/api-reference/non-cryptographic/#polars_hash.NonCryptographicHashingNameSpace.cityhash32).

The GxHash expressions need a CPU with AES instructions and have no software fallback.
Every x86, x86-64 and aarch64 wheel is built for them; there are no `linux-armv7` or
`linux-ppc64le` wheels from 0.8.0 on, because GxHash cannot be built for either. See
[the GxHash reference](https://ion-elgreco.github.io/polars-hash/latest/api-reference/non-cryptographic/#polars_hash.NonCryptographicHashingNameSpace.gxhash32).

### Byte Encoding

`bytes` writes a value as its own bytes, so that a hasher reads the value itself and
not a string form of it. Each type keeps its own width: `Int32` makes 4 bytes and
`Float64` makes 8. `Utf8` and `Binary` have no byte order of their own and pass
through unchanged.

```python
df = pl.DataFrame({"literal": [1]}, schema={"literal": pl.Int32})

result = df.select(
    plh.col('literal').bytes.to_le().alias('le'),
    plh.col('literal').bytes.to_be().alias('be'),
)
print(result)
┌─────────────────────┬─────────────────────┐
│ le                  ┆ be                  │
│ ---                 ┆ ---                 │
│ binary              ┆ binary              │
╞═════════════════════╪═════════════════════╡
│ b"\x01\x00\x00\x00" ┆ b"\x00\x00\x00\x01" │
└─────────────────────┴─────────────────────┘
```

Send the result to a hasher for a hash of the value and not of its text:

```python
plh.col("id").cast(pl.Int64).bytes.to_le().nchash.murmur32()
```

### Geo Hashers
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
    plh.col('coord').geohash.from_coords().alias('geohash')
)
shape: (1, 2)
┌─────────────────────┬──────────────┐
│ coord               ┆ geohash      │
│ ---                 ┆ ---          │
│ struct[2]           ┆ str          │
╞═════════════════════╪══════════════╡
│ {-120.6623,35.3003} ┆ 9q60y60rhsgg │
└─────────────────────┴──────────────┘


pl.select(pl.lit('9q60y60rhs').geohash.to_coords().alias('coordinates'))
shape: (1, 1)
┌───────────────────────┐
│ coordinates           │
│ ---                   │
│ struct[2]             │
╞═══════════════════════╡
│ {-120.6623,35.300298} │
└───────────────────────┘
```

### H3 Spatial Index
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
    plh.col('coord').h3.from_coords().alias('h3')
)
shape: (1, 2)
┌─────────────────────┬─────────────────┐
│ coord               ┆ h3              │
│ ---                 ┆ ---             │
│ struct[2]           ┆ str             │
╞═════════════════════╪═════════════════╡
│ {-120.6623,35.3003} ┆ 8c29adc423821ff │
└─────────────────────┴─────────────────┘
```


### Time Hasher

Bins timestamps into variable-precision sliding windows of time, so rows that
fall in the same window share a hash. Timestamps must lie between 1970-01-01 and
2098-01-01. A higher precision means a shorter window: 10 covers about 4 seconds,
8 about 4 minutes.

Precision may be 1 to 32, but past about 18 the hash stops changing for present-day
timestamps and the extra characters are padding. The exact point depends on the date:
timestamps close to 1970 keep splitting to about 21, far-future ones run out sooner.

```python
from datetime import datetime

df = pl.DataFrame({"datetime": [datetime(2017, 2, 21, 20, 15, 13)]})

df.with_columns(
    plh.col('datetime').timehash.from_datetime().alias('timehash')
)
shape: (1, 2)
┌─────────────────────┬────────────┐
│ datetime            ┆ timehash   │
│ ---                 ┆ ---        │
│ datetime[μs]        ┆ str        │
╞═════════════════════╪════════════╡
│ 2017-02-21 20:15:13 ┆ afcccc0e1b │
└─────────────────────┴────────────┘


pl.select(pl.lit('afcccc0e1b').timehash.to_datetime().alias('datetime'))
shape: (1, 1)
┌────────────────────────────────┐
│ datetime                       │
│ ---                            │
│ datetime[μs, UTC]              │
╞════════════════════════════════╡
│ 2017-02-21 20:15:11.292315 UTC │
└────────────────────────────────┘


pl.select(pl.lit('afcccc0e1b').timehash.neighbors().alias('neighbors'))
shape: (1, 1)
┌─────────────────────────────┐
│ neighbors                   │
│ ---                         │
│ struct[2]                   │
╞═════════════════════════════╡
│ {"afcccc0e1a","afcccc0e1c"} │
└─────────────────────────────┘
```

### Deterministic UUIDs

`uuidhash` makes UUID version 5 values (RFC 4122). A v5 UUID is a SHA-1 digest of a
namespace UUID and a name, so the same namespace and the same name always give the same
UUID. A null input gives null.

```python
df = pl.DataFrame({"literal": ["hello", None, "world"]})

df.select(plh.col('literal').uuidhash.uuid5())
shape: (3, 1)
┌──────────────────────────────────────┐
│ literal                              │
│ ---                                  │
│ str                                  │
╞══════════════════════════════════════╡
│ 9342d47a-1bab-5709-9869-c840b2eac501 │
│ null                                 │
│ b3a4c24e-f57a-5448-b81b-a643f6768036 │
└──────────────────────────────────────┘


pl.select(pl.lit('https://example.com').uuidhash.uuid5('url').alias('uuid'))
shape: (1, 1)
┌──────────────────────────────────────┐
│ uuid                                 │
│ ---                                  │
│ str                                  │
╞══════════════════════════════════════╡
│ 4fd35a71-71ef-5a55-a9d9-aa75c889a6d0 │
└──────────────────────────────────────┘
```

The namespace is `"dns"`, `"url"`, `"oid"`, `"x500"`, or a custom UUID of your own.

## Create hash from multiple columns
Give a `separator` value. Without one, `("ab", "c")` and `("a", "bc")` make the same
string and therefore the same digest.

```python
df = pl.DataFrame({"foo": ["hello_world"], "bar": ["today"]})

result = df.select(plh.concat_str("foo", "bar", separator="|").chash.sha2_256())
```

To hash a row of any column type, and not only strings, use `hash_rows` below.

## Hash a whole row

`hash_rows` gives each row bytes that no other row can make, for all column types.
Any hasher then reads those bytes.

```python
df = pl.DataFrame(
    {"foo": ["hello_world"], "bar": [42], "baz": [[1, 2, 3]], "qux": [{"a": 1}]}
)

df.select(plh.hash_rows(pl.all()).chash.sha2_256())
shape: (1, 1)
┌──────────────────────────────────────────────────────────────────┐
│ foo                                                              │
│ ---                                                              │
│ str                                                              │
╞══════════════════════════════════════════════════════════════════╡
│ 9055866af8d3c113e0a8fdb729ce8e6fa67ed5f6f51efa8235a588e88ea972f4 │
└──────────────────────────────────────────────────────────────────┘
```

The encoder reads the meaning of a value, not the polars storage of it. An `Int32` and
the `Int64` next to it make the same hash. A `Datetime` in milliseconds and the same
time in nanoseconds also make the same hash, and a `Categorical` makes the hash of its
string. The encoder does not read the column names. Therefore a new name keeps the
hash, but a new order does not. The
[reference](https://ion-elgreco.github.io/polars-hash/latest/api-reference/rows/)
gives all the rules and the byte layout of version 1, which does not change.
