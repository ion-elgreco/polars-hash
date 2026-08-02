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

result = df.select(plh.col('foo').chash.sha256())

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
[the CityHash reference](https://ion-elgreco.github.io/polars-hash/latest/api-reference/non-cryptographic/#cityhash32).

The GxHash expressions need a CPU with AES instructions and have no software fallback.
Every x86, x86-64 and aarch64 wheel is built for them; there are no `linux-armv7` or
`linux-ppc64le` wheels from 0.8.0 on, because GxHash cannot be built for either. See
[the GxHash reference](https://ion-elgreco.github.io/polars-hash/latest/api-reference/non-cryptographic/#gxhash32).

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

## Create hash from multiple columns
```python
df = pl.DataFrame({"foo": ["hello_world"], "bar": ["today"]})

result = df.select(plh.concat_str("foo", "bar").chash.sha256())
```

## Hash a whole row

`concat_str` runs the columns together, so `("ab", "c")` and `("a", "bc")` reach the
same digest, one null makes the whole row null, and a `List` or a `Struct` has no
string form at all. `encode_rows` gives each row bytes no other row can produce, and
any hasher takes them from there.

```python
df = pl.DataFrame(
    {"foo": ["hello_world"], "bar": [42], "baz": [[1, 2, 3]], "qux": [{"a": 1}]}
)

df.select(plh.encode_rows(pl.all()).chash.sha2_256())
shape: (1, 1)
┌──────────────────────────────────────────────────────────────────┐
│ row                                                              │
│ ---                                                              │
│ str                                                              │
╞══════════════════════════════════════════════════════════════════╡
│ 9055866af8d3c113e0a8fdb729ce8e6fa67ed5f6f51efa8235a588e88ea972f4 │
└──────────────────────────────────────────────────────────────────┘
```

`plh.hash_rows(frame)` is the same thing for a whole `DataFrame` or `LazyFrame`, and
adds the result as a column.

A value is hashed for what it means, not for how polars holds it: an `Int32` hashes
like the `Int64` beside it, a millisecond `Datetime` like the same instant in
nanoseconds, and a `Categorical` like the string it stands for. Column names are not
hashed, so renaming is free while reordering is not. The
[reference](https://ion-elgreco.github.io/polars-hash/latest/api-reference/rows/)
states every rule and the byte layout, which is frozen at version 1.
