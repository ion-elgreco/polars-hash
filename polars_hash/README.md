This plugin provides stable hashing functionality across different polars versions.

## Examples
### Cryptographic Hashers

```python
import polars
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
┌─────────────────────┬────────────┐
│ coord               ┆ geohash    │
│ ---                 ┆ ---        │
│ struct[2]           ┆ str        │
╞═════════════════════╪════════════╡
│ {-120.6623,35.3003} ┆ 9q60y60rhs │
└─────────────────────┴────────────┘


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


## Create hash from multiple columns
```python
df = pl.DataFrame({
    "foo":["hello_world"],
    "bar": ["today"]
})

result = df.select(plh.concat_str('foo','bar').chash.sha256())
```
