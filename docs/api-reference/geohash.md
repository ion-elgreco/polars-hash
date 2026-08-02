# `geohash` — Geohash

polars-hash registers these expressions on `pl.Expr` as `.geohash`. They encode
coordinates, decode geohashes, and find neighbor cells.

A geohash is a base-32 string. It gives the name of a rectangular cell on the earth.
The prefixes are hierarchical. Cell `9q60y` contains all the points with a geohash that
starts with `9q60y`. A `starts_with` filter is therefore a query for a rectangular
area. Two geohashes with a long identical prefix are near to each other.

All the examples on this page use this data:

```python
import polars as pl
import polars_hash as plh

df = pl.DataFrame(
    {"coord": [{"longitude": -120.6623, "latitude": 35.3003}]},
    schema={
        "coord": pl.Struct(
            [pl.Field("longitude", pl.Float64), pl.Field("latitude", pl.Float64)]
        ),
    },
)
```

| Expression | Input | Output |
|------------|-------|--------|
| [`from_coords(len)`](#from_coords) | Struct | Utf8 |
| [`to_coords()`](#to_coords) | Utf8 | Struct |
| [`neighbors()`](#neighbors) | Utf8 | Struct |

---

## `from_coords(len)` { #from_coords }

Encodes a coordinate struct to a geohash string.

```python
df.select(plh.col("coord").geohash.from_coords(5))
```

```text
┌───────┐
│ coord │
│ ---   │
│ str   │
╞═══════╡
│ 9q60y │
└───────┘
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `len` | `int \| str \| pl.Expr` | `12` | The number of characters, from 1 to 12. An `int` applies to all the rows. polars-hash reads a `str` as a column name and a `pl.Expr` as an expression. The precision can then be different for each row. |

**Input:** a `Struct` with a `latitude` field and a `longitude` field. Both fields must
be `Float32` or `Float64`. polars-hash casts `Float32` to `Float64`. It finds the
fields by name, thus the order of the fields is not important. It ignores the other
fields.

**Returns:** Utf8

**Precision.** Each character adds approximately 5 bits. The cells thus become small
quickly:

| `len` | Approximate cell size |
|-------|-----------------------|
| 1 | 5000 × 5000 km |
| 3 | 156 × 156 km |
| 5 | 4.9 × 4.9 km |
| 7 | 153 × 153 m |
| 9 | 4.8 × 4.8 m |
| 12 | 3.7 × 1.9 cm |

**Errors.** Polars raises all of these as `ComputeError` when it collects the data:

| Condition | Message |
|-----------|---------|
| `len` is less than 1 or more than 12 | `Invalid length specified: 13. Accepted values are between 1 and 12, inclusive` |
| The latitude is outside −90 to 90, or the longitude is outside −180 to 180 | `invalid coordinate range: COORD(-120.6623 91.0)` |
| `len` is null | `Length may not be null` |
| `latitude` or `longitude` is not a float | `Latitude input needs to be float` |

A null `latitude` or a null `longitude` does not cause an error. That row gives null. A
float `len` does not cause an error either. polars-hash casts it to `Int64`, thus `5.9`
gives a precision of 5.

**Different precision for each row.** Give a column name or an expression:

```python
df = pl.DataFrame({"latitude": [35.3003], "longitude": [-120.6623], "n": [5]})
df = df.with_columns(coord=pl.struct(["latitude", "longitude"]))

df.select(plh.col("coord").geohash.from_coords("n"))
df.select(plh.col("coord").geohash.from_coords(pl.col("n")))
```

Both lines give the same result.

All integer data types are permitted, signed and unsigned. polars-hash casts the value
to `Int64`.

---

## `to_coords()` { #to_coords }

Decodes a geohash string to coordinates. The result is the **center** of the cell. If
you encode a coordinate and then decode the geohash, the result is thus different from
the initial coordinate. The difference is less than the size of the cell.

```python
pl.select(pl.lit("9q60y60rhs").geohash.to_coords().alias("coordinates"))  # type: ignore
```

```text
┌───────────────────────┐
│ coordinates           │
│ ---                   │
│ struct[2]             │
╞═══════════════════════╡
│ {-120.6623,35.300298} │
└───────────────────────┘
```

**Returns:** a `Struct` with two `Float64` fields in this order:

| Field | Type | Description |
|-------|------|-------------|
| `longitude` | `Float64` | The longitude of the cell center. |
| `latitude` | `Float64` | The latitude of the cell center. |

!!! note "The first field is longitude"
    `from_coords` finds the fields of the input struct by name. But `to_coords` puts
    `longitude` before `latitude` in the output. Use `unnest`, or select the fields by
    name and not by position:

    ```python
    df.select(plh.col("coord").geohash.from_coords(12).geohash.to_coords()).unnest("coord")
    ```

**Errors.** An incorrect geohash string raises `ComputeError`. A null row gives a
struct with two null fields.

---

## `neighbors()` { #neighbors }

Gives the eight geohash cells around a cell. The neighbor cells have the same precision
as the input cell. Use this expression to find near points. A point near the edge of a
cell can have near points in an adjacent cell. If you search the cell and its eight
neighbors, you find these points also.

```python
(
    pl.from_dicts({"h1": "sp1xk2m6194y"})
    .with_columns(plh.col("h1").geohash.neighbors())
    .unnest("h1")
)
```

```text
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ n            ┆ ne           ┆ e            ┆ se           ┆ s            ┆ sw           ┆ w            ┆ nw           │
│ ---          ┆ ---          ┆ ---          ┆ ---          ┆ ---          ┆ ---          ┆ ---          ┆ ---          │
│ str          ┆ str          ┆ str          ┆ str          ┆ str          ┆ str          ┆ str          ┆ str          │
╞══════════════╪══════════════╪══════════════╪══════════════╪══════════════╪══════════════╪══════════════╪══════════════╡
│ sp1xk2m6194z ┆ sp1xk2m6195p ┆ sp1xk2m6195n ┆ sp1xk2m6195j ┆ sp1xk2m6194v ┆ sp1xk2m6194t ┆ sp1xk2m6194w ┆ sp1xk2m6194x │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

**Returns:** a `Struct` with eight `Utf8` fields in this order:

| Field | Direction |
|-------|-----------|
| `n` | North |
| `ne` | North-east |
| `e` | East |
| `se` | South-east |
| `s` | South |
| `sw` | South-west |
| `w` | West |
| `nw` | North-west |

**Errors.** An incorrect geohash string raises `ComputeError`. A null row gives a struct
with eight null fields.
