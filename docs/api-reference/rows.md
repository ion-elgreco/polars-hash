# Rows — hash a whole row

polars-hash gives two functions that hash a row rather than a value:
`plh.encode_rows`, an expression, and `plh.hash_rows`, the frame-level shorthand.

Concatenating the columns first cannot do this. `("ab", "c")` and `("a", "bc")` reach
the same string, so they reach the same digest. One null makes the whole row null. A
`List`, an `Array` or a `Struct` has no string form to concatenate at all.

`encode_rows` writes each row as bytes that no other row can produce, and any hasher
in this package takes them from there.

All the examples on this page use this data:

```python
import polars as pl
import polars_hash as plh

df = pl.DataFrame(
    {"foo": ["hello_world"], "bar": [42], "baz": [[1, 2, 3]], "qux": [{"a": 1}]}
)
```

| Function | Input | Output |
|----------|-------|--------|
| [`encode_rows(exprs, version)`](#encode_rows) | Any columns | Binary |
| [`hash_rows(frame, subset, ...)`](#hash_rows) | DataFrame or LazyFrame | The frame, plus a column |

---

## `encode_rows(exprs, *more_exprs, version)` { #encode_rows }

Encodes whole rows to Binary, ready for any hasher in this package.

```python
df.select(plh.encode_rows(pl.all()).chash.sha2_256())
```

```text
233f69694365e098ff2f019ee4ba8bce0e035c0c684fe1bd0417068ca2d371df
```

The bytes themselves are a value you can keep, compare or store:

```python
df.select(plh.encode_rows(pl.all()))
```

```text
b'\r\x05\x0bhello_world\x03\x01*\x0c\x03\x03\x01\x01\x03\x01\x02\x03\x01\x03\r\x03\x01\x01'
```

Naming the columns works as well, and fixes the order of the row:

```python
df.select(plh.encode_rows("foo", "bar").nchash.xxh3_64())
# 1727121123591980269
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exprs` | `IntoExpr \| Iterable[IntoExpr]` | required | The columns that make up the row, in order. Accepts anything `pl.struct` does, selectors included. |
| `*more_exprs` | `IntoExpr` | — | More columns, as positional arguments. |
| `version` | `int` | `1` | The [encoding](#encoding) to write. Version 1 is frozen. |

**Returns:** Binary, one value per row. Never null, even for a row that holds nulls.

---

## `hash_rows(frame, subset, ...)` { #hash_rows }

Adds a column holding the hash of each row.

```python
plh.hash_rows(df.select("foo", "bar"))
```

```text
┌─────────────┬─────┬─────────────────────┐
│ foo         ┆ bar ┆ hash                │
│ ---         ┆ --- ┆ ---                 │
│ str         ┆ i64 ┆ u64                 │
╞═════════════╪═════╪═════════════════════╡
│ hello_world ┆ 42  ┆ 1727121123591980269 │
└─────────────┴─────┴─────────────────────┘
```

It is `encode_rows` plus a hasher, so these two are the same:

```python
plh.hash_rows(df)
df.with_columns(plh.encode_rows(pl.all()).nchash.xxh3_64().alias("hash"))
```

Use `hash_rows` when a frame needs a fingerprint column, and `encode_rows` when the
hash belongs inside a larger expression.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frame` | `pl.DataFrame \| pl.LazyFrame` | required | The frame to fingerprint. A `LazyFrame` stays lazy. |
| `subset` | `IntoExpr \| Iterable[IntoExpr] \| None` | `None` | The columns that make up a row. `None` reads them all. |
| `algorithm` | `str` | `"xxh3_64"` | Any hasher in the `chash`, `nchash` or `uuidhash` namespaces, by name. |
| `name` | `str` | `"hash"` | The name of the column to add. An existing column of that name is replaced. |
| `version` | `int` | `1` | The [encoding](#encoding) to write. Version 1 is frozen. |
| `**kwargs` | `Any` | — | Arguments for the hasher, such as `key` for `hmac_sha256`. |

**Returns:** The frame, with the hash column added.

---

## The encoding { #encoding }

Version 1. These bytes are frozen. A stored hash outlives the release that wrote it,
so a change to the format takes a new version number rather than an edit, and
`version=1` keeps giving these bytes.

A row is a struct value: every column in order, and nothing else. Each value is a tag
byte and a payload. A payload is either fixed width or carries its own length, so the
values need no separator between them and no row can be read two ways.

| Tag | Class | Payload |
|-----|-------|---------|
| `00` | Null | none |
| `01` | False | none |
| `02` | True | none |
| `03` | Integer | sign byte, `1` for zero and above, then the magnitude as a varint |
| `04` | Float | 8 bytes, `f64` big-endian |
| `05` | String | length as a varint, then the UTF-8 bytes |
| `06` | Binary | length as a varint, then the bytes |
| `07` | Date | days from the epoch, as an integer payload |
| `08` | Time | nanoseconds from midnight, as an integer payload |
| `09` | Datetime | nanoseconds from the epoch, as an integer payload |
| `0a` | Duration | nanoseconds, as an integer payload |
| `0b` | Decimal | the unscaled value as an integer payload, then the scale as a varint |
| `0c` | List | element count as a varint, then the elements |
| `0d` | Struct | the fields, in order |

A varint is base 128, least significant group first, with the high bit set on every
group but the last. `Int64(1)` therefore reaches four bytes in a one-column frame:
`0d` for the row, then `03 01 01`.

### What a value is read as

polars holds one value in more than one way. The encoding reads the value, so how it
is held drops out:

| Rule | Effect |
|------|--------|
| Every integer width shares one class | `Int8(1)`, `Int64(1)` and `UInt64(1)` agree |
| `Float32` widens to `Float64` | `Float32(1.5)` and `Float64(1.5)` agree |
| `-0.0` becomes `0.0`, and every NaN payload becomes one NaN | rows polars calls equal cannot hash apart |
| Temporal values become nanoseconds | `Datetime("ms")` and `Datetime("ns")` agree on an instant |
| A time zone is not read | a zone is how an instant is shown, not which instant it is |
| A decimal drops trailing zeros | `1.50` and `1.500` agree |
| `Categorical` and `Enum` are read as their string | the physical index depends on insertion order, so it is not read |
| `Array` is read as a `List` | `Array(Int64, 2)` of `[1, 2]` and `List` of `[1, 2]` agree |

### What stays apart

| Rule | Effect |
|------|--------|
| A null is a value | null, `""` and `0` are three values, and a row holding one still hashes |
| A null `List` is not an empty `List` | and a null `Struct` is not a struct of nulls |
| Each class has its own tag | `1` and `1.0` differ, and so do a `Date` and the `Datetime` at midnight |
| Column order is read | `(1, 2)` and `(2, 1)` differ |

### What is not read at all

**Column names.** Renaming a column keeps every hash. Reordering the columns does not,
because order is what tells the values apart.

**Chunking and slicing.** A frame read in one chunk, in ten, or sliced out of a larger
one gives the same bytes for the same rows.

**The schema beyond the row.** Two columns of `(Int, Int)` and one `Struct` of two
`Int` fields reach the same bytes. Comparing hashes across schemas is outside what the
encoding promises; comparing them within one schema is the whole point.
