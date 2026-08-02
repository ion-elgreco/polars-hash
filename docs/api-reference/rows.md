# Rows — hash a whole row

`plh.hash_rows` is an expression that hashes a row and not one value.

A hash of the joined columns is not sufficient. The rows `("ab", "c")` and
`("a", "bc")` make the same string, and therefore the same digest. One null makes the
full row null. A `List`, an `Array` or a `Struct` column has no string form.

`hash_rows` writes each row as bytes that no other row can make. Any hasher in this
package then reads those bytes.

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
| [`hash_rows(exprs, version)`](#hash_rows) | Any columns | Binary |

---

## `hash_rows(exprs, *more_exprs, version)` { #hash_rows }

Changes each row into Binary, for use with any hasher in this package.

```python
df.select(plh.hash_rows(pl.all()).chash.sha2_256())
```

```text
9055866af8d3c113e0a8fdb729ce8e6fa67ed5f6f51efa8235a588e88ea972f4
```

You can keep, compare or store the bytes:

```python
df.select(plh.hash_rows(pl.all()))
```

```text
b'\r\x04\x05\x0bhello_world\x03\x01*\x0c\x03\x03\x01\x01\x03\x01\x02\x03\x01\x03\r\x01\x03\x01\x01'
```

You can also give the column names. The names set the order of the row:

```python
df.select(plh.hash_rows("foo", "bar").nchash.xxh3_64())
# 9123089596710669414
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exprs` | `IntoExpr \| Iterable[IntoExpr]` | required | The columns of the row, in order. This argument accepts all that `pl.struct` accepts, and also selectors. |
| `*more_exprs` | `IntoExpr` | — | More columns, as positional arguments. |
| `version` | `int` | `1` | The [encoding](#encoding) to write. Version 1 does not change. |

**Returns:** Binary. There is one value for each row, in a column with the name
`row`. No value is null, and a row with nulls also has a value. The name does not come
from the first column. Therefore `with_columns` adds the encoding and does not replace
a column with it. Use `.alias()` for a different name.

---

## The encoding { #encoding }

This is version 1. These bytes do not change. A user can keep a hash for longer than
the release that made it. Therefore a new format takes a new version number, and
`version=1` always gives these bytes.

A row is a struct value. It contains each column in order, and nothing more. Each
value has a tag byte and a payload. A payload has a fixed width, or it starts with its
own length. Therefore the values need no separator, and a row has only one reading.

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
| `0d` | Struct | field count as a varint, then the fields |

A varint is an integer in base 128. The least significant group is first. Each group
has the high bit set, but the last group does not. Therefore `Int64(1)` takes five
bytes in a frame with one column: `0d 01` for the row and its one field, and then
`03 01 01`.

### How the encoder reads a value

Polars can store one value in more than one way. The encoder reads the value, and not
the storage:

| Rule | Effect |
|------|--------|
| All the integer widths use one class | `Int8(1)`, `Int64(1)` and `UInt64(1)` give the same bytes |
| `Float16` and `Float32` change to `Float64` | all the float widths give the same bytes for `1.5` |
| `-0.0` changes to `0.0`, and each NaN payload changes to one NaN | two rows that polars reads as equal make one hash |
| Time values change to nanoseconds | `Datetime("ms")` and `Datetime("ns")` give the same bytes for one time |
| The encoder does not read a time zone | a zone changes the display of a time, but not the time |
| A decimal loses the zeros at the end | `1.50` and `1.500` give the same bytes |
| `Categorical` and `Enum` change to their string | the physical index depends on the order of the values, and therefore the encoder does not read it |
| An `Array` changes to a `List` | `Array(Int64, 2)` of `[1, 2]` and `List` of `[1, 2]` give the same bytes |

### Which values stay different

| Rule | Effect |
|------|--------|
| A null is a value | null, `""` and `0` are three values, and a row with a null also has a hash |
| A null `List` is not an empty `List` | a null `Struct` is also not a struct of nulls |
| Each class has its own tag | `1` and `1.0` are different, and a `Date` and the `Datetime` at midnight are also different |
| The encoder reads the column order | `(1, 2)` and `(2, 1)` are different |

### What the encoder does not read

**The column names.** A new name for a column keeps all the hashes. A new order of the
columns does not, because the order identifies the values.

**The chunks and the slices.** One chunk, ten chunks, or a slice of a larger frame all
give the same bytes for the same rows.

**Nothing more.** The field counts make two columns of `(Int, Int)` different from one
`Struct` with two `Int` fields. Therefore a change to the schema also changes the hash,
even if the values stay the same.
