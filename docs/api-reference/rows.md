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

**Returns:** Binary. There is one value for each row, and no value is null. A row
with nulls also has a value.

The output column keeps the name of the first column, as `pl.struct`, `pl.concat_str`
and each `*_horizontal` expression do. Therefore `with_columns` replaces that column.
Use `.alias()` to keep it:

```python
df.with_columns(plh.hash_rows(pl.all()).nchash.xxh3_64().alias("hash"))
```

---

## The encoding { #encoding }

This is version 1. These bytes do not change. A user can keep a hash for longer than
the release that made it. Therefore a new format takes a new version number, and
`version=1` always gives these bytes.

A row is a struct value. It contains each column in order, and nothing more.

### An example

```python
pl.DataFrame({"id": [7], "name": ["ok"]}).select(plh.hash_rows(pl.all()))
```

```text
0d 02 03 01 07 05 02 6f 6b
```

Read the bytes from the left:

| Bytes | Meaning |
|-------|---------|
| `0d` | The row is a struct. |
| `02` | The struct has two fields. |
| `03` | The first field is an integer. |
| `01` | The integer is zero or more. |
| `07` | Its magnitude is 7. |
| `05` | The second field is a string. |
| `02` | The string has two bytes. |
| `6f 6b` | Those two bytes are `ok`. |

Each value starts with one tag byte, which gives the class of the value. The bytes
after the tag are the payload. A payload has a fixed width, or it starts with its own
length or count. Therefore the values need no separator, and a row has only one
reading.

A null value is only its tag. A List and a Struct contain their values in the same
form, one after the other. This row has a null and a list of 1 and 300:

```python
pl.DataFrame({"id": pl.Series([None], dtype=pl.Int64), "tags": [[1, 300]]}).select(
    plh.hash_rows(pl.all())
)
```

```text
0d 02 00 0c 02 03 01 01 03 01 ac 02
```

| Bytes | Meaning |
|-------|---------|
| `0d 02` | A struct with two fields. |
| `00` | The first field is null. |
| `0c` | The second field is a list. |
| `02` | The list has two elements. |
| `03 01 01` | The first element is the integer 1. |
| `03 01 ac 02` | The second element is the integer 300. |

### Numbers with a variable length

A count, a length, and the magnitude of an integer all use a varint. A varint takes
one byte for a value below 128, and one more byte for each 7 bits after that.

Each byte of a varint holds 7 bits of the value, and the least significant 7 bits come
first. Each byte has the high bit set, but the last byte does not:

| Value | Varint |
|-------|--------|
| 1 | `01` |
| 127 | `7f` |
| 128 | `80 01` |
| 300 | `ac 02` |

An integer payload is a sign byte and then a varint of the magnitude. The sign byte is
`01` for zero and more, and `00` for less than zero. Therefore 300 is `01 ac 02`, and
-300 is `00 ac 02`.

### The tags

| Tag | Class | Payload |
|-----|-------|---------|
| `00` | Null | none |
| `01` | False | none |
| `02` | True | none |
| `03` | Integer | an integer payload |
| `04` | Float | 8 bytes, `f64` big-endian |
| `05` | String | the length as a varint, then the UTF-8 bytes |
| `06` | Binary | the length as a varint, then the bytes |
| `07` | Date | the days from 1970-01-01, as an integer payload |
| `08` | Time | the nanoseconds from midnight, as an integer payload |
| `09` | Datetime | the nanoseconds from 1970-01-01, as an integer payload |
| `0a` | Duration | the nanoseconds, as an integer payload |
| `0b` | Decimal | the unscaled value as an integer payload, then the scale as a varint |
| `0c` | List | the element count as a varint, then the elements |
| `0d` | Struct | the field count as a varint, then the fields |

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
| A `Struct` is not its fields | the field count makes one `Struct` of two `Int` fields different from two `Int` columns |

### What the encoder does not read

**The column names.** A new name for a column keeps all the hashes. A new order of the
columns does not, because the order identifies the values. A column that you give two
times is two values of the row.

**The field names of a `Struct`.** The encoder writes the fields in their order, as it
writes the columns of a row. Therefore a new name for a field keeps the hash, and a new
order of the fields does not.

Polars compares two `Struct` values by name, and the encoder reads the order. Therefore
the two do not always agree:

```python
first = pl.Series([{"a": 2, "b": 1}], dtype=pl.Struct({"a": pl.Int64, "b": pl.Int64}))
second = pl.Series([{"b": 1, "a": 2}], dtype=pl.Struct({"b": pl.Int64, "a": pl.Int64}))

first.equals(second)  # True: each name has the same value
# The hashes are different, because the values are in a different order.
```

**The data type of an empty `List`.** An empty `List(Int64)` and an empty
`List(String)` both give the count 0 and no elements.

**The chunks and the slices.** One chunk, ten chunks, or a slice of a larger frame all
give the same bytes for the same rows.
