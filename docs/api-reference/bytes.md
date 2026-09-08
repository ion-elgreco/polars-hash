# `bytes` — byte encoding

polars-hash registers this namespace on `pl.Expr` as `.bytes`. It turns a value into
its own bytes, so that the result can be piped into any hasher in [`nchash`](non-cryptographic.md)
or [`chash`](cryptographic.md), or written out directly.

Each type keeps its own width: `Int8` becomes 1 byte, `Int32` becomes 4, `Float64`
becomes 8, and so on. `to_le()` and `to_be()` differ only in the byte order of that
width. A value that is already bytes -- `Utf8` or `Binary` -- has no endianness of its
own, and passes through unchanged either way.

All the examples on this page use this data:

```python
import polars as pl
import polars_hash as plh

df = pl.DataFrame({"literal": [1]}, schema={"literal": pl.Int32})
```

| Expression | Input | Output |
|------------|-------|--------|
| [`to_le()`](#to_le) | Boolean, Int8/16/32/64, UInt8/16/32/64, Float32/64, Utf8, Binary | Binary |
| [`to_be()`](#to_be) | Boolean, Int8/16/32/64, UInt8/16/32/64, Float32/64, Utf8, Binary | Binary |

---

## `to_le()` { #to_le }

Encodes the value as its own bytes, little-endian.

```python
df.select(plh.col("literal").bytes.to_le())
```

```text
┌─────────────────────┐
│ literal              │
│ ---                  │
│ binary               │
╞══════════════════════╡
│ b"\x01\x00\x00\x00"  │
└──────────────────────┘
```

**Input:** `Boolean`, `Int8`/`16`/`32`/`64`, `UInt8`/`16`/`32`/`64`, `Float32`/`64`,
`Utf8`, or `Binary`.

**Returns:** `Binary`, the same width as the input type. `Boolean` and the two 8-bit
integer types write a single byte (`Boolean` as `0x00`/`0x01`) -- there is no second
byte to order, so `to_le()` and `to_be()` agree on those three types. `Utf8` writes its
raw UTF-8 bytes and `Binary` passes through, in both cases unchanged by endianness.

**A different width.** Cast first. `pl.col("x").cast(pl.Int64).bytes.to_le()` widens a
narrower integer to 8 bytes before encoding it, sign-extended the way any Polars
numeric cast sign-extends.

**Errors.** Polars raises this as `ComputeError` when it collects the data:

| Condition | Message |
|-----------|---------|
| The input is a type this namespace does not list, e.g. `Date` or `Decimal` | `expected a numeric, Boolean, String or Binary input, got \`date\`` |

## `to_be()` { #to_be }

Encodes the value as its own bytes, big-endian. Everything on [`to_le()`](#to_le)
applies the same way, except for the byte order.

```python
df.select(plh.col("literal").bytes.to_be())
```

```text
┌──────────────────────┐
│ literal               │
│ ---                   │
│ binary                │
╞═══════════════════════╡
│ b"\x00\x00\x00\x01"   │
└───────────────────────┘
```

!!! note "Composing with a hasher"
    `bytes` encodes a value; it does not hash one. Pipe the result into a hasher to
    get a byte-precise hash of the value itself, rather than of a string
    representation of it:

    ```python
    pl.col("id").cast(pl.Int64).bytes.to_le().nchash.murmur32(seed=0)
    ```
