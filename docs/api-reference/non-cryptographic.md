# `nchash` — Non-cryptographic hash functions

polars-hash registers these expressions on `pl.Expr` as `.nchash`. They are fast and
their output is stable. A null input gives a null output.

All the examples on this page use this data:

```python
import polars as pl
import polars_hash as plh

df = pl.DataFrame({"foo": ["hello_world"]})
```

| Expression | Input | Output | Seed |
|------------|-------|--------|------|
| [`wyhash()`](#wyhash) | Utf8, Binary | UInt64 | always 0 |
| [`xxhash32(seed)`](#xxhash32) | Utf8 | UInt32 | `u32` |
| [`xxhash64(seed)`](#xxhash64) | Utf8 | UInt64 | `u64` |
| [`xxh3_64(seed)`](#xxh3_64) | Utf8 | UInt64 | `u64` |
| [`xxh3_128(seed)`](#xxh3_128) | Utf8 | Binary | `u64` |
| [`murmur32(seed)`](#murmur32) | Utf8 | UInt32 | `u32` |
| [`murmur128(seed)`](#murmur128) | Utf8 | Binary | `u32` |
| [`farmhash32()`](#farmhash32) | Utf8 | UInt32 | — |
| [`farmhash64()`](#farmhash64) | Utf8 | UInt64 | — |
| [`md5()`](#md5) | Utf8, Binary | Utf8 | — |
| [`sha1()`](#sha1) | Utf8 | Utf8 | — |

---

## `wyhash()` { #wyhash }

wyhash with 64-bit output. This expression is very fast. You cannot set the seed; it is
always `0`.

```python
df.select(plh.col("foo").nchash.wyhash())
```

```text
┌──────────────────────┐
│ foo                  │
│ ---                  │
│ u64                  │
╞══════════════════════╡
│ 16737367591072095403 │
└──────────────────────┘
```

This expression also accepts a `Binary` column and hashes the bytes:

```python
pl.select(pl.lit(b"my_bytes").nchash.wyhash())  # type: ignore
# 5112362246832359110
```

**Input:** Utf8 or Binary. A different type raises `ComputeError`:
`wyhash only works on strings or binary data`.

**Returns:** UInt64

---

## `xxhash32(seed)` { #xxhash32 }

XXH32, the original 32-bit xxHash.

```python
df.select(plh.col("foo").nchash.xxhash32())
# 1605956417

df.select(plh.col("foo").nchash.xxhash32(seed=42))
# 1544934469
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | Keyword-only. The value must be in the range of a `u32`, that is `0` to `4294967295`. A value outside this range or a value of `None` raises `expected u32`. |

**Returns:** UInt32

---

## `xxhash64(seed)` { #xxhash64 }

XXH64, the 64-bit xxHash. [`xxh3_64()`](#xxh3_64) is faster. Use `xxhash64()` when you
must get the same values as a different system.

```python
df.select(plh.col("foo").nchash.xxhash64())
# 5654987600477331689

df.select(plh.col("foo").nchash.xxhash64(seed=42))
# 17477110538672341566
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | Keyword-only. The value must be in the range of a `u64`. |

**Returns:** UInt64

---

## `xxh3_64(seed)` { #xxh3_64 }

XXH3 with 64-bit output. For usual string lengths, this is the fastest expression in
the namespace.

```python
df.select(plh.col("foo").nchash.xxh3_64())
# 7060460777671424209

df.select(plh.col("foo").nchash.xxh3_64(seed=42))
# 827481053383045869
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | Keyword-only. The value must be in the range of a `u64`. |

**Returns:** UInt64

---

## `xxh3_128(seed)` { #xxh3_128 }

XXH3 with 128-bit output. The output type is `Binary`, because no integer data type in
Polars holds 128 bits.

```python
df.select(plh.col("foo").nchash.xxh3_128())
```

```text
┌────────────────────────────────────────────────┐
│ foo                                            │
│ ---                                            │
│ binary                                         │
╞════════════════════════════════════════════════╡
│ b"\x03o\xfe!^\x18\xfbg"\xc6=\xaf^\x1c\xd3\xbe" │
└────────────────────────────────────────────────┘
```

To get a hex string, encode the output:

```python
df.select(plh.col("foo").nchash.xxh3_128().bin.encode("hex"))
# 036ffe215e18fb6722c63daf5e1cd3be
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | Keyword-only. The value must be in the range of a `u64`. |

**Returns:** Binary (16 bytes)

---

## `murmur32(seed)` { #murmur32 }

MurmurHash3, x86 32-bit variant. Many systems have an implementation of this
algorithm, for example Spark, Kafka, and bloom filter libraries.

```python
df.select(plh.col("foo").nchash.murmur32())
# 3531928679

df.select(plh.col("foo").nchash.murmur32(seed=42))
# 259561949
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | Keyword-only. The value must be in the range of a `u32`. |

**Returns:** UInt32

With the default seed, an empty string gives `0`. With a different seed, an empty
string gives a value that is not `0`. This is the correct MurmurHash3 result. It is not
a null value in the output.

---

## `murmur128(seed)` { #murmur128 }

MurmurHash3, x64 128-bit variant. The output type is `Binary`.

```python
df.select(plh.col("foo").nchash.murmur128())
# b"\x98,\xf3\x9e\x1c\x1a\xa5]\x1b\x07\x97\x16\x07l\x8de"

df.select(plh.col("foo").nchash.murmur128().bin.encode("hex"))
# 982cf39e1c1aa55d1b079716076c8d65
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | Keyword-only. The value must be in the range of a `u32`. The 128-bit variant also uses a 32-bit seed. |

**Returns:** Binary (16 bytes)

---

## `farmhash32()` { #farmhash32 }

Google FarmHash `fingerprint32`. The fingerprint functions give the same value on all
platforms. BigQuery uses them for its `FARM_FINGERPRINT` function. This expression has
no seed.

```python
pl.DataFrame({"foo": ["hello world"]}).select(plh.col("foo").nchash.farmhash32())
# 430397466
```

**Returns:** UInt32

---

## `farmhash64()` { #farmhash64 }

Google FarmHash `fingerprint64`, the 64-bit fingerprint. This expression has no seed.

```python
pl.DataFrame({"foo": ["hello world"]}).select(plh.col("foo").nchash.farmhash64())
# 6381520714923946011
```

**Returns:** UInt64

!!! note "Signed and unsigned values"
    The `FARM_FINGERPRINT` function in BigQuery gives the same 64 bits as a signed
    `INT64`. To compare the two results, use `.cast(pl.Int64)` on the polars-hash
    output.

---

## `md5()` { #md5 }

MD5, hex-encoded.

```python
df.select(plh.col("foo").nchash.md5())
# 99b1ff8f11781541f7f89f9bd41c4a17
```

This expression also accepts a `Binary` column:

```python
pl.select(pl.lit(b"my_bytes").nchash.md5())  # type: ignore
# 4445d78d11baa258c5f4ac1b8d33b8ba
```

**Input:** Utf8 or Binary. A different type raises `ComputeError`:
`md5 only works on strings or binary data`.

**Returns:** Utf8 with 32 characters

---

## `sha1()` { #sha1 }

SHA-1, hex-encoded.

```python
df.select(plh.col("foo").nchash.sha1())
# e4ecd6fc11898565af24977e992cea0c9c7b7025
```

**Input:** Utf8 only. A Binary column raises `ComputeError: invalid series dtype:
expected String, got binary`. This expression has no path for bytes, but
[`md5()`](#md5) has one.

**Returns:** Utf8 with 40 characters
