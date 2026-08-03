# `nchash` — Non-cryptographic hash functions

polars-hash registers these expressions on `pl.Expr` as `.nchash`. They are fast and
their output is constant. Each expression accepts Utf8 or Binary. A hash reads bytes,
and therefore the data type of the input does not change the digest. A null input
gives a null output.

All the examples on this page use this data:

```python
import polars as pl
import polars_hash as plh

df = pl.DataFrame({"foo": ["hello_world"]})
```

| Expression | Input | Output | Seed |
|------------|-------|--------|------|
| [`wyhash()`](#wyhash) | Utf8, Binary | UInt64 | always 0 |
| [`xxhash32(seed)`](#xxhash32) | Utf8, Binary | UInt32 | `u32` |
| [`xxhash64(seed)`](#xxhash64) | Utf8, Binary | UInt64 | `u64` |
| [`xxh3_64(seed)`](#xxh3_64) | Utf8, Binary | UInt64 | `u64` |
| [`xxh3_128(seed)`](#xxh3_128) | Utf8, Binary | UInt128 | `u64` |
| [`murmur32(seed)`](#murmur32) | Utf8, Binary | UInt32 | `u32` |
| [`murmur128(seed)`](#murmur128) | Utf8, Binary | UInt128 | `u32` |
| [`farmhash32()`](#farmhash32) | Utf8, Binary | UInt32 | — |
| [`farmhash64()`](#farmhash64) | Utf8, Binary | UInt64 | — |
| [`cityhash32()`](#cityhash32) | Utf8, Binary | UInt32 | — |
| [`cityhash64(seed)`](#cityhash64) | Utf8, Binary | UInt64 | `u64`, optional |
| [`cityhash128()`](#cityhash128) | Utf8, Binary | UInt128 | — |
| [`gxhash32(seed)`](#gxhash32) | Utf8, Binary | UInt32 | `u64` |
| [`gxhash64(seed)`](#gxhash64) | Utf8, Binary | UInt64 | `u64` |
| [`gxhash128(seed)`](#gxhash128) | Utf8, Binary | UInt128 | `u64` |
| [`md5()`](#md5) | Utf8, Binary | Utf8 | — |
| [`sha1()`](#sha1) | Utf8, Binary | Utf8 | — |

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

This expression hashes the bytes of a `Binary` column. Each expression on this page
does the same:

```python
pl.select(pl.lit(b"my_bytes").nchash.wyhash())  # type: ignore
# 5112362246832359110
```

A dtype that is neither raises ``ComputeError: expected `String` or `Binary` input``.

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

XXH3 with 128-bit output.

```python
df.select(plh.col("foo").nchash.xxh3_128())
# 253649469245435599925940275794906345219

df.select(plh.col("foo").nchash.xxh3_128(seed=42))
# 314735830047873782861649874643137875266
```

The value matches `xxhash.xxh128_intdigest()`, and formatting it as 32 hex digits
gives the canonical XXH128 digest, the same string as `xxh128_hexdigest()`:

```python
f"{253649469245435599925940275794906345219:032x}"
# bed31c5eaf3dc62267fb185e21fe6f03
```

!!! warning "0.8.0 changed this output from `Binary` to `UInt128`"
    Up to 0.7.0 this expression returned 16 bytes. The bytes held the value in the
    reverse of the canonical order, so `.bin.encode("hex")` gave
    `036ffe215e18fb6722c63daf5e1cd3be` where the reference gives
    `bed31c5eaf3dc62267fb185e21fe6f03`. The integer now agrees with the reference, and
    `f"{value:032x}"` replaces `.bin.encode("hex")`. To read data hashed by an older
    release, reverse the old bytes: `int.from_bytes(old, "little")`.

    A `UInt128` column also cannot reach pandas or NumPy, which the old `Binary` one
    could. [`cityhash128()`](#cityhash128) describes that limitation.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | Keyword-only. The value must be in the range of a `u64`. |

**Returns:** UInt128

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

MurmurHash3, x64 128-bit variant.

```python
df.select(plh.col("foo").nchash.murmur128())
# 134986332493155497415370161450594282648

df.select(plh.col("foo").nchash.murmur128(seed=42))
# 128378975539535818103252123378652633995
```

The value matches `mmh3.hash128(..., signed=False)`. MurmurHash3 writes its digest as
two little-endian halves, so the canonical bytes come back the other way round from
[`xxh3_128()`](#xxh3_128):

```python
(134986332493155497415370161450594282648).to_bytes(16, "little").hex()
# 982cf39e1c1aa55d1b079716076c8d65
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | Keyword-only. The value must be in the range of a `u32`. The 128-bit variant also uses a 32-bit seed. |

**Returns:** UInt128

!!! warning "0.8.0 changed this output from `Binary` to `UInt128`"
    Up to 0.7.0 this expression returned the 16 digest bytes, so `.bin.encode("hex")`
    gave the string above. The bytes were the canonical ones; only the container
    changed. `int.from_bytes(old, "little")` converts a stored value.

    A `UInt128` column also cannot reach pandas or NumPy, which the old `Binary` one
    could. [`cityhash128()`](#cityhash128) describes that limitation.

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

## `cityhash32()` { #cityhash32 }

Google CityHash `CityHash32`, from CityHash v1.1.1. FarmHash replaced CityHash, so use
[`farmhash32()`](#farmhash32) for new work. Use `cityhash32()` when you must get the
same values as a different system. This expression has no seed; `CityHash32` takes none.

!!! warning "Older CityHash releases give other values"
    Every CityHash expression here gives the values of v1.1.1, the last release Google
    published. `CityHash64` changed during the v1.0 series and `CityHash128` changed
    after v1.0.3, so a system built on an earlier release gives a different value for
    the same input. Check which release the other system uses before you compare.

```python
pl.DataFrame({"foo": ["hello_world"]}).select(plh.col("foo").nchash.cityhash32())
# 1719156559
```

**Returns:** UInt32

!!! note "CityHash and FarmHash agree on short input"
    FarmHash reuses CityHash for short input, so `cityhash32()` and
    [`farmhash32()`](#farmhash32) give the same value for input up to 12 bytes — the
    example above is 11 — as do [`cityhash64()`](#cityhash64) and
    [`farmhash64()`](#farmhash64) up to 32 bytes. They part above those lengths. The
    equal values are not a bug.

!!! note "The input is hashed as UTF-8"
    These expressions take Utf8 and hash the UTF-8 encoding of it, so `"élève"` hashes
    as 7 bytes, not 5 characters. A system that feeds UTF-16 or Latin-1 bytes to
    CityHash agrees on ASCII input and disagrees on everything else.

---

## `cityhash64(seed)` { #cityhash64 }

Google CityHash `CityHash64`, from CityHash v1.1.1.

```python
df.select(plh.col("foo").nchash.cityhash64())
# 15605398435621216523

df.select(plh.col("foo").nchash.cityhash64(seed=42))
# 10175920941468920074
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int \| None` | `None` | Keyword-only. The value must be in the range of a `u64`. |

**Returns:** UInt64

!!! warning "A seed of 0 is not the same as no seed"
    Without a seed this expression is `CityHash64`; with one it is
    `CityHash64WithSeed`, a separate function that gives a different value for every
    seed, `0` included. This is why the seed defaults to `None`.

---

## `cityhash128()` { #cityhash128 }

Google CityHash `CityHash128`, from CityHash v1.1.1. The output is a `UInt128`, so the
whole hash is one integer and needs no decoding. `CityHash128WithSeed` is not wrapped,
so this expression has no seed.

```python
df.select(plh.col("foo").nchash.cityhash128())
# 133423608296839006301901834072762183026
```

**Returns:** UInt128

!!! note "How the two 64-bit halves are packed"
    C++ returns `CityHash128` as a pair. This expression packs it the way
    `python-cityhash` does — `Uint128Low64(h) << 64 | Uint128High64(h)`, so the C++
    *low* word is the *high* half of the integer. A system that composes the halves
    the other way round, or that stores the raw 16 bytes, needs a word swap before
    the values compare equal.

!!! warning "`UInt128` does not leave Polars yet"
    Polars encodes `UInt128` as a private Arrow type, so `to_arrow()` and
    `to_pandas()` raise `ArrowInvalid` and `to_numpy()` fails on this column.
    `write_parquet`, `write_ipc`, joins, `group_by` and sorting all work. Cast to
    `pl.Binary` or split the halves if the column has to reach pandas or NumPy.

---

## `gxhash32(seed)` { #gxhash32 }

GxHash with 32-bit output. GxHash reaches its speed through the AES block cipher, which
the CPU runs as a single instruction, so it is the fastest expression in the namespace
for input above about a hundred bytes. Below that,
[`xxh3_64()`](#xxh3_64) is the one to beat.

```python
df.select(plh.col("foo").nchash.gxhash32())
# 2751540945

df.select(plh.col("foo").nchash.gxhash32(seed=42))
# 3382299372
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | Keyword-only. The value must be in the range of a `u64`. |

**Returns:** UInt32

!!! warning "GxHash needs a CPU with AES instructions"
    The algorithm has no software fallback. The published wheels cover x86, x86-64 and
    aarch64, and every one of them is built with the instructions enabled, so a CPU
    without them stops the process the moment a GxHash expression runs. On x86 the
    instructions arrived with Westmere in 2010 and every processor since has them. On
    ARM they are an optional extension: Apple silicon and server parts have them, and
    some small boards, such as the Raspberry Pi 4, do not.

    The instructions are enabled for the whole build rather than for GxHash alone, so on
    x86 `ahash`, which the [`h3`](h3.md) namespace pulls in, switches to its AES-NI
    implementation as well and the `h3` expressions come to need them too. On aarch64
    `ahash` keeps its portable path, so there GxHash is the only namespace affected.

    There are no `linux-armv7` or `linux-ppc64le` wheels from 0.8.0 on, because GxHash
    cannot be built for either.

!!! note "The seed is unsigned here and signed upstream"
    GxHash takes an `i64` seed. This namespace presents every 64-bit seed as a `u64`
    for consistency, so a seed at or above `2**63` is the upstream seed minus `2**64`:
    `seed=2**64 - 1` here is `-1` there. Both reach the same 64 bits.

---

## `gxhash64(seed)` { #gxhash64 }

GxHash with 64-bit output.

```python
df.select(plh.col("foo").nchash.gxhash64())
# 2180020304351407825

df.select(plh.col("foo").nchash.gxhash64(seed=42))
# 15254170022685821676
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | Keyword-only. The value must be in the range of a `u64`. |

**Returns:** UInt64

!!! warning "The values are stable for GxHash 3 only"
    GxHash holds its output stable across platforms, but only within a major version.
    polars-hash pins GxHash 3 exactly, so the values here do not change without a
    release that says so. A system on GxHash 2 gives different values for the same
    input and seed.

!!! note "Seed 0 is the default, not a separate mode"
    Every GxHash expression is seeded, and the default seed is `0`. Unlike
    [`cityhash64()`](#cityhash64), there is no unseeded form to differ from.

---

## `gxhash128(seed)` { #gxhash128 }

GxHash with 128-bit output. Like [`cityhash128()`](#cityhash128) the output is a
`UInt128`, so the whole hash is one integer.

```python
df.select(plh.col("foo").nchash.gxhash128())
# 56218077491375249900279963678916292305

df.select(plh.col("foo").nchash.gxhash128(seed=42))
# 11136336363892181958542060125951740652
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | Keyword-only. The value must be in the range of a `u64`. |

**Returns:** UInt128

!!! note "The three widths are one hash, cut short"
    GxHash builds a single 128-bit state and each width reads the low part of it, so
    `gxhash32()` is the low 32 bits of `gxhash64()`, which is the low 64 bits of
    `gxhash128()`. Ask for the width you need; the narrow ones cost no less than the
    wide one.

!!! warning "`UInt128` does not leave Polars yet"
    The same limitation [`cityhash128()`](#cityhash128) describes applies here: the
    column cannot reach pandas or NumPy without a cast.

---

## `md5()` { #md5 }

MD5, hex-encoded.

```python
df.select(plh.col("foo").nchash.md5())
# 99b1ff8f11781541f7f89f9bd41c4a17
```

This expression hashes the bytes of a `Binary` column:

```python
pl.select(pl.lit(b"my_bytes").nchash.md5())  # type: ignore
# 4445d78d11baa258c5f4ac1b8d33b8ba
```

**Returns:** Utf8 with 32 characters

---

## `sha1()` { #sha1 }

SHA-1, hex-encoded.

```python
df.select(plh.col("foo").nchash.sha1())
# e4ecd6fc11898565af24977e992cea0c9c7b7025
```

**Returns:** Utf8 with 40 characters
