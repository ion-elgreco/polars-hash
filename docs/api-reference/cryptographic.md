# `chash` — Cryptographic hash functions

polars-hash registers these expressions on `pl.Expr` as `.chash`. Each expression
accepts Utf8 or Binary, and gives a hexadecimal string in lowercase. A hash reads
bytes. Therefore the data type of the input does not change the digest. A null input
gives a null output.

All the examples on this page use this data:

```python
import polars as pl
import polars_hash as plh

df = pl.DataFrame({"foo": ["hello_world"]})
```

| Expression | Digest | Output length |
|------------|--------|---------------|
| [`sha2_224()`](#sha2_224) | SHA-224 | 56 characters |
| [`sha2_256()`](#sha2_256) | SHA-256 | 64 characters |
| [`sha2_384()`](#sha2_384) | SHA-384 | 96 characters |
| [`sha2_512()`](#sha2_512) | SHA-512 | 128 characters |
| [`sha3_224()`](#sha3_224) | SHA3-224 | 56 characters |
| [`sha3_256()`](#sha3_256) | SHA3-256 | 64 characters |
| [`sha3_384()`](#sha3_384) | SHA3-384 | 96 characters |
| [`sha3_512()`](#sha3_512) | SHA3-512 | 128 characters |
| [`sha3_shake128(length)`](#sha3_shake128) | SHAKE128 | `2 × length` characters |
| [`blake3()`](#blake3) | BLAKE3 | 64 characters |
| [`hmac_sha256(key)`](#hmac_sha256) | HMAC-SHA256 | 64 characters |
| [`sha256()`](#sha256) | SHA-256, **deprecated** | 64 characters |

---

## `sha2_224()` { #sha2_224 }

SHA-224 from the SHA-2 family.

```python
df.select(plh.col("foo").chash.sha2_224())
```

```text
69c9392f54e5a0e0fff8945e9ed6475ef89236092a52b2005776912c
```

**Returns:** Utf8

---

## `sha2_256()` { #sha2_256 }

SHA-256 from the SHA-2 family.

```python
df.select(plh.col("foo").chash.sha2_256())
```

```text
35072c1ae546350e0bfa7ab11d49dc6f129e72ccd57ec7eb671225bbd197c8f1
```

**Returns:** Utf8

---

## `sha2_384()` { #sha2_384 }

SHA-384 from the SHA-2 family.

```python
df.select(plh.col("foo").chash.sha2_384())
```

```text
7f251a65acbe92af4c6a6d624c0860d9be77329e10e5beb3b9594f7916128cd95610a4d84e3a83a24a72362f6c8f9c46
```

**Returns:** Utf8

---

## `sha2_512()` { #sha2_512 }

SHA-512 from the SHA-2 family.

```python
df.select(plh.col("foo").chash.sha2_512())
```

```text
94f427efefa74c1230c3e93c35104dcbaa8ff71ba4537583ed83c0449d607c4e61b39c4c5eea5543e01d76a68e223da02b500530a82156625cb96ee8c8c80a85
```

**Returns:** Utf8

---

## `sha3_224()` { #sha3_224 }

SHA3-224 from the SHA-3 (Keccak) family.

```python
df.select(plh.col("foo").chash.sha3_224())
```

```text
e24c066a49e260ba46a7b73d5d2374bfe86670be8ebbdf547bfce343
```

**Returns:** Utf8

---

## `sha3_256()` { #sha3_256 }

SHA3-256 from the SHA-3 family. Same digest size as SHA-256, different construction —
the two do not produce the same value.

```python
df.select(plh.col("foo").chash.sha3_256())
```

```text
fed30406b832b6c457e1e3605016eadfe7b57074c050e16ce2321de734ab29f4
```

**Returns:** Utf8

---

## `sha3_384()` { #sha3_384 }

SHA3-384 from the SHA-3 family.

```python
df.select(plh.col("foo").chash.sha3_384())
```

```text
d407e9fb45a350dce0d557f4d3d514f0a7db816163d3666b9e3ae61339a8b0a500129ef456fb9af7105c606599bc3ca1
```

**Returns:** Utf8

---

## `sha3_512()` { #sha3_512 }

SHA3-512 from the SHA-3 family.

```python
df.select(plh.col("foo").chash.sha3_512())
```

```text
3d96f9b16a74980badc6aa05f8f102d781212744aee86e4c20f75c427f79ccea709487b2562c6e633607b53d0b247c389b88a9c3a9e032fadbdfe6ab9e00c528
```

**Returns:** Utf8

---

## `sha3_shake128(length)` { #sha3_shake128 }

SHAKE128, the extendable-output function from the SHA-3 family. The other digests on
this page have a fixed size. For this expression, you set the number of bytes.

```python
df.select(plh.col("foo").chash.sha3_shake128(length=10))
```

```text
6b57b385e070e3534257
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `length` | `int` | required | Keyword-only. The digest size **in bytes**. The hex output has two characters for each byte. `length` must not be negative. A `length` of `0` gives an empty string. A negative `length` raises `ComputeError: could not parse kwargs: 'decoding error: invalid value: integer -1, expected usize'`. |

**Returns:** Utf8

!!! note "Prefix property"
    A short SHAKE128 output is the first part of a longer output for the same input.
    If you cut a long digest to *n* bytes, you get the same string as a digest with
    `length=n`.

---

## `blake3()` { #blake3 }

BLAKE3 with the default 256-bit output. BLAKE3 is much faster than SHA-2. Use it for
large quantities of data.

```python
df.select(plh.col("foo").chash.blake3())
```

```text
9833e5324eb2400de814730f4e92810905351bc0451e10b75847210c1d7c37ed
```

This expression hashes the bytes of a `Binary` column. Each expression on this page
does the same:

```python
pl.select(pl.lit(b"my_bytes").chash.blake3())  # type: ignore
```

```text
4656d42e3468733c9316ef5d4e4488682fc41ad441644ca63cde6aced8378605
```

**Returns:** Utf8

---

## `hmac_sha256(key)` { #hmac_sha256 }

Keyed HMAC-SHA256 (RFC 2104). The digest is a function of the input and the key. One
input with two different keys gives two different digests.

```python
df.select(plh.col("foo").chash.hmac_sha256(key="secret"))
```

```text
e0f5b5bb7264e77b340a55a694a6c9ca4edc035c394c703a0408f099563be1ca
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | `str` | required | Keyword-only. The key can have any length. An empty key is permitted. polars-hash expands the key one time for each expression, not one time for each row. |

**Returns:** Utf8

!!! note "The key is part of the query plan"
    polars-hash writes `key` into the keyword arguments of the expression. The key
    therefore appears in the output of `explain()` and in each plan that you cache or
    write to a log.

---

## `sha256()` { #sha256 }

**Deprecated.** This expression gives the same result as [`sha2_256()`](#sha2_256) and
also shows a `DeprecationWarning`. The name is older than the `sha2_` and `sha3_`
names. It does not show which family the digest comes from.

```python
df.select(plh.col("foo").chash.sha256())
# DeprecationWarning: Call to deprecated method chash.sha256. Use chash.sha2_256() instead.
```

**Returns:** Utf8

Use `sha2_256()` instead. The output does not change, because the two expressions give
the same digest.
