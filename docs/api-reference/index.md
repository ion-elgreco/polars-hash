# API Reference

This page lists everything public in polars-hash. One import registers the six
namespaces on `pl.Expr`:

```python
import polars_hash as plh
```

`plh.col` and `plh.concat_str` are typed wrappers around `pl.col` and `pl.concat_str`.
They declare these namespaces. Refer to
[`plh.col` and `pl.col`](../getting-started.md#plh-col-vs-pl-col).

## `chash` — cryptographic

Each expression accepts Utf8 or Binary and gives hex-encoded Utf8, unless the table
shows a different type. Full page: [chash](cryptographic.md).

| Expression | Input | Output | Description |
|------------|-------|--------|-------------|
| [`chash.sha2_224()`](cryptographic.md#sha2_224) | Utf8, Binary | Utf8 | SHA-224 from the SHA-2 family. |
| [`chash.sha2_256()`](cryptographic.md#sha2_256) | Utf8, Binary | Utf8 | SHA-256 from the SHA-2 family. |
| [`chash.sha2_384()`](cryptographic.md#sha2_384) | Utf8, Binary | Utf8 | SHA-384 from the SHA-2 family. |
| [`chash.sha2_512()`](cryptographic.md#sha2_512) | Utf8, Binary | Utf8 | SHA-512 from the SHA-2 family. |
| [`chash.sha3_224()`](cryptographic.md#sha3_224) | Utf8, Binary | Utf8 | SHA3-224 from the SHA-3 family. |
| [`chash.sha3_256()`](cryptographic.md#sha3_256) | Utf8, Binary | Utf8 | SHA3-256 from the SHA-3 family. |
| [`chash.sha3_384()`](cryptographic.md#sha3_384) | Utf8, Binary | Utf8 | SHA3-384 from the SHA-3 family. |
| [`chash.sha3_512()`](cryptographic.md#sha3_512) | Utf8, Binary | Utf8 | SHA3-512 from the SHA-3 family. |
| [`chash.sha3_shake128(length)`](cryptographic.md#sha3_shake128) | Utf8, Binary | Utf8 | SHAKE128 extendable-output function. Gives `length` bytes. |
| [`chash.blake3()`](cryptographic.md#blake3) | Utf8, Binary | Utf8, Binary | BLAKE3 with 256-bit output. |
| [`chash.hmac_sha256(key)`](cryptographic.md#hmac_sha256) | Utf8, Binary | Utf8 | Keyed HMAC-SHA256. |
| [`chash.sha256()`](cryptographic.md#sha256) | Utf8, Binary | Utf8 | **Deprecated.** Alias of `sha2_256()`. |

## `nchash` — non-cryptographic

Full page: [nchash](non-cryptographic.md).

| Expression | Input | Output | Description |
|------------|-------|--------|-------------|
| [`nchash.wyhash()`](non-cryptographic.md#wyhash) | Utf8, Binary | UInt64 | wyhash. The seed is always 0. |
| [`nchash.xxhash32(seed)`](non-cryptographic.md#xxhash32) | Utf8, Binary | UInt32 | XXH32. |
| [`nchash.xxhash64(seed)`](non-cryptographic.md#xxhash64) | Utf8, Binary | UInt64 | XXH64. |
| [`nchash.xxh3_64(seed)`](non-cryptographic.md#xxh3_64) | Utf8, Binary | UInt64 | XXH3 with 64-bit output. |
| [`nchash.xxh3_128(seed)`](non-cryptographic.md#xxh3_128) | Utf8, Binary | UInt128 | XXH3 with 128-bit output. |
| [`nchash.murmur32(seed)`](non-cryptographic.md#murmur32) | Utf8, Binary | UInt32 | MurmurHash3, x86 32-bit variant. |
| [`nchash.murmur128(seed)`](non-cryptographic.md#murmur128) | Utf8, Binary | UInt128 | MurmurHash3, x64 128-bit variant. |
| [`nchash.farmhash32()`](non-cryptographic.md#farmhash32) | Utf8, Binary | UInt32 | FarmHash `fingerprint32`. |
| [`nchash.farmhash64()`](non-cryptographic.md#farmhash64) | Utf8, Binary | UInt64 | FarmHash `fingerprint64`. |
| [`nchash.cityhash32()`](non-cryptographic.md#cityhash32) | Utf8, Binary | UInt32 | CityHash `CityHash32`. |
| [`nchash.cityhash64(seed)`](non-cryptographic.md#cityhash64) | Utf8, Binary | UInt64 | CityHash `CityHash64`, or `CityHash64WithSeed` when given a seed. |
| [`nchash.cityhash128()`](non-cryptographic.md#cityhash128) | Utf8, Binary | UInt128 | CityHash `CityHash128`. |
| [`nchash.gxhash32(seed)`](non-cryptographic.md#gxhash32) | Utf8, Binary | UInt32 | GxHash with 32-bit output. Needs a CPU with AES instructions. |
| [`nchash.gxhash64(seed)`](non-cryptographic.md#gxhash64) | Utf8, Binary | UInt64 | GxHash with 64-bit output. Needs a CPU with AES instructions. |
| [`nchash.gxhash128(seed)`](non-cryptographic.md#gxhash128) | Utf8, Binary | UInt128 | GxHash with 128-bit output. Needs a CPU with AES instructions. |
| [`nchash.md5()`](non-cryptographic.md#md5) | Utf8, Binary | Utf8, Binary | MD5. |
| [`nchash.sha1()`](non-cryptographic.md#sha1) | Utf8, Binary | Utf8 | SHA-1. |

## `geohash` — geohash

Full page: [geohash](geohash.md).

| Expression | Input | Output | Description |
|------------|-------|--------|-------------|
| [`geohash.from_coords(len)`](geohash.md#from_coords) | Struct | Utf8 | Encodes `{latitude, longitude}` to a geohash of `len` characters. `len` is 1 to 12. |
| [`geohash.to_coords()`](geohash.md#to_coords) | Utf8 | Struct | Decodes a geohash to `{longitude, latitude}`. |
| [`geohash.neighbors()`](geohash.md#neighbors) | Utf8 | Struct | Gives the eight adjacent geohashes: `n`, `ne`, `e`, `se`, `s`, `sw`, `w`, `nw`. |

## `h3` — H3 index

Full page: [h3](h3.md).

| Expression | Input | Output | Description |
|------------|-------|--------|-------------|
| [`h3.from_coords(len)`](h3.md#from_coords) | Struct | Utf8 | Encodes `{latitude, longitude}` to an H3 cell index at resolution `len`. `len` is 1 to 15. |

## `timehash` — time bucket

Full page: [timehash](timehash.md).

| Expression | Input | Output | Description |
|------------|-------|--------|-------------|
| [`timehash.from_datetime(precision, strict)`](timehash.md#from_datetime) | Datetime, Date, epoch seconds | Utf8 | Encodes an instant to the timehash of the window that holds it. `precision` is 1 to 32. |
| [`timehash.to_datetime()`](timehash.md#to_datetime) | Utf8 | Datetime (UTC) | Decodes a timehash to the midpoint of its window. |
| [`timehash.neighbors()`](timehash.md#neighbors) | Utf8 | Struct | Gives the preceding and succeeding hash: `before`, `after`. |

## `uuidhash` — UUID v5

Full page: [uuidhash](uuid.md).

| Expression | Input | Output | Description |
|------------|-------|--------|-------------|
| [`uuidhash.uuid5(namespace)`](uuid.md#uuid5) | Utf8, Binary | Utf8 | Makes a UUID v5 in a standard or a custom namespace. |
| [`uuidhash.uuid5_concat(other, default)`](uuid.md#uuid5_concat) | Utf8 | Utf8 | Concatenates two columns and makes a UUID v5 in the DNS namespace. |

## Rows — whole-row hashing

These are functions on `plh`, not expressions in a namespace. They hash a whole row,
which concatenating the columns cannot do. Full page: [rows](rows.md).

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| [`plh.encode_rows(exprs, version)`](rows.md#encode_rows) | Any columns | Binary | Encodes each row to bytes that no other row can produce, ready for any hasher above. |
| [`plh.hash_rows(frame, subset, ...)`](rows.md#hash_rows) | DataFrame or LazyFrame | The frame, plus a column | Adds a column holding the hash of each row. |

## Conventions

These rules apply to all the expressions above.

- **Elementwise.** Each expression has `is_elementwise=True`. You can use it in
  `select`, in `with_columns`, in `group_by(...).agg`, and in streaming mode. Polars
  can also divide the data into chunks and change the order of operations.
- **Null values.** A null input gives a null output. The expression does not hash a
  substitute value. [`encode_rows`](rows.md#encode_rows) is the exception: a null is
  one of the values a row can hold, so a row holding one still has a hash. The scalar
  arguments are different again: `length`, `key`, `namespace`,
  `default`, `len` and `precision` must not be null, and neither may `seed` — except
  on [`cityhash64()`](non-cryptographic.md#cityhash64), where `seed=None` is how you
  ask for the unseeded algorithm.
- **Output name.** The output column has the same name as the input column. To keep
  both columns, use `.alias()`.
- **Incorrect input type.** The expression raises an error when the input type is not
  permitted. This occurs when Polars collects the data, not when you build the
  expression. All errors from the plugin become
  `polars.exceptions.ComputeError` in Python. The message starts with `the plugin
  failed with message:`.
- **Stability.** The same input and the same arguments always give the same output.
  This does not change between polars-hash releases or Polars releases. The exception is
  [GxHash](non-cryptographic.md#gxhash64), whose values hold within one major version of
  the algorithm. polars-hash pins that version, so only a release that says so can move
  them.
