# API Reference

::: polars_hash
    options:
      members: false

## The namespaces

One import registers all seven namespaces on `pl.Expr`. Each page below is generated
from the docstrings of the namespace.

| Namespace | Contents |
|-----------|----------|
| [`chash`](cryptographic.md) | SHA-2, SHA-3, SHAKE128, BLAKE3 and HMAC-SHA256 |
| [`nchash`](non-cryptographic.md) | wyhash, xxHash, XXH3, MurmurHash3, FarmHash, CityHash, GxHash, CRC-32C, MD5 and SHA-1 |
| [`bytes`](bytes.md) | The bytes of a value, least or most significant byte first |
| [`geohash`](geohash.md) | Geohash encode, decode and neighbors |
| [`h3`](h3.md) | The H3 hexagonal cell index |
| [`timehash`](timehash.md) | Time-bucket encode, decode and neighbors |
| [`uuidhash`](uuid.md) | Deterministic UUID v5 |

[`hash_rows`](rows.md) is a function and not a namespace. It hashes a full row, which a
hash of the joined columns cannot do.

## Typed wrappers

::: polars_hash.col
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true

::: polars_hash.concat_str
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true

::: polars_hash.HExpr
    options:
      heading_level: 3
      members: false
      show_root_heading: true
      show_root_toc_entry: true

## Conventions

These rules apply to every expression above.

- **Elementwise.** Each expression has `is_elementwise=True`. You can use it in
  `select`, in `with_columns`, in `group_by(...).agg`, and in streaming mode. Polars
  can also divide the data into chunks and change the order of operations.
- **Null values.** A null input gives a null output. The expression does not hash a
  substitute value. [`hash_rows`][polars_hash.hash_rows] is the exception. A null is
  one of the values of a row, and therefore a row with a null also has a hash. The
  rules for the scalar arguments are different: `length`, `key`, `namespace`,
  `default`, `len` and `precision` must not be null, and neither may `seed` — except
  on [`cityhash64()`][polars_hash.NonCryptographicHashingNameSpace.cityhash64], where
  `seed=None` is how you ask for the unseeded algorithm.
- **Output name.** The output column has the same name as the input column. To keep
  both columns, use `.alias()`. [`hash_rows`][polars_hash.hash_rows] reads more than
  one column, and it keeps the name of the first, as the polars `*_horizontal`
  expressions do.
- **Object columns.** Polars sends an `Object` column to a plugin as `Binary`, and it
  keeps no mark to identify the two. Therefore a hasher reads the eight bytes of the
  CPython pointer and not the value. These bytes change with each run. The digest is
  not repeatable, and two equal objects give two different digests. Change an `Object`
  column to a usual data type before you hash it. [`hash_rows`][polars_hash.hash_rows]
  rejects such a column.
- **Incorrect input type.** The expression raises an error when the input type is not
  permitted. This occurs when Polars collects the data, not when you build the
  expression. All errors from the plugin become `polars.exceptions.ComputeError` in
  Python. The message starts with `the plugin failed with message:`.
- **Stability.** The same input and the same arguments always give the same output.
  This does not change between polars-hash releases or Polars releases. The exception
  is [GxHash][polars_hash.NonCryptographicHashingNameSpace.gxhash64], whose values
  hold within one major version of the algorithm. polars-hash pins that version, so
  only a release that says so can move them.
