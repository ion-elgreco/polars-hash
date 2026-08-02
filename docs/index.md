# polars-hash

**Stable non-cryptographic and cryptographic hash functions for Polars.**

polars-hash is a Polars plugin written in Rust. It adds six expression namespaces:
`chash`, `nchash`, `geohash`, `h3`, `timehash`, and `uuidhash`. These namespaces give the same
output on every Polars version. The `hash()` function in Polars does not give this
guarantee. Its output can change when you install a new Polars release.

## Install

```bash
pip install polars-hash
```

## Quick example

```python
import polars as pl
import polars_hash as plh

df = pl.DataFrame({"foo": ["hello_world"]})

df.select(plh.col("foo").chash.sha2_256())
```

```text
┌──────────────────────────────────────────────────────────────────┐
│ foo                                                              │
│ ---                                                              │
│ str                                                              │
╞══════════════════════════════════════════════════════════════════╡
│ 35072c1ae546350e0bfa7ab11d49dc6f129e72ccd57ec7eb671225bbd197c8f1 │
└──────────────────────────────────────────────────────────────────┘
```

## Key features

- **Stable output.** The same input and the same arguments always give the same hash.
  This is true for every polars-hash release.
- **Cryptographic hash functions.** SHA-2, SHA-3, SHAKE128, BLAKE3, and HMAC-SHA256 in
  [`chash`](api-reference/cryptographic.md).
- **Non-cryptographic hash functions.** wyhash, xxHash, XXH3, MurmurHash3, FarmHash,
  CityHash, MD5, and SHA-1 in [`nchash`](api-reference/non-cryptographic.md). Most of
  them accept a seed.
- **Geospatial indexes.** The [`geohash`](api-reference/geohash.md) namespace encodes
  coordinates, decodes geohashes, and finds neighbor cells. The
  [`h3`](api-reference/h3.md) namespace encodes H3 cell indexes.
- **Time buckets.** The [`timehash`](api-reference/timehash.md) namespace encodes an
  instant to the window that holds it, decodes it back, and finds adjacent windows.
- **Deterministic UUIDs.** The [`uuidhash`](api-reference/uuid.md) namespace makes
  UUID v5 values from one or two columns.
- **Type checker support.** [`plh.col` and
  `plh.concat_str`](getting-started.md#plh-col-vs-pl-col) declare the namespaces. You do
  not need `# type: ignore`.
- **Rust speed.** Each expression runs elementwise in compiled Rust and reads the Arrow
  buffers directly. There is no Python callback for each row.

## Next steps

- [Getting Started](getting-started.md). Install the plugin and hash your first column.
- [API Reference](api-reference/index.md). All expressions with their inputs and
  outputs.
