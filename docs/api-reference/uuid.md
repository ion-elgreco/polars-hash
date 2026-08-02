# `uuidhash` — UUID v5

polars-hash registers these expressions on `pl.Expr` as `.uuidhash`. They make
deterministic UUID version 5 values (RFC 4122).

A v5 UUID is a SHA-1 digest of a namespace UUID and a name, in UUID format. The result
is deterministic: the same namespace and the same name always give the same UUID. You
can therefore use a v5 UUID as a key for a value that you have. A null input gives a
null output.

| Expression | Input | Output |
|------------|-------|--------|
| [`uuid5(namespace)`](#uuid5) | Utf8 | Utf8 |
| [`uuid5_concat(other, default)`](#uuid5_concat) | Utf8 | Utf8 |

---

## `uuid5(namespace)` { #uuid5 }

Makes a UUID v5 from a string column.

```python
import polars as pl
import polars_hash as plh

df = pl.DataFrame({"literal": ["hello", None, "world"]})

df.select(plh.col("literal").uuidhash.uuid5())
```

```text
┌──────────────────────────────────────┐
│ literal                              │
│ ---                                  │
│ str                                  │
╞══════════════════════════════════════╡
│ 9342d47a-1bab-5709-9869-c840b2eac501 │
│ null                                 │
│ b3a4c24e-f57a-5448-b81b-a643f6768036 │
└──────────────────────────────────────┘
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `namespace` | `UUIDNamespace \| str` | `UUIDNamespace.DNS` | A [standard namespace](#uuidnamespace): `"dns"`, `"url"`, `"oid"`, or `"x500"`. Uppercase and lowercase letters are equivalent. Each other string is a custom namespace UUID. |

**Returns:** Utf8, the 36-character format with hyphens, for example
`9342d47a-1bab-5709-9869-c840b2eac501`.

### `UUIDNamespace` { #uuidnamespace }

`plh.UUIDNamespace` contains the four RFC 4122 namespaces. It is a `str` enum. You can
therefore use the member or its value:

| Member | Value | Namespace |
|--------|-------|-----------|
| `UUIDNamespace.DNS` | `"dns"` | Fully qualified domain names. |
| `UUIDNamespace.URL` | `"url"` | URLs. |
| `UUIDNamespace.OID` | `"oid"` | ISO object identifiers. |
| `UUIDNamespace.X500` | `"x500"` | X.500 distinguished names. |

To use a standard namespace, give its name:

```python
pl.select(pl.lit("https://example.com").uuidhash.uuid5("url"))
# 4fd35a71-71ef-5a55-a9d9-aa75c889a6d0

pl.select(pl.lit("https://example.com").uuidhash.uuid5(plh.UUIDNamespace.URL))
# the same result, because UUIDNamespace is a str enum
```

To use a custom namespace, give your own UUID. Two different namespaces give two
different UUIDs for the same input:

```python
TENANT_NS = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

df.select(plh.col("literal").uuidhash.uuid5(TENANT_NS))
```

**Errors:**

| Condition | Error |
|-----------|-------|
| `namespace` is not a standard name and is not a correct UUID | `ComputeError: Invalid namespace '{value}': {reason}` |
| `namespace` is null | `ComputeError: Namespace must be provided` |

---

## `uuid5_concat(other, default)` { #uuid5_concat }

Concatenates two string columns and makes a UUID v5 from the result. This expression
always uses the **DNS** namespace. Use it to make a key from two columns with one
expression, in place of a `concat_str` and a [`uuid5`](#uuid5).

```python
df = pl.DataFrame({"id": ["abc-123"], "side": ["a"]})

df.select(plh.col("id").uuidhash.uuid5_concat(pl.col("side")))
```

```text
┌──────────────────────────────────────┐
│ id                                   │
│ ---                                  │
│ str                                  │
╞══════════════════════════════════════╡
│ e89d330c-f123-519c-a7a1-e48e46f30ccf │
└──────────────────────────────────────┘
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `other` | `pl.Expr` | required | The second column. polars-hash puts it after the first column. It must be Utf8. If you set `default`, polars-hash casts it to Utf8 first. |
| `default` | `str \| None` | `None` | The value that replaces a null in `other`. With `None`, a null in `other` becomes an empty string. |

**Returns:** Utf8

**Null values.** The two columns are not equivalent:

| First column | Second column (`other`) | `default` | Result |
|--------------|-------------------------|-----------|--------|
| null | any value | any value | null |
| a value | a value | any value | UUID of the first value and the second value |
| a value | null | `None` | UUID of the first value only |
| a value | null | `"a"` | UUID of the first value and `"a"` |

The `default` value and the empty string both give a UUID. Only the added text is
different:

```python
df = pl.DataFrame({"id": ["abc-123"], "side": pl.Series([None], dtype=pl.Utf8)})

df.select(plh.col("id").uuidhash.uuid5_concat(pl.col("side"), default="a"))
# e89d330c-f123-519c-a7a1-e48e46f30ccf, the same result as side="a" above
```

**Errors:** `default` must not be null. A null `default` raises
`ComputeError: Default value may not be null`.

!!! note "This expression adds no separator"
    polars-hash joins the two values directly. `("ab", "c")` and `("a", "bc")`
    therefore give the same UUID. If your data can have this condition, make the key
    with a separator that the data does not contain:

    ```python
    plh.concat_str("id", "side", separator="|").uuidhash.uuid5()
    ```

    With this method you can also select the namespace. `uuid5_concat` always uses the
    DNS namespace.
