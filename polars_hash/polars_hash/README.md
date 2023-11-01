Hellooo :)

This plugin is a work in progress. The main goal of this plugin is to provide a stable hashing functionality across different polars versions.

Main drive behind this plugin is, to generate surrogate table keys that can be determinstic across multiple polars versions.


## Examples
### Cryptographic Hashers

```python
import polars
import polars_hash as plh

df = pl.DataFrame({
    "foo":["hello_world"]
})

result = df.select(plh.col('foo').chash.sha256())

print(result)

┌──────────────────────────────────────────────────────────────────┐
│ foo                                                              │
│ ---                                                              │
│ str                                                              │
╞══════════════════════════════════════════════════════════════════╡
│ 35072c1ae546350e0bfa7ab11d49dc6f129e72ccd57ec7eb671225bbd197c8f1 │
└──────────────────────────────────────────────────────────────────┘
```

### Non-cryptographic Hashers
```python
df = pl.DataFrame({
    "foo":["hello_world"]
})

result = df.select(plh.col('foo').nchash.wyhash())
print(result)
┌──────────────────────┐
│ foo                  │
│ ---                  │
│ str                  │
╞══════════════════════╡
│ 16737367591072095403 │
└──────────────────────┘

```

## Create hash from multiple columns
```python
df = pl.DataFrame({
    "foo":["hello_world"],
    "bar": ["today"]
})

result = df.select(plh.concat_str('foo','bar').chash.sha256())
```
