# `timehash` — time bucket

polars-hash registers these expressions on `pl.Expr` as `.timehash`. A timehash is a
short string that names the window of time an instant falls in. Two instants in the
same window get the same hash, and a shorter hash names a wider window, so a prefix
comparison is a coarser bucket.

All the examples on this page use this data:

```python
from datetime import datetime

import polars as pl
import polars_hash as plh

df = pl.DataFrame({"t": [datetime(2024, 5, 17, 12, 30, 45)]})
```

| Expression | Input | Output |
|------------|-------|--------|
| [`from_datetime(precision, strict)`](#from_datetime) | Datetime, Date, epoch seconds | Utf8 |
| [`to_datetime()`](#to_datetime) | Utf8 | Datetime (UTC) |
| [`neighbors()`](#neighbors) | Utf8 | Struct |

---

## `from_datetime(precision, strict)` { #from_datetime }

Encodes an instant to the timehash of the window that holds it.

```python
df.select(plh.col("t").timehash.from_datetime())
```

```text
┌────────────┐
│ t          │
│ ---        │
│ str        │
╞════════════╡
│ bb1c00aaf0 │
└────────────┘
```

A lower precision gives a shorter hash and a wider window:

```python
df.select(plh.col("t").timehash.from_datetime(8))
# bb1c00aa
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `precision` | `int \| str \| pl.Expr` | `10` | Characters in the hash, 1 to 32. Precision 10 covers about 4 seconds and 8 about 4 minutes. |
| `strict` | `bool` | `True` | Keyword-only. With `False`, a timestamp outside the supported range gives null instead of raising. |

**Returns:** Utf8

!!! note "Accepted input"
    Datetime and Date columns work directly. Epoch seconds may be Float64 or any
    integer type; Float32 cannot hold one closely enough to land in the right window.
    Timestamps must fall between 1970-01-01 and 2098-01-01.

!!! note "Precision past about 18 is padding"
    For present-day timestamps the hash stops changing past roughly 18 characters, so
    the extra ones carry no information.

!!! tip "`strict=False` instead of `when`/`then`"
    A `when`/`then` guard cannot skip out-of-range timestamps, because polars
    evaluates both branches over the whole column. Pass `strict=False` instead.
    Precision stays strict either way.

---

## `to_datetime()` { #to_datetime }

Decodes a timehash to the midpoint of the window it names.

```python
pl.DataFrame({"h": ["bb1c00aaf0"]}).select(plh.col("h").timehash.to_datetime())
```

```text
┌────────────────────────────────┐
│ h                              │
│ ---                            │
│ datetime[μs, UTC]              │
╞════════════════════════════════╡
│ 2024-05-17 12:30:46.327543 UTC │
└────────────────────────────────┘
```

**Returns:** Datetime, microseconds, UTC

!!! note "The zone is not recoverable"
    A timehash holds an instant, not a wall clock, so the original time zone is gone.
    The result is UTC; use `.dt.convert_time_zone(tz)` for another zone. The midpoint
    is not the instant you encoded — it is the centre of the window that instant fell
    in, so a round trip is only exact up to the precision you used.

---

## `neighbors()` { #neighbors }

Gives the windows on either side of the one the hash names.

```python
pl.DataFrame({"h": ["bb1c00aaf0"]}).select(plh.col("h").timehash.neighbors()).unnest("h")
```

```text
┌────────────┬────────────┐
│ before     ┆ after      │
│ ---        ┆ ---        │
│ str        ┆ str        │
╞════════════╪════════════╡
│ bb1c00aaef ┆ bb1c00aaf1 │
└────────────┴────────────┘
```

**Returns:** Struct with the fields `before` and `after`, both Utf8
