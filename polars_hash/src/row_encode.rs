//! A canonical byte encoding for a row, so that hashing one is stable.
//!
//! Concatenating columns is not enough to hash a row: `("ab", "c")` and `("a", "bc")`
//! reach the same string, a null swallows the whole row, and a List or a Struct has no
//! string form at all. This module writes a row as bytes that no other row can produce.
//!
//! Every value becomes a tag byte and a payload. A payload is either fixed width or
//! carries its own length, so values sit next to each other with no separator and stay
//! unambiguous. The tags are frozen: [`VERSION`] promises these bytes never change.
//!
//! A value is encoded by what it means rather than by how polars holds it, so
//! `Int8(1)`, `Int64(1)` and `UInt64(1)` all reach the same bytes, as do a `Datetime`
//! in milliseconds and the same instant in nanoseconds. The docs list every rule.

use polars::prelude::*;

/// The encoding this module writes. A change to any byte below needs a new version
/// rather than an edit, because a stored hash outlives the release that wrote it.
pub const VERSION: u64 = 1;

const TAG_NULL: u8 = 0x00;
const TAG_FALSE: u8 = 0x01;
const TAG_TRUE: u8 = 0x02;
const TAG_INT: u8 = 0x03;
const TAG_FLOAT: u8 = 0x04;
const TAG_STRING: u8 = 0x05;
const TAG_BINARY: u8 = 0x06;
const TAG_DATE: u8 = 0x07;
const TAG_TIME: u8 = 0x08;
const TAG_DATETIME: u8 = 0x09;
const TAG_DURATION: u8 = 0x0a;
const TAG_DECIMAL: u8 = 0x0b;
const TAG_LIST: u8 = 0x0c;
const TAG_STRUCT: u8 = 0x0d;

/// Base-128, least significant group first, high bit set on every group but the last.
fn push_varint(out: &mut Vec<u8>, mut value: u128) {
    while value >= 0x80 {
        out.push(value as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Sign and magnitude, so that one encoding covers every integer width and both
/// signs: `i64::MIN` and `u128::MAX` are equally reachable, and a small value stays
/// short. The sign byte is what keeps `-1` clear of a magnitude that spends every bit.
fn push_scalar(out: &mut Vec<u8>, tag: u8, value: i128) {
    out.push(tag);
    out.push(u8::from(value >= 0));
    push_varint(out, value.unsigned_abs());
}

/// The `u128` values above `i128::MAX` reach this instead, and land on the same bytes
/// [`push_scalar`] would give them if they fit.
fn push_unsigned(out: &mut Vec<u8>, value: u128) {
    out.push(TAG_INT);
    out.push(1);
    push_varint(out, value);
}

/// `-0.0` and `0.0` are one value to polars, and so are the NaN payloads, so both
/// collapse here: rows that compare equal may not hash apart.
fn push_float(out: &mut Vec<u8>, value: f64) {
    let value = if value.is_nan() {
        f64::NAN
    } else {
        value + 0.0
    };
    out.push(TAG_FLOAT);
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_bytes(out: &mut Vec<u8>, tag: u8, value: &[u8]) {
    out.push(tag);
    push_varint(out, value.len() as u128);
    out.extend_from_slice(value);
}

/// Trailing zeros of a decimal carry no value, so `1.50` and `1.500` reach the same
/// bytes however wide the column that held them was declared.
fn push_decimal(out: &mut Vec<u8>, unscaled: i128, scale: usize) {
    let (mut unscaled, mut scale) = (unscaled, scale);
    while scale > 0 && unscaled % 10 == 0 {
        unscaled /= 10;
        scale -= 1;
    }
    push_scalar(out, TAG_DECIMAL, unscaled);
    push_varint(out, scale as u128);
}

/// A column ready to encode. Every dtype match, cast and downcast happens once while
/// building this, so encoding a row is only a walk over values that are already typed.
enum Column {
    Null,
    Bool(BooleanChunked),
    Signed(Int64Chunked),
    Unsigned(UInt64Chunked),
    Signed128(Int128Chunked),
    Unsigned128(UInt128Chunked),
    Float(Float64Chunked),
    Text(StringChunked),
    Bytes(BinaryChunked),
    /// A temporal column. Its physical value times `per_unit` gives the payload, which
    /// is what lets one time unit hash like another.
    Temporal {
        tag: u8,
        values: Int64Chunked,
        per_unit: i128,
    },
    Days(Int32Chunked),
    Decimal {
        values: Int128Chunked,
        scale: usize,
    },
    /// A List, and a fixed-size Array once it is cast to one.
    List {
        valid: Option<Vec<bool>>,
        offsets: Vec<i64>,
        values: Box<Column>,
    },
    Struct {
        valid: Option<Vec<bool>>,
        fields: Vec<Column>,
    },
}

/// Row-level nulls of a nested column, which its child values cannot report.
/// `None` when the column has none, so the common case reads no memory at all.
fn row_validity(s: &Series) -> PolarsResult<Option<Vec<bool>>> {
    if s.null_count() == 0 {
        return Ok(None);
    }
    Ok(Some(
        s.is_not_null().iter().map(|v| v == Some(true)).collect(),
    ))
}

fn is_valid(valid: &Option<Vec<bool>>, i: usize) -> bool {
    valid.as_ref().is_none_or(|valid| valid[i])
}

/// Nanoseconds in one unit, so a `Datetime` or a `Duration` reaches the same payload
/// whichever unit polars stored it in.
fn per_unit(unit: &TimeUnit) -> i128 {
    match unit {
        TimeUnit::Nanoseconds => 1,
        TimeUnit::Microseconds => 1_000,
        TimeUnit::Milliseconds => 1_000_000,
    }
}

impl Column {
    fn prepare(s: &Series) -> PolarsResult<Self> {
        let s = s.rechunk();
        let column = match s.dtype() {
            DataType::Null => Column::Null,
            DataType::Boolean => Column::Bool(s.bool()?.clone()),
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                Column::Signed(s.cast(&DataType::Int64)?.i64()?.clone())
            }
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                Column::Unsigned(s.cast(&DataType::UInt64)?.u64()?.clone())
            }
            DataType::Int128 => Column::Signed128(s.i128()?.clone()),
            DataType::UInt128 => Column::Unsigned128(s.u128()?.clone()),
            DataType::Float32 | DataType::Float64 => {
                Column::Float(s.cast(&DataType::Float64)?.f64()?.clone())
            }
            DataType::String => Column::Text(s.str()?.clone()),
            // A category is hashed as the string it stands for. Its physical index
            // depends on the order the values were first seen, which is not a property
            // of the data.
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                Column::Text(s.cast(&DataType::String)?.str()?.clone())
            }
            DataType::Binary | DataType::BinaryOffset => {
                Column::Bytes(s.cast(&DataType::Binary)?.binary()?.clone())
            }
            DataType::Date => Column::Days(s.to_physical_repr().i32()?.clone()),
            DataType::Time => Column::Temporal {
                tag: TAG_TIME,
                values: s.to_physical_repr().i64()?.clone(),
                per_unit: 1,
            },
            // The time zone is how an instant is shown, not which instant it is, so it
            // does not reach the bytes.
            DataType::Datetime(unit, _) => Column::Temporal {
                tag: TAG_DATETIME,
                values: s.to_physical_repr().i64()?.clone(),
                per_unit: per_unit(unit),
            },
            DataType::Duration(unit) => Column::Temporal {
                tag: TAG_DURATION,
                values: s.to_physical_repr().i64()?.clone(),
                per_unit: per_unit(unit),
            },
            DataType::Decimal(_, scale) => Column::Decimal {
                values: s.to_physical_repr().i128()?.clone(),
                scale: *scale,
            },
            DataType::List(_) => {
                let ca = s.list()?;
                let arr = ca
                    .downcast_get(0)
                    .expect("a rechunked column has one chunk");
                Column::List {
                    valid: row_validity(&s)?,
                    offsets: arr.offsets().as_slice().to_vec(),
                    values: Box::new(Column::prepare(&ca.get_inner())?),
                }
            }
            // An Array holds the same values a List does, only with its length in the
            // dtype, so it takes the same bytes rather than a shape of its own.
            DataType::Array(inner, _) => Column::prepare(&s.cast(&DataType::List(inner.clone()))?)?,
            DataType::Struct(_) => Column::Struct {
                valid: row_validity(&s)?,
                fields: s
                    .struct_()?
                    .fields_as_series()
                    .iter()
                    .map(Column::prepare)
                    .collect::<PolarsResult<_>>()?,
            },
            dtype => polars_bail!(InvalidOperation: "cannot encode a {} column into a row", dtype),
        };
        Ok(column)
    }

    fn encode(&self, i: usize, out: &mut Vec<u8>) {
        match self {
            Column::Null => out.push(TAG_NULL),
            Column::Bool(ca) => out.push(match ca.get(i) {
                None => TAG_NULL,
                Some(false) => TAG_FALSE,
                Some(true) => TAG_TRUE,
            }),
            Column::Signed(ca) => match ca.get(i) {
                None => out.push(TAG_NULL),
                Some(v) => push_scalar(out, TAG_INT, v as i128),
            },
            Column::Unsigned(ca) => match ca.get(i) {
                None => out.push(TAG_NULL),
                Some(v) => push_scalar(out, TAG_INT, v as i128),
            },
            Column::Signed128(ca) => match ca.get(i) {
                None => out.push(TAG_NULL),
                Some(v) => push_scalar(out, TAG_INT, v),
            },
            Column::Unsigned128(ca) => match ca.get(i) {
                None => out.push(TAG_NULL),
                Some(v) => push_unsigned(out, v),
            },
            Column::Float(ca) => match ca.get(i) {
                None => out.push(TAG_NULL),
                Some(v) => push_float(out, v),
            },
            Column::Text(ca) => match ca.get(i) {
                None => out.push(TAG_NULL),
                Some(v) => push_bytes(out, TAG_STRING, v.as_bytes()),
            },
            Column::Bytes(ca) => match ca.get(i) {
                None => out.push(TAG_NULL),
                Some(v) => push_bytes(out, TAG_BINARY, v),
            },
            Column::Days(ca) => match ca.get(i) {
                None => out.push(TAG_NULL),
                Some(v) => push_scalar(out, TAG_DATE, v as i128),
            },
            Column::Temporal {
                tag,
                values,
                per_unit,
            } => match values.get(i) {
                None => out.push(TAG_NULL),
                Some(v) => push_scalar(out, *tag, v as i128 * per_unit),
            },
            Column::Decimal { values, scale } => match values.get(i) {
                None => out.push(TAG_NULL),
                Some(v) => push_decimal(out, v, *scale),
            },
            Column::List {
                valid,
                offsets,
                values,
            } => {
                if !is_valid(valid, i) {
                    return out.push(TAG_NULL);
                }
                let (start, end) = (offsets[i] as usize, offsets[i + 1] as usize);
                out.push(TAG_LIST);
                push_varint(out, (end - start) as u128);
                for element in start..end {
                    values.encode(element, out);
                }
            }
            Column::Struct { valid, fields } => {
                if !is_valid(valid, i) {
                    return out.push(TAG_NULL);
                }
                // The field count for the same reason a list writes its element
                // count: without it the last field of a struct runs into whatever
                // follows, and two rows of different shape reach the same bytes.
                out.push(TAG_STRUCT);
                push_varint(out, fields.len() as u128);
                for field in fields {
                    field.encode(i, out);
                }
            }
        }
    }
}

/// Encode every row of `inputs` into its canonical bytes.
///
/// A row is the inputs in order, so the caller decides what a row is and column names
/// never reach the bytes. The result has no nulls of its own: a null is a value the
/// encoding holds, which is what lets a row containing one still have a hash.
pub fn encode_rows(inputs: &[Series], version: u64) -> PolarsResult<BinaryChunked> {
    if version != VERSION {
        polars_bail!(
            InvalidOperation:
            "unknown row encoding version {}, this build writes version {}", version, VERSION
        );
    }

    let rows = inputs[0].len();
    if let Some(other) = inputs.iter().find(|s| s.len() != rows) {
        polars_bail!(
            ShapeMismatch:
            "column {} has length {} and column {} has length {}, expected equal lengths",
            inputs[0].name(), rows, other.name(), other.len()
        );
    }

    let columns: Vec<Column> = inputs
        .iter()
        .map(Column::prepare)
        .collect::<PolarsResult<_>>()?;

    let mut builder = BinaryChunkedBuilder::new(inputs[0].name().clone(), rows);
    let mut row = Vec::with_capacity(64);
    for i in 0..rows {
        row.clear();
        for column in &columns {
            column.encode(i, &mut row);
        }
        builder.append_value(&row);
    }
    Ok(builder.finish())
}
