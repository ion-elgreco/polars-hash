//! Changes a row into bytes. The bytes are constant, and therefore a hash of them is
//! also constant.
//!
//! A hash of joined columns is not sufficient. The rows `("ab", "c")` and `("a", "bc")`
//! make the same string. One null value makes the full row null. A List column or a
//! Struct column has no string form. This module writes each row as bytes that no
//! other row can make.
//!
//! Each value has a tag byte and a payload. A payload has a fixed width, or it starts
//! with its own length. Therefore the values need no separator between them.
//! [`VERSION`] keeps these bytes constant.
//!
//! The encoder reads the meaning of a value, not the polars storage of it. Therefore
//! `Int8(1)`, `Int64(1)` and `UInt64(1)` make the same bytes. A `Datetime` in
//! milliseconds and the same time in nanoseconds also make the same bytes. The
//! documentation gives all the rules.
//!
//! # Why this module is not a wrapper
//!
//! Each of the other expressions wraps a crate. That is the rule for this package.
//! This module is a permitted exception. It writes a layout of the polars data types,
//! not a published algorithm. The `polars-row` crate is not a substitute: it does not
//! normalize the widths or the time units, its functions are internal, and its bytes
//! can change with each polars release. This package must give constant bytes.
//!
//! The cost of the exception stays here. [`VERSION`] keeps these bytes, and the golden
//! vectors in the tests make sure that they do not change.

use polars::prelude::*;

/// The encoding version that this module writes. A user can keep a hash for longer
/// than the release that made it. Therefore do not change the bytes below. Add a new
/// version.
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

/// Writes an integer in base 128. The least significant group is first. Each group
/// has the high bit set, but the last group does not.
fn push_varint(out: &mut Vec<u8>, mut value: u128) {
    while value >= 0x80 {
        out.push(value as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Writes a sign byte and then the magnitude. One encoding is sufficient for all the
/// integer widths and for both signs. The sign byte keeps `-1` different from a
/// magnitude that uses all the bits. Therefore `i64::MIN` and `u128::MAX` are both
/// possible, and a small value stays short.
fn push_scalar(out: &mut Vec<u8>, tag: u8, value: i128) {
    out.push(tag);
    out.push(u8::from(value >= 0));
    push_varint(out, value.unsigned_abs());
}

/// Writes a `u128` value that is more than `i128::MAX`. The bytes are the same bytes
/// that [`push_scalar`] writes for a smaller value.
fn push_unsigned(out: &mut Vec<u8>, value: u128) {
    out.push(TAG_INT);
    out.push(1);
    push_varint(out, value);
}

/// Polars reads `-0.0` and `0.0` as one value. It also reads all the NaN payloads as
/// one value. This function does the same, because two equal rows must make the same
/// hash.
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

/// Removes the zeros at the end of the value. Therefore `1.50` and `1.500` make the
/// same bytes, and the declared width of the column has no effect.
///
/// The function removes the zeros with powers of ten. `Decimal(38, 18)` is the usual
/// type for money. One digit at each step made 18 divisions for each row, and a zero
/// value made the most divisions. Six divisions are sufficient for all `i128` values.
fn push_decimal(out: &mut Vec<u8>, unscaled: i128, scale: usize) {
    let (mut unscaled, mut scale) = (unscaled, scale);
    if unscaled == 0 {
        scale = 0;
    } else if scale > 0 && unscaled % 10 == 0 {
        for step in [16usize, 8, 4, 2, 1] {
            let power = 10i128.pow(step as u32);
            while scale >= step && unscaled % power == 0 {
                unscaled /= power;
                scale -= step;
            }
        }
    }
    push_scalar(out, TAG_DECIMAL, unscaled);
    push_varint(out, scale as u128);
}

/// A column that is ready to encode. This code does each data type match, cast and
/// downcast one time. Therefore the encoder reads only values that have a known type.
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
    /// The payload is the physical value multiplied by `per_unit`. Therefore two
    /// different time units make the same hash.
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
    /// A List column. An Array column also uses this variant, after a cast.
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

/// Gives the null rows of a nested column. The child values cannot show these nulls.
/// The result is `None` if the column has no nulls, because that is the usual
/// condition.
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

/// Gives the number of nanoseconds in one unit. Therefore a `Datetime` or a
/// `Duration` makes the same payload for each unit.
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
            DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                Column::Float(s.cast(&DataType::Float64)?.f64()?.clone())
            }
            DataType::String => Column::Text(s.str()?.clone()),
            // The encoder hashes the string of a category, not its physical index.
            // The index depends on the order of the first values, not on the data.
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
            // A time zone changes the display of a time, but not the time itself.
            // Therefore the encoder does not read it.
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
                let offsets = arr.offsets();
                // The offsets apply to the slice, but `get_inner` gives the full
                // values buffer of the source column. This code cuts the buffer.
                // Therefore a slice of ten rows does not read the values before it.
                let (first, last) = (*offsets.first(), *offsets.last());
                let values = ca.get_inner().slice(first, (last - first) as usize);
                Column::List {
                    valid: row_validity(&s)?,
                    offsets: offsets.as_slice().iter().map(|o| o - first).collect(),
                    values: Box::new(Column::prepare(&values)?),
                }
            }
            // An Array contains the same values as a List, but it keeps its length
            // in the data type. Therefore it makes the same bytes as a List.
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
                // The encoder writes the field count, as it does for the element
                // count of a list. Without the count, the last field continues into
                // the next value, and two different rows make the same bytes.
                out.push(TAG_STRUCT);
                push_varint(out, fields.len() as u128);
                for field in fields {
                    field.encode(i, out);
                }
            }
        }
    }
}

/// Changes each row of `inputs` into its canonical bytes.
///
/// A row is the inputs in their given order. Therefore the caller sets the content of
/// a row, and the encoder does not read the column names. The result has no null
/// values. A null is one of the values that the encoding writes. Therefore a row with
/// a null value also has a hash.
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
