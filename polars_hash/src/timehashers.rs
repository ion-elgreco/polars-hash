use polars::chunked_array::ops::arity::{try_unary_elementwise, unary_elementwise};
use polars::prelude::*;
use timeharsh::timehash;

/// End of the timehash range (2098-01-01). Past it every timestamp saturates to
/// the same hash, so encoding errors rather than bin to the wrong window.
const MAX_EPOCH_SECONDS: f64 = 4_039_372_800.0;

/// Allocation guard: upstream emits one character per unit of precision, per row.
/// The useful limit is lower and depends on the date -- the f64 interval stops
/// splitting past ~18 characters for present-day timestamps, ~21 close to 1970.
const MAX_PRECISION: i64 = 32;

/// `before`/`after` index the hash by byte but walk it by character, so a
/// multi-byte character panics there before it is ever rejected. Both values are
/// debug-formatted: a raw NUL would panic pyo3-polars as it builds the CString.
fn validate_timehash(value: &str) -> PolarsResult<()> {
    if value.is_empty() {
        polars_bail!(ComputeError: "timehash may not be empty")
    }
    match value.chars().find(|c| !matches!(c, '0' | '1' | 'a'..='f')) {
        Some(c) => {
            polars_bail!(ComputeError: "invalid timehash character {:?} in {:?}", c, value)
        }
        None => Ok(()),
    }
}

/// A source with no non-null values infers `Null` rather than `String`, so treat it
/// as an all-null hash column: whether a query runs should not depend on inference.
pub fn hash_column(s: &Series) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Null => s.cast(&DataType::String),
        _ => Ok(s.clone()),
    }
}

/// Datetime and Date carry their unit in the dtype; other numerics are taken as
/// epoch seconds as-is. Float32 is refused, not widened: near 1.5e9 it spaces
/// values 128 seconds apart, far wider than the 3.8 second window at precision 10.
pub fn epoch_seconds(s: &Series) -> PolarsResult<Float64Chunked> {
    match s.dtype() {
        // Same reasoning as `hash_column`: all-null data must not depend on inference.
        DataType::Null => Ok(Float64Chunked::full_null(s.name().clone(), s.len())),
        // Nanoseconds run past 2^53, so `v as f64` would round the instant before it
        // is scaled. Split first: whole seconds and remainder each convert exactly.
        DataType::Datetime(time_unit, _) => {
            let scale: i64 = match time_unit {
                TimeUnit::Nanoseconds => 1_000_000_000,
                TimeUnit::Microseconds => 1_000_000,
                TimeUnit::Milliseconds => 1_000,
            };
            let physical = s.cast(&DataType::Int64)?;
            Ok(unary_elementwise(physical.i64()?, |v| {
                v.map(|v| (v / scale) as f64 + (v % scale) as f64 / scale as f64)
            }))
        }
        DataType::Date => {
            let physical = s.cast(&DataType::Int64)?;
            Ok(unary_elementwise(physical.i64()?, |v| {
                v.map(|v| v as f64 * 86_400.0)
            }))
        }
        DataType::Float32 => {
            polars_bail!(
                InvalidOperation:
                "Float32 cannot hold epoch seconds precisely enough for timehash, cast to Float64"
            )
        }
        DataType::Float64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => Ok(s.cast(&DataType::Float64)?.f64()?.clone()),
        dtype => {
            polars_bail!(
                InvalidOperation:
                "timehash input needs to be Datetime, Date or numeric epoch seconds, got {}", dtype
            )
        }
    }
}

/// A scalar precision is checked once by `thash_encode` before any row is touched:
/// left to the per-row path it would go unchecked on an all-null or empty column,
/// so the same argument would be rejected or accepted depending on the data.
pub fn validate_precision(precision: i64) -> PolarsResult<()> {
    if !(1..=MAX_PRECISION).contains(&precision) {
        polars_bail!(
            InvalidOperation:
            "expected precision between 1 and {}, got {}", MAX_PRECISION, precision
        )
    }
    Ok(())
}

pub fn timehash_encoder(
    seconds: Option<f64>,
    precision: Option<i64>,
) -> PolarsResult<Option<String>> {
    match seconds {
        Some(seconds) => match precision {
            Some(precision) => {
                validate_precision(precision)?;
                if !(0.0..=MAX_EPOCH_SECONDS).contains(&seconds) {
                    polars_bail!(
                        ComputeError:
                        "invalid timestamp range: {} epoch seconds is outside 0 (1970-01-01) to {} (2098-01-01)",
                        seconds, MAX_EPOCH_SECONDS
                    )
                }
                timehash::encode(seconds, precision as usize)
                    .map(Some)
                    .map_err(|e| PolarsError::ComputeError(e.into()))
            }
            _ => Err(PolarsError::ComputeError(
                "Precision may not be null".to_string().into(),
            )),
        },
        _ => Ok(None),
    }
}

/// Decodes to the midpoint of the hashed window. The hash holds an instant, not a
/// wall clock, so the output is UTC and the caller converts to their own zone.
pub fn timehash_decoder(ca: &StringChunked) -> PolarsResult<Series> {
    let out: Int64Chunked = try_unary_elementwise(ca, |value| -> PolarsResult<Option<i64>> {
        match value {
            Some(value) => {
                validate_timehash(value)?;
                let seconds =
                    timehash::decode(value).map_err(|e| PolarsError::ComputeError(e.into()))?;
                Ok(Some((seconds * 1e6).round() as i64))
            }
            _ => Ok(None),
        }
    })?;

    out.into_series().cast(&DataType::Datetime(
        TimeUnit::Microseconds,
        Some("UTC".into()),
    ))
}

pub fn timehash_neighbors(ca: &StringChunked) -> PolarsResult<StructChunked> {
    let mut before_ca = StringChunkedBuilder::new("before".into(), ca.len());
    let mut after_ca = StringChunkedBuilder::new("after".into(), ca.len());

    for value in ca.into_iter() {
        match value {
            Some(value) => {
                validate_timehash(value)?;
                let (before, after) =
                    timehash::neighbors(value).map_err(|e| PolarsError::ComputeError(e.into()))?;
                // The first and last window have no neighbor; upstream returns
                // an empty hash for it.
                match before.is_empty() {
                    true => before_ca.append_null(),
                    false => before_ca.append_value(before),
                }
                match after.is_empty() {
                    true => after_ca.append_null(),
                    false => after_ca.append_value(after),
                }
            }
            _ => {
                before_ca.append_null();
                after_ca.append_null();
            }
        }
    }
    let ser_before = before_ca.finish().into_series();
    let ser_after = after_ca.finish().into_series();

    StructChunked::from_series(
        ca.name().clone().into(),
        ca.len(),
        [ser_before, ser_after].iter(),
    )
}
