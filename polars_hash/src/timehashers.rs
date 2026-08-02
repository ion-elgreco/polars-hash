use polars::chunked_array::ops::arity::{try_unary_elementwise, unary_elementwise};
use polars::prelude::*;
use timeharsh::timehash;

/// End of the timehash range (2098-01-01). Past it every timestamp saturates to
/// the same hash, so encoding errors rather than bin to the wrong window.
const MAX_EPOCH_SECONDS: f64 = 4_039_372_800.0;

/// Past ~18 characters the f64 interval stops splitting and adds no information.
const MAX_PRECISION: i64 = 32;

/// `before`/`after` index the hash by byte but walk it by character, so a
/// multi-byte character panics there before it is ever rejected.
fn validate_timehash(value: &str) -> PolarsResult<()> {
    if value.is_empty() {
        polars_bail!(ComputeError: "timehash may not be empty")
    }
    match value.chars().find(|c| !matches!(c, '0' | '1' | 'a'..='f')) {
        Some(c) => {
            polars_bail!(ComputeError: "invalid timehash character '{}' in '{}'", c, value)
        }
        None => Ok(()),
    }
}

/// Datetime and Date carry their unit in the dtype; other numerics are taken as
/// epoch seconds as-is. Float32 is refused rather than widened: it keeps about 7
/// significant digits, so near 1.5e9 it can only space values 128 seconds apart --
/// wider than every window up to precision 5, and the error is silent once widened.
pub fn epoch_seconds(s: &Series) -> PolarsResult<Float64Chunked> {
    match s.dtype() {
        DataType::Datetime(time_unit, _) => {
            let scale = match time_unit {
                TimeUnit::Nanoseconds => 1e9,
                TimeUnit::Microseconds => 1e6,
                TimeUnit::Milliseconds => 1e3,
            };
            let physical = s.cast(&DataType::Int64)?;
            Ok(unary_elementwise(physical.i64()?, |v| {
                v.map(|v| v as f64 / scale)
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

pub fn timehash_encoder(
    seconds: Option<f64>,
    precision: Option<i64>,
) -> PolarsResult<Option<String>> {
    match seconds {
        Some(seconds) => match precision {
            Some(precision) => {
                if !(1..=MAX_PRECISION).contains(&precision) {
                    polars_bail!(
                        InvalidOperation:
                        "expected precision between 1 and {}, got {}", MAX_PRECISION, precision
                    )
                }
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

/// Decodes to the midpoint of the hashed window.
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

    out.into_series()
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
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
