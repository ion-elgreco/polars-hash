use polars::prelude::*;

/// Encode a column into the exact bytes the Iceberg spec hashes for its `bucket(N)`
/// partition transform (Appendix B: 32-bit Hash Requirements).
pub fn encode_iceberg_bytes(s: &Series) -> PolarsResult<BinaryChunked> {
    match s.dtype() {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64 => {
            let ca = s.cast(&DataType::Int64)?;
            Ok(binary_from_i64(ca.i64()?))
        }
        DataType::Float32 | DataType::Float64 => {
            let ca = s.cast(&DataType::Float64)?;
            Ok(binary_from_f64(ca.f64()?))
        }
        DataType::String => {
            let ca = s.str()?;
            let mut builder = BinaryChunkedBuilder::new(s.name().clone(), s.len());
            for value in ca.iter() {
                match value {
                    Some(value) => builder.append_value(value.as_bytes()),
                    None => builder.append_null(),
                }
            }
            Ok(builder.finish())
        }
        DataType::Binary => Ok(s.binary()?.clone()),
        dtype => polars_bail!(
            InvalidOperation: "expected Boolean, Int8/16/32/64, Float32/64, String or Binary input, got `{}`", dtype
        ),
    }
}

/// Sign-extended two's-complement `i64`, 8 bytes little-endian. Covers Boolean (as
/// 0/1) and every signed integer width the spec lists, since a numeric cast to
/// `Int64` already sign-extends narrower integers and widens `Boolean` to 0/1.
fn binary_from_i64(ca: &Int64Chunked) -> BinaryChunked {
    let mut builder = BinaryChunkedBuilder::new(ca.name().clone(), ca.len());
    for value in ca.iter() {
        match value {
            Some(value) => builder.append_value(value.to_le_bytes()),
            None => builder.append_null(),
        }
    }
    builder.finish()
}

/// IEEE-754 bit pattern of an `f64`, 8 bytes little-endian. `-0.0 == 0.0` is `true`
/// under IEEE-754, so assigning the literal `0.0` on that branch normalizes away the
/// sign bit before it reaches `to_bits`. The spec requires `-0.0` and `0.0` to hash
/// identically.
fn binary_from_f64(ca: &Float64Chunked) -> BinaryChunked {
    let mut builder = BinaryChunkedBuilder::new(ca.name().clone(), ca.len());
    for value in ca.iter() {
        match value {
            Some(value) => {
                let normalized = if value == 0.0 { 0.0 } else { value };
                builder.append_value(normalized.to_bits().to_le_bytes());
            }
            None => builder.append_null(),
        }
    }
    builder.finish()
}
