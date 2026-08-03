use polars::chunked_array::ops::arity::{unary_elementwise, unary_elementwise_values};
use polars::prelude::*;

/// Runs `op` on the bytes of each value of a String column or a Binary column.
///
/// A hash reads bytes. Therefore the two data types are different only in the method
/// to read an element. This function keeps that match in one place. Each expression
/// stays one line, all of them give the same error message, and `encode_rows` can send
/// its bytes to any of them.
///
/// A column with nulls uses the walk that omits them. The `_values` walk reads each
/// slot, and therefore a column of nulls cost as much as a full column. The `_values`
/// walk is still the faster walk if there are no nulls. Both walks are here.
pub fn hash_bytes<V, F, R>(s: &Series, op: F) -> PolarsResult<ChunkedArray<V>>
where
    V: PolarsDataType,
    F: Fn(&[u8]) -> R,
    V::Array: ArrayFromIter<R> + ArrayFromIter<Option<R>>,
{
    match (s.dtype(), s.null_count() == 0) {
        (DataType::String, true) => Ok(unary_elementwise_values(s.str()?, |v: &str| {
            op(v.as_bytes())
        })),
        (DataType::String, false) => Ok(unary_elementwise(s.str()?, |v: Option<&str>| {
            v.map(|v| op(v.as_bytes()))
        })),
        (DataType::Binary, true) => Ok(unary_elementwise_values(s.binary()?, &op)),
        (DataType::Binary, false) => Ok(unary_elementwise(s.binary()?, |v: Option<&[u8]>| {
            v.map(&op)
        })),
        (dtype, _) => polars_bail!(
            InvalidOperation: "expected `String` or `Binary` input, got `{}`", dtype
        ),
    }
}

/// The equivalent of [`hash_bytes`] for a digest in hexadecimal.
///
/// `op` writes to a buffer that each row uses again. It does not return a `String`.
/// Therefore a digest needs no memory of its own.
pub fn hash_bytes_into_string<F>(s: &Series, mut op: F) -> PolarsResult<StringChunked>
where
    F: FnMut(&[u8], &mut std::string::String),
{
    match s.dtype() {
        DataType::String => Ok(s
            .str()?
            .apply_into_string_amortized(|v, out| op(v.as_bytes(), out))),
        DataType::Binary => Ok(s.binary()?.apply_into_string_amortized(op)),
        dtype => polars_bail!(
            InvalidOperation: "expected `String` or `Binary` input, got `{}`", dtype
        ),
    }
}

/// The equivalent of [`hash_bytes`] for a digest of a set number of bytes.
///
/// [`hash_bytes`] cannot make a `BinaryChunked` from an array of bytes. It collects
/// what `ArrayFromIter` accepts, and that trait reads a slice or a `Vec`. An array
/// therefore needs a `Vec` of its own for each row. A builder takes the array as a
/// slice instead, and writes it straight into the one buffer of the column.
pub fn hash_bytes_into_binary<const N: usize, F>(s: &Series, op: F) -> PolarsResult<BinaryChunked>
where
    F: Fn(&[u8]) -> [u8; N],
{
    let mut builder = BinaryChunkedBuilder::new(s.name().clone(), s.len());
    match s.dtype() {
        DataType::String => {
            for value in s.str()?.iter() {
                match value {
                    Some(value) => builder.append_value(op(value.as_bytes())),
                    None => builder.append_null(),
                }
            }
        }
        DataType::Binary => {
            for value in s.binary()?.iter() {
                match value {
                    Some(value) => builder.append_value(op(value)),
                    None => builder.append_null(),
                }
            }
        }
        dtype => polars_bail!(
            InvalidOperation: "expected `String` or `Binary` input, got `{}`", dtype
        ),
    }
    Ok(builder.finish())
}

/// Coerce an integer argument to Int64. `_length_expr` already casts on the Python
/// side, so this only bites callers using `register_plugin_function` directly -- but
/// the three encoders used to disagree about which widths they accepted.
pub fn integer_arg(s: &Series, label: &str) -> PolarsResult<Series> {
    match s.dtype() {
        dtype if dtype.is_integer() => s.cast(&DataType::Int64),
        dtype => {
            polars_bail!(InvalidOperation: "{} input needs to be integer, got {}", label, dtype)
        }
    }
}

/// Coerce a float argument to Float64.
pub fn float_arg(s: &Series, label: &str) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Float32 => s.cast(&DataType::Float64),
        DataType::Float64 => Ok(s.clone()),
        dtype => polars_bail!(InvalidOperation: "{} input needs to be float, got {}", label, dtype),
    }
}

/// A length-1 argument broadcasts over the other operands; anything else is per-row.
/// The `unsafe` index is sound only under that length check, so both live here once
/// rather than being restated at every call site.
pub fn scalar_arg(ca: &Int64Chunked, label: &str) -> PolarsResult<Option<i64>> {
    if ca.len() != 1 {
        return Ok(None);
    }
    match unsafe { ca.get_unchecked(0) } {
        Some(value) => Ok(Some(value)),
        None => polars_bail!(ComputeError: "{} may not be null", label),
    }
}

/// Fan a string column out into a struct of string fields, one call to `f` per row.
/// The null row appends a null to every builder: correctness needs exactly one append
/// per builder per row, and keeping that in one place is the point -- a missed append
/// silently misaligns the whole column rather than failing.
pub fn string_struct<const N: usize>(
    ca: &StringChunked,
    names: [&str; N],
    f: impl Fn(&str) -> PolarsResult<[Option<String>; N]>,
) -> PolarsResult<StructChunked> {
    let mut builders: [StringChunkedBuilder; N] =
        std::array::from_fn(|i| StringChunkedBuilder::new(names[i].into(), ca.len()));

    for value in ca.iter() {
        match value {
            Some(value) => {
                for (builder, field) in builders.iter_mut().zip(f(value)?) {
                    match field {
                        Some(field) => builder.append_value(field),
                        None => builder.append_null(),
                    }
                }
            }
            _ => builders
                .iter_mut()
                .for_each(|builder| builder.append_null()),
        }
    }

    let fields: Vec<Series> = builders
        .into_iter()
        .map(|builder| builder.finish().into_series())
        .collect();
    StructChunked::from_series(ca.name().clone(), ca.len(), fields.iter())
}
