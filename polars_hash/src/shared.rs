use polars::chunked_array::ops::arity::{unary_elementwise, unary_elementwise_values};
use polars::prelude::*;

/// Run `op` over the bytes of every value, for either a String or a Binary column.
///
/// A hash reads bytes, so the two dtypes differ only in how an element is reached.
/// Keeping the match here lets each expression stay a single line, gives them all
/// one error message, and is what lets `encode_rows` feed any of them.
///
/// The walk over a column that holds nulls skips them. The `_values` walk visits
/// every slot, so a mostly-null column used to cost what a full one does and to throw
/// away an allocation per null for the digests that return a `Vec`. It is still the
/// faster of the two when there is nothing to skip, so both are here.
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

/// The [`hash_bytes`] counterpart for a digest written out as hex.
///
/// `op` appends to a buffer that is reused across rows rather than returning a
/// `String`, so a digest costs no allocation of its own.
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
