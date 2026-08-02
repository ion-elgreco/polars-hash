use polars::prelude::*;

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

    for value in ca.into_iter() {
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
