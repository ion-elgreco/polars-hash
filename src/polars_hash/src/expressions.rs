use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use sha256::digest;
use std::fmt::Write;

fn sha256_hash(value: &str, output: &mut String) {
    let hash = digest(value);
    write!(output, "{hash}").unwrap()
}

#[polars_expr(output_type=Utf8)]
fn sha256(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked = ca.apply_to_buffer(|value, output| sha256_hash(value, output));
    Ok(out.into_series())
}
