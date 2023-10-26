use polars::prelude::*;
// use polars_plan::dsl::FieldsMapper;
use pyo3_polars::derive::polars_expr;
// use serde::Deserialize;
use std::fmt::Write;
use sha256::digest;

fn sha256_hash(value: &str, output: &mut String) -> String {
    let hash = digest(value);
    write!(ouptut, "{hash}").unwrap()
}

#[polars_expr(output_type=Utf8)]
fn sha256(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked = ca.apply_to_buffer(|value, output| sha256_hash(value, output));
    Ok(out.into_series())
}