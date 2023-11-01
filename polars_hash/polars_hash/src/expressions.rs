use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use sha256::digest;
use std::fmt::Write;
use wyhash::wyhash as real_wyhash;

fn sha256_hash(value: &str, output: &mut String) {
    let hash = digest(value);
    write!(output, "{hash}").unwrap()
}

// Return this as uint instead
fn wyhash_hash(value: &str, output: &mut String) {
    let hash = real_wyhash(value.as_bytes().as_ref(), 0);
    write!(output, "{hash}").unwrap()
}

#[polars_expr(output_type=Utf8)]
fn sha256(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked = ca.apply_to_buffer(|value, output| sha256_hash(value, output));
    Ok(out.into_series())
}

#[polars_expr(output_type=Utf8)]
fn wyhash(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked = ca.apply_to_buffer(|value, output| wyhash_hash(value, output));
    Ok(out.into_series())
}
