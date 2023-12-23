use crate::geohashers::{geohash_decoder, geohash_encoder, geohash_neighbors};
use polars::{chunked_array::ops::arity::try_ternary_elementwise, prelude::*};
use polars_core::datatypes::{
    DataType::{Float64, Struct, Utf8},
    Field,
};
use pyo3_polars::derive::polars_expr;
use sha2::{Digest, Sha224, Sha256, Sha384, Sha512};
use std::fmt::Write;
use std::str;
use wyhash::wyhash as real_wyhash;

fn sha256_hash(value: &str, output: &mut String) {
    let hash = Sha256::digest(value);
    write!(output, "{:x}", hash).unwrap()
}

fn sha512_hash(value: &str, output: &mut String) {
    let hash = Sha512::digest(value);
    write!(output, "{:x}", hash).unwrap()
}

fn sha384_hash(value: &str, output: &mut String) {
    let hash = Sha384::digest(value);
    write!(output, "{:x}", hash).unwrap()
}

fn sha224_hash(value: &str, output: &mut String) {
    let hash = Sha224::digest(value);
    write!(output, "{:x}", hash).unwrap()
}

fn wyhash_hash(value: Option<&str>) -> Option<u64> {
    value.map(|v| real_wyhash(v.as_bytes(), 0))
}

#[polars_expr(output_type=Utf8)]
fn sha256(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked = ca.apply_to_buffer(sha256_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=Utf8)]
fn sha512(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked = ca.apply_to_buffer(sha512_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=Utf8)]
fn sha384(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked = ca.apply_to_buffer(sha384_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=Utf8)]
fn sha224(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked = ca.apply_to_buffer(sha224_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt64)]
fn wyhash(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: ChunkedArray<UInt64Type> = ca.apply_generic(wyhash_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=Utf8)]
fn ghash_encode(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].struct_()?;
    let len = match inputs[1].dtype() {
        DataType::Int64 => inputs[1].clone(),
        DataType::Int32 => inputs[1].cast(&DataType::Int64)?,
        DataType::Int16 => inputs[1].cast(&DataType::Int64)?,
        DataType::Int8 => inputs[1].cast(&DataType::Int64)?,
        _ => polars_bail!(InvalidOperation:"Length input needs to be integer"),
    };
    let len = len.i64()?;

    let lat = ca.field_by_name("latitude")?;
    let long = ca.field_by_name("longitude")?;
    let lat = match lat.dtype() {
        DataType::Float32 => lat.cast(&DataType::Float64)?,
        DataType::Float64 => lat,
        _ => polars_bail!(InvalidOperation:"Latitude input needs to be float"),
    };

    let long = match long.dtype() {
        DataType::Float32 => long.cast(&DataType::Float64)?,
        DataType::Float64 => long,
        _ => polars_bail!(InvalidOperation:"Longitude input needs to be float"),
    };

    let ca_lat = lat.f64()?;
    let ca_long = long.f64()?;

    try_ternary_elementwise(ca_lat, ca_long, len, geohash_encoder)
        .map(|ca: Utf8Chunked| ca.into_series())
}

pub fn geohash_decode_output(_: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![
        Field::new("longitude", Float64),
        Field::new("latitude", Float64),
    ];
    Ok(Field::new("coordinates", Struct(v)))
}

#[polars_expr(output_type_func=geohash_decode_output)]
fn ghash_decode(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &ChunkedArray<Utf8Type> = inputs[0].utf8()?;

    Ok(geohash_decoder(ca)?.into_series())
}

pub fn geohash_neighbors_output(_: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![
        Field::new("n", Utf8),
        Field::new("ne", Utf8),
        Field::new("e", Utf8),
        Field::new("se", Utf8),
        Field::new("s", Utf8),
        Field::new("sw", Utf8),
        Field::new("w", Utf8),
        Field::new("nw", Utf8),
    ];
    Ok(Field::new("neighbors", Struct(v)))
}

#[polars_expr(output_type_func=geohash_neighbors_output)]
fn ghash_neighbors(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &Utf8Chunked = inputs[0].utf8()?;

    Ok(geohash_neighbors(ca)?.into_series())
}
