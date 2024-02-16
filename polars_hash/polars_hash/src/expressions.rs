use crate::geohashers::{geohash_decoder, geohash_encoder, geohash_neighbors};
use crate::sha_hashers::*;
use polars::{
    chunked_array::ops::arity::{try_binary_elementwise, try_ternary_elementwise},
    prelude::*,
};
use polars_core::datatypes::{
    DataType::{Float64, String, Struct},
    Field,
};
use pyo3_polars::derive::polars_expr;
use std::fmt::Write;
use std::{str, string};
use wyhash::wyhash as real_wyhash;

pub fn blake3_hash(value: &str, output: &mut string::String) {
    let hash = blake3::hash(value.as_bytes());
    write!(output, "{}", hash).unwrap()
}

pub fn md5_hash(value: &str, output: &mut string::String) {
    let hash = md5::compute(value);
    write!(output, "{:x}", hash).unwrap()
}

fn wyhash_hash(value: Option<&str>) -> Option<u64> {
    value.map(|v| real_wyhash(v.as_bytes(), 0))
}

#[polars_expr(output_type=UInt64)]
fn wyhash(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: ChunkedArray<UInt64Type> = ca.apply_generic(wyhash_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn blake3(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(blake3_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha1(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(sha1_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn md5(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(md5_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha2_256(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(sha2_256_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha2_512(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(sha2_512_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha2_384(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(sha2_384_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha2_224(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(sha2_224_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_256(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(sha3_256_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_512(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(sha3_512_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_384(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(sha3_384_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_224(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(sha3_224_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
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

    let out: StringChunked = match len.len() {
        1 => match unsafe { len.get_unchecked(0) } {
            Some(len) => try_binary_elementwise(ca_lat, ca_long, |ca_lat_opt, ca_long_opt| {
                geohash_encoder(ca_lat_opt, ca_long_opt, Some(len))
            }),
            _ => Err(PolarsError::ComputeError(
                "Length may not be null".to_string().into(),
            )),
        },
        _ => try_ternary_elementwise(ca_lat, ca_long, len, geohash_encoder),
    }?;
    Ok(out.into_series())
}

pub fn geohash_decode_output(field: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![
        Field::new("longitude", Float64),
        Field::new("latitude", Float64),
    ];
    Ok(Field::new(field[0].name(), Struct(v)))
}

#[polars_expr(output_type_func=geohash_decode_output)]
fn ghash_decode(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    Ok(geohash_decoder(ca)?.into_series())
}

pub fn geohash_neighbors_output(field: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![
        Field::new("n", String),
        Field::new("ne", String),
        Field::new("e", String),
        Field::new("se", String),
        Field::new("s", String),
        Field::new("sw", String),
        Field::new("w", String),
        Field::new("nw", String),
    ];
    Ok(Field::new(field[0].name(), Struct(v)))
}

#[polars_expr(output_type_func=geohash_neighbors_output)]
fn ghash_neighbors(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    Ok(geohash_neighbors(ca)?.into_series())
}
