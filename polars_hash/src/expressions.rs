use crate::geohashers::{geohash_decoder, geohash_encoder, geohash_neighbors};
use crate::h3::h3_encoder;
use crate::murmurhash_hashers::*;
use crate::sha_hashers::*;
use crate::xxhash_hashers::*;
use polars::{
    chunked_array::ops::arity::{
        try_binary_elementwise, try_ternary_elementwise, unary_elementwise,
    },
    prelude::*,
};

use polars_core::datatypes::{
    DataType::{Float64, String, Struct},
    Field,
};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::fmt::Write;
use std::{str, string};
use wyhash::wyhash as real_wyhash;

#[derive(Deserialize)]
struct SeedKwargs32bit {
    seed: u32,
}

#[derive(Deserialize)]
struct SeedKwargs64bit {
    seed: u64,
}

#[derive(Deserialize)]
struct LengthKwargs {
    length: usize,
}

pub fn blake3_hash_str(value: &str, output: &mut string::String) {
    let hash = blake3::hash(value.as_bytes());
    write!(output, "{}", hash).unwrap()
}

pub fn blake3_hash_bytes(value: Option<&[u8]>) -> Option<string::String> {
    let hash = blake3::hash(value.unwrap());
    let res = format!("{}", hash);
    Some(res)
}

pub fn md5_hash_str(value: &str, output: &mut string::String) {
    let hash = md5::compute(value);
    write!(output, "{:x}", hash).unwrap()
}

pub fn md5_hash_bytes(value: Option<&[u8]>) -> Option<string::String> {
    let hash = md5::compute(value.unwrap());
    let res = format!("{:x}", hash);
    Some(res)
}

fn wyhash_hash_str(value: Option<&str>) -> Option<u64> {
    value.map(|v| real_wyhash(v.as_bytes(), 0))
}

fn wyhash_hash_bytes(value: Option<&[u8]>) -> Option<u64> {
    value.map(|v| real_wyhash(v, 0))
}

#[polars_expr(output_type=UInt64)]
fn wyhash(inputs: &[Series]) -> PolarsResult<Series> {
    let s = inputs.get(0).expect("no series received");

    match s.dtype() {
        DataType::String => {
            let ca = s.str()?;
            let out: ChunkedArray<UInt64Type> = unary_elementwise(ca, wyhash_hash_str);
            Ok(out.into_series())
        }
        DataType::Binary => {
            let ca = s.binary()?;
            let out: ChunkedArray<UInt64Type> = unary_elementwise(ca, wyhash_hash_bytes);
            Ok(out.into_series())
        }
        _ => Err(PolarsError::InvalidOperation(
            "wyhash only works on strings or binary data".into(),
        )),
    }
}

#[polars_expr(output_type=String)]
fn blake3(inputs: &[Series]) -> PolarsResult<Series> {
    let s = inputs.get(0).expect("no series received");

    match s.dtype() {
        DataType::String => {
            let ca = s.str()?;
            let out: StringChunked = ca.apply_into_string_amortized(blake3_hash_str);
            Ok(out.into_series())
        }
        DataType::Binary => {
            let ca = s.binary()?;
            let out: StringChunked = unary_elementwise(ca, blake3_hash_bytes);
            Ok(out.into_series())
        }
        _ => Err(PolarsError::InvalidOperation(
            "blake3 only works on strings or binary data".into(),
        )),
    }
}

#[polars_expr(output_type=String)]
fn md5(inputs: &[Series]) -> PolarsResult<Series> {
    let s = inputs.get(0).expect("no series received");

    match s.dtype() {
        DataType::String => {
            let ca = s.str()?;
            let out: StringChunked = ca.apply_into_string_amortized(md5_hash_str);
            Ok(out.into_series())
        }
        DataType::Binary => {
            let ca = s.binary()?;
            let out: StringChunked = unary_elementwise(ca, md5_hash_bytes);
            Ok(out.into_series())
        }
        _ => Err(PolarsError::InvalidOperation(
            "md5 only works on strings or binary data".into(),
        )),
    }
}

#[polars_expr(output_type=String)]
fn sha1(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(sha1_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha2_256(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(sha2_256_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha2_512(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(sha2_512_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha2_384(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(sha2_384_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha2_224(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(sha2_224_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_256(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(sha3_256_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_512(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(sha3_512_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_384(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(sha3_384_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_224(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(sha3_224_hash);
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_shake128(inputs: &[Series], kwargs: LengthKwargs) -> PolarsResult<Series> {

    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(|value: &str, output: &mut string::String| {
        sha3_shake128_hash(value, output, kwargs.length)
    });

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

#[polars_expr(output_type=String)]
fn h3_encode(inputs: &[Series]) -> PolarsResult<Series> {
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
                h3_encoder(ca_lat_opt, ca_long_opt, Some(len))
            }),
            _ => Err(PolarsError::ComputeError(
                "Length may not be null".to_string().into(),
            )),
        },
        _ => try_ternary_elementwise(ca_lat, ca_long, len, h3_encoder),
    }?;
    Ok(out.into_series())
}

pub fn geohash_decode_output(field: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![
        Field::new("longitude".into(), Float64),
        Field::new("latitude".into(), Float64),
    ];
    Ok(Field::new(field[0].name().clone(), Struct(v)))
}

#[polars_expr(output_type_func=geohash_decode_output)]
fn ghash_decode(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    Ok(geohash_decoder(ca)?.into_series())
}

pub fn geohash_neighbors_output(field: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![
        Field::new("n".into(), String),
        Field::new("ne".into(), String),
        Field::new("e".into(), String),
        Field::new("se".into(), String),
        Field::new("s".into(), String),
        Field::new("sw".into(), String),
        Field::new("w".into(), String),
        Field::new("nw".into(), String),
    ];
    Ok(Field::new(field[0].name().clone(), Struct(v)))
}

#[polars_expr(output_type_func=geohash_neighbors_output)]
fn ghash_neighbors(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    Ok(geohash_neighbors(ca)?.into_series())
}

#[polars_expr(output_type=UInt32)]
fn murmur32(inputs: &[Series], kwargs: SeedKwargs32bit) -> PolarsResult<Series> {
    let seeded_hash_function = |v| murmurhash3_32(v, kwargs.seed);

    let ca = inputs[0].str()?;
    let out: ChunkedArray<UInt32Type> = unary_elementwise(ca, seeded_hash_function);
    Ok(out.into_series())
}

#[polars_expr(output_type=Binary)]
fn murmur128(inputs: &[Series], kwargs: SeedKwargs32bit) -> PolarsResult<Series> {
    let seeded_hash_function = |v| murmurhash3_128(v, kwargs.seed);

    let ca = inputs[0].str()?;
    let out: ChunkedArray<BinaryType> = unary_elementwise(ca, seeded_hash_function);
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt32)]
fn xxhash32(inputs: &[Series], kwargs: SeedKwargs32bit) -> PolarsResult<Series> {
    let seeded_hash_function = |v| xxhash_32(v, kwargs.seed);

    let ca = inputs[0].str()?;
    let out: ChunkedArray<UInt32Type> = unary_elementwise(ca, seeded_hash_function);
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt64)]
fn xxhash64(inputs: &[Series], kwargs: SeedKwargs64bit) -> PolarsResult<Series> {
    let seeded_hash_function = |v| xxhash_64(v, kwargs.seed);

    let ca = inputs[0].str()?;
    let out: ChunkedArray<UInt64Type> = unary_elementwise(ca, seeded_hash_function);
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt64)]
fn xxh3_64(inputs: &[Series], kwargs: SeedKwargs64bit) -> PolarsResult<Series> {
    let seeded_hash_function = |v| xxhash3_64(v, kwargs.seed);

    let ca = inputs[0].str()?;
    let out: ChunkedArray<UInt64Type> = unary_elementwise(ca, seeded_hash_function);
    Ok(out.into_series())
}

#[polars_expr(output_type=Binary)]
fn xxh3_128(inputs: &[Series], kwargs: SeedKwargs64bit) -> PolarsResult<Series> {
    let seeded_hash_function = |v| xxhash3_128(v, kwargs.seed);

    let ca = inputs[0].str()?;
    let out: ChunkedArray<BinaryType> = unary_elementwise(ca, seeded_hash_function);
    Ok(out.into_series())
}
