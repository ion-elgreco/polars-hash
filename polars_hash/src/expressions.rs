use crate::geohashers::{geohash_decoder, geohash_encoder, geohash_neighbors};
use crate::h3::h3_encoder;
use crate::hmac_hashers::*;
use crate::murmurhash_hashers::*;
use crate::sha_hashers::*;
use crate::shared::{float_arg, hash_bytes, hash_bytes_into_string, integer_arg, scalar_arg};
use crate::timehashers::{
    epoch_seconds, hash_column, timehash_decoder, timehash_encoder, timehash_neighbors,
    validate_precision,
};
use crate::xxhash_hashers::*;
use hmac::KeyInit;
use polars::{
    chunked_array::ops::arity::{
        try_binary_elementwise, try_ternary_elementwise, try_unary_elementwise,
    },
    prelude::*,
};

use polars_core::datatypes::{
    DataType::{Datetime, Float64, String, Struct},
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

/// Kwargs arrive as a pickle, whose integers are `i64`, so the Python side sends a
/// `u64` seed as its two's-complement counterpart and `as u64` puts it back.
#[derive(Deserialize)]
struct SeedKwargs64bit {
    seed: i64,
}

#[derive(Deserialize)]
struct LengthKwargs {
    length: usize,
}

#[derive(Deserialize)]
struct HmacKwargs {
    key: string::String,
}

#[derive(Deserialize)]
struct StrictKwargs {
    strict: bool,
}

#[derive(Deserialize)]
struct VersionKwargs {
    version: u64,
}

pub fn blake3_hash(value: &[u8], output: &mut string::String) {
    write!(output, "{}", blake3::hash(value)).unwrap()
}

pub fn md5_hash(value: &[u8], output: &mut string::String) {
    write!(output, "{:x}", md5::compute(value)).unwrap()
}

// `cityhasher::hash` picks the algorithm from its return type alone, so the turbofish
// is what makes these CityHash32 and CityHash64 rather than each other.
//
// The two crates name different CityHash releases — `cityhasher` says v1.1, and
// `cityhash_110_128` is v1.1.0 — but the docs promise v1.1.1 for all three. That holds:
// upstream NEWS for v1.1.1 reads "No changes to any of the functions, unless you had
// been using CityHash32()", and that change was the signed-`char` fix which `cityhasher`
// applies as `*v as i8`.
fn cityhash_32(value: &[u8]) -> u32 {
    cityhasher::hash::<u32>(value)
}

fn cityhash_64(value: &[u8]) -> u64 {
    cityhasher::hash::<u64>(value)
}

fn cityhash_64_with_seed(value: &[u8], seed: u64) -> u64 {
    cityhasher::hash_with_seed::<u64>(value, seed)
}

fn cityhash_128(value: &[u8]) -> u128 {
    cityhash_rs::cityhash_110_128(value)
}

// gxhash seeds are `i64`, the type kwargs already travel in, so these pass
// `kwargs.seed` through where the other hashers cast it back to `u64`.
fn gxhash_32(value: &[u8], seed: i64) -> u32 {
    gxhash::gxhash32(value, seed)
}

fn gxhash_64(value: &[u8], seed: i64) -> u64 {
    gxhash::gxhash64(value, seed)
}

fn gxhash_128(value: &[u8], seed: i64) -> u128 {
    gxhash::gxhash128(value, seed)
}

#[polars_expr(output_type=UInt32)]
fn farmhash32(inputs: &[Series]) -> PolarsResult<Series> {
    let out: UInt32Chunked = hash_bytes(&inputs[0], farmhash::fingerprint32)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt64)]
fn farmhash64(inputs: &[Series]) -> PolarsResult<Series> {
    let out: UInt64Chunked = hash_bytes(&inputs[0], farmhash::fingerprint64)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt32)]
fn cityhash32(inputs: &[Series]) -> PolarsResult<Series> {
    let out: UInt32Chunked = hash_bytes(&inputs[0], cityhash_32)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt64)]
fn cityhash64(inputs: &[Series]) -> PolarsResult<Series> {
    let out: UInt64Chunked = hash_bytes(&inputs[0], cityhash_64)?;
    Ok(out.into_series())
}

/// Its own expression rather than a seed defaulting to 0, because
/// `CityHash64WithSeed(v, 0)` is a different hash from `CityHash64(v)`.
#[polars_expr(output_type=UInt64)]
fn cityhash64_with_seed(inputs: &[Series], kwargs: SeedKwargs64bit) -> PolarsResult<Series> {
    let seed = kwargs.seed as u64;
    let out: UInt64Chunked = hash_bytes(&inputs[0], |v| cityhash_64_with_seed(v, seed))?;
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt128)]
fn cityhash128(inputs: &[Series]) -> PolarsResult<Series> {
    let out: UInt128Chunked = hash_bytes(&inputs[0], cityhash_128)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt32)]
fn gxhash32(inputs: &[Series], kwargs: SeedKwargs64bit) -> PolarsResult<Series> {
    let seed = kwargs.seed;
    let out: UInt32Chunked = hash_bytes(&inputs[0], |v| gxhash_32(v, seed))?;
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt64)]
fn gxhash64(inputs: &[Series], kwargs: SeedKwargs64bit) -> PolarsResult<Series> {
    let seed = kwargs.seed;
    let out: UInt64Chunked = hash_bytes(&inputs[0], |v| gxhash_64(v, seed))?;
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt128)]
fn gxhash128(inputs: &[Series], kwargs: SeedKwargs64bit) -> PolarsResult<Series> {
    let seed = kwargs.seed;
    let out: UInt128Chunked = hash_bytes(&inputs[0], |v| gxhash_128(v, seed))?;
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt64)]
fn wyhash(inputs: &[Series]) -> PolarsResult<Series> {
    let out: UInt64Chunked = hash_bytes(&inputs[0], |v| real_wyhash(v, 0))?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn blake3(inputs: &[Series]) -> PolarsResult<Series> {
    let out = hash_bytes_into_string(&inputs[0], blake3_hash)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn md5(inputs: &[Series]) -> PolarsResult<Series> {
    let out = hash_bytes_into_string(&inputs[0], md5_hash)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha1(inputs: &[Series]) -> PolarsResult<Series> {
    let out = hash_bytes_into_string(&inputs[0], sha1_hash)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha2_256(inputs: &[Series]) -> PolarsResult<Series> {
    let out = hash_bytes_into_string(&inputs[0], sha2_256_hash)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha2_512(inputs: &[Series]) -> PolarsResult<Series> {
    let out = hash_bytes_into_string(&inputs[0], sha2_512_hash)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha2_384(inputs: &[Series]) -> PolarsResult<Series> {
    let out = hash_bytes_into_string(&inputs[0], sha2_384_hash)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha2_224(inputs: &[Series]) -> PolarsResult<Series> {
    let out = hash_bytes_into_string(&inputs[0], sha2_224_hash)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_256(inputs: &[Series]) -> PolarsResult<Series> {
    let out = hash_bytes_into_string(&inputs[0], sha3_256_hash)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_512(inputs: &[Series]) -> PolarsResult<Series> {
    let out = hash_bytes_into_string(&inputs[0], sha3_512_hash)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_384(inputs: &[Series]) -> PolarsResult<Series> {
    let out = hash_bytes_into_string(&inputs[0], sha3_384_hash)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_224(inputs: &[Series]) -> PolarsResult<Series> {
    let out = hash_bytes_into_string(&inputs[0], sha3_224_hash)?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn sha3_shake128(inputs: &[Series], kwargs: LengthKwargs) -> PolarsResult<Series> {
    let out = hash_bytes_into_string(&inputs[0], |value, output| {
        sha3_shake128_hash(value, output, kwargs.length)
    })?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn hmac_sha256(inputs: &[Series], kwargs: HmacKwargs) -> PolarsResult<Series> {
    let keyed_mac = HmacSha256::new_from_slice(kwargs.key.as_bytes())
        .map_err(|e| PolarsError::ComputeError(format!("invalid HMAC key: {e}").into()))?;
    let out = hash_bytes_into_string(&inputs[0], |value, output| {
        hmac_sha256_hash(value, output, &keyed_mac)
    })?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn ghash_encode(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].struct_()?;
    let len = integer_arg(&inputs[1], "Length")?;
    let len = len.i64()?;
    let lat = float_arg(&ca.field_by_name("latitude")?, "Latitude")?;
    let long = float_arg(&ca.field_by_name("longitude")?, "Longitude")?;
    let (ca_lat, ca_long) = (lat.f64()?, long.f64()?);

    let out: StringChunked = match scalar_arg(len, "Length")? {
        Some(len) => try_binary_elementwise(ca_lat, ca_long, |lat, long| {
            geohash_encoder(lat, long, Some(len))
        }),
        None => try_ternary_elementwise(ca_lat, ca_long, len, geohash_encoder),
    }?;
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn h3_encode(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].struct_()?;
    let len = integer_arg(&inputs[1], "Length")?;
    let len = len.i64()?;
    let lat = float_arg(&ca.field_by_name("latitude")?, "Latitude")?;
    let long = float_arg(&ca.field_by_name("longitude")?, "Longitude")?;
    let (ca_lat, ca_long) = (lat.f64()?, long.f64()?);

    let out: StringChunked = match scalar_arg(len, "Length")? {
        Some(len) => try_binary_elementwise(ca_lat, ca_long, |lat, long| {
            h3_encoder(lat, long, Some(len))
        }),
        None => try_ternary_elementwise(ca_lat, ca_long, len, h3_encoder),
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

#[polars_expr(output_type=String)]
fn thash_encode(inputs: &[Series], kwargs: StrictKwargs) -> PolarsResult<Series> {
    let seconds = epoch_seconds(&inputs[0])?;
    let precision = integer_arg(&inputs[1], "Precision")?;
    let precision = precision.i64()?;
    let strict = kwargs.strict;

    let out: StringChunked = match scalar_arg(precision, "Precision")? {
        Some(precision) => {
            validate_precision(precision)?;
            try_unary_elementwise(&seconds, |seconds_opt| {
                timehash_encoder(seconds_opt, Some(precision), strict)
            })
        }
        None if seconds.len() == 1 => {
            let seconds = unsafe { seconds.get_unchecked(0) };
            try_unary_elementwise(precision, |precision_opt| {
                timehash_encoder(seconds, precision_opt, strict)
            })
            .map(|out| out.with_name(inputs[0].name().clone()))
        }
        None if seconds.len() == precision.len() => {
            try_binary_elementwise(&seconds, precision, |seconds_opt, precision_opt| {
                timehash_encoder(seconds_opt, precision_opt, strict)
            })
        }
        None => polars_bail!(
            ShapeMismatch:
            "timestamp column has length {} and precision has length {}, expected equal lengths or a scalar",
            seconds.len(), precision.len()
        ),
    }?;
    Ok(out.into_series())
}

pub fn timehash_decode_output(field: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        field[0].name().clone(),
        Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC)),
    ))
}

#[polars_expr(output_type_func=timehash_decode_output)]
fn thash_decode(inputs: &[Series]) -> PolarsResult<Series> {
    let s = hash_column(&inputs[0])?;

    timehash_decoder(s.str()?)
}

pub fn timehash_neighbors_output(field: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![
        Field::new("before".into(), String),
        Field::new("after".into(), String),
    ];
    Ok(Field::new(field[0].name().clone(), Struct(v)))
}

#[polars_expr(output_type_func=timehash_neighbors_output)]
fn thash_neighbors(inputs: &[Series]) -> PolarsResult<Series> {
    let s = hash_column(&inputs[0])?;

    Ok(timehash_neighbors(s.str()?)?.into_series())
}

#[polars_expr(output_type=UInt32)]
fn murmur32(inputs: &[Series], kwargs: SeedKwargs32bit) -> PolarsResult<Series> {
    let seed = kwargs.seed;
    let out: UInt32Chunked = hash_bytes(&inputs[0], |v| murmurhash3_32(v, seed))?;
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt128)]
fn murmur128(inputs: &[Series], kwargs: SeedKwargs32bit) -> PolarsResult<Series> {
    let seed = kwargs.seed;
    let out: UInt128Chunked = hash_bytes(&inputs[0], |v| murmurhash3_128(v, seed))?;
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt32)]
fn xxhash32(inputs: &[Series], kwargs: SeedKwargs32bit) -> PolarsResult<Series> {
    let seed = kwargs.seed;
    let out: UInt32Chunked = hash_bytes(&inputs[0], |v| xxhash_32(v, seed))?;
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt64)]
fn xxhash64(inputs: &[Series], kwargs: SeedKwargs64bit) -> PolarsResult<Series> {
    let seed = kwargs.seed as u64;
    let out: UInt64Chunked = hash_bytes(&inputs[0], |v| xxhash_64(v, seed))?;
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt64)]
fn xxh3_64(inputs: &[Series], kwargs: SeedKwargs64bit) -> PolarsResult<Series> {
    let seed = kwargs.seed as u64;
    let out: UInt64Chunked = hash_bytes(&inputs[0], |v| xxhash3_64(v, seed))?;
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt128)]
fn xxh3_128(inputs: &[Series], kwargs: SeedKwargs64bit) -> PolarsResult<Series> {
    let seed = kwargs.seed as u64;
    let out: UInt128Chunked = hash_bytes(&inputs[0], |v| xxhash3_128(v, seed))?;
    Ok(out.into_series())
}

#[polars_expr(output_type=Binary)]
fn encode_rows(inputs: &[Series], kwargs: VersionKwargs) -> PolarsResult<Series> {
    Ok(crate::row_encode::encode_rows(inputs, kwargs.version)?.into_series())
}

#[polars_expr(output_type=String)]
fn uuid5(inputs: &[Series]) -> PolarsResult<Series> {
    let namespace_str = inputs[1].str()?;
    let ns_value = namespace_str
        .get(0)
        .ok_or_else(|| PolarsError::ComputeError("Namespace must be provided".into()))?;

    let namespace = match ns_value.to_lowercase().as_str() {
        "dns" => uuid::Uuid::NAMESPACE_DNS,
        "url" => uuid::Uuid::NAMESPACE_URL,
        "oid" => uuid::Uuid::NAMESPACE_OID,
        "x500" => uuid::Uuid::NAMESPACE_X500,
        _ => uuid::Uuid::parse_str(ns_value).map_err(|e| {
            PolarsError::ComputeError(format!("Invalid namespace '{}': {}", ns_value, e).into())
        })?,
    };

    let out = hash_bytes_into_string(&inputs[0], |value, output| {
        output.push_str(
            &uuid::Uuid::new_v5(&namespace, value)
                .hyphenated()
                .to_string(),
        )
    })?;
    Ok(out.into_series())
}

/// A null second value falls back to `default`, which is `""` when the caller gave none.
fn uuid5_of_pair(a: Option<&str>, b: Option<&str>, default: &str) -> Option<string::String> {
    a.map(|a| {
        let b = b.unwrap_or(default);
        let mut input = string::String::with_capacity(a.len() + b.len());
        input.push_str(a);
        input.push_str(b);
        uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_DNS, input.as_bytes())
            .hyphenated()
            .to_string()
    })
}

/// Zipping the two columns would drop rows, so a length-1 column is broadcast and
/// any other length mismatch is an error.
fn uuid5_concat_impl(
    col1: &StringChunked,
    col2: &StringChunked,
    default: &str,
) -> PolarsResult<StringChunked> {
    let out: StringChunked = match (col1.len(), col2.len()) {
        (_, 1) => {
            let b = col2.get(0);
            col1.iter().map(|a| uuid5_of_pair(a, b, default)).collect()
        }
        (1, _) => {
            let a = col1.get(0);
            col2.iter().map(|b| uuid5_of_pair(a, b, default)).collect()
        }
        (len1, len2) if len1 == len2 => col1
            .iter()
            .zip(col2.iter())
            .map(|(a, b)| uuid5_of_pair(a, b, default))
            .collect(),
        (len1, len2) => polars_bail!(
            ShapeMismatch:
            "first column has length {} and second column has length {}, expected equal lengths or a scalar",
            len1, len2
        ),
    };
    Ok(out.with_name(col1.name().clone()))
}

#[polars_expr(output_type=String)]
fn uuid5_concat(inputs: &[Series]) -> PolarsResult<Series> {
    let col1 = inputs[0].str()?;
    let col2 = inputs[1].str()?;

    Ok(uuid5_concat_impl(col1, col2, "")?.into_series())
}

#[polars_expr(output_type=String)]
fn uuid5_concat_default(inputs: &[Series]) -> PolarsResult<Series> {
    let col1 = inputs[0].str()?;
    let col2_casted = inputs[1].cast(&DataType::String)?;
    let col2 = col2_casted.str()?;
    let default = inputs[2].str()?;
    let default_val = default
        .get(0)
        .ok_or_else(|| PolarsError::ComputeError("Default value may not be null".into()))?;

    Ok(uuid5_concat_impl(col1, col2, default_val)?.into_series())
}
