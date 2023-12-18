use geohash::{decode, encode, Coord};
use polars::{chunked_array::ops::arity::try_ternary_elementwise, prelude::*};
use polars_core::datatypes::{
    DataType::{Float64, Struct},
    Field,
};
use pyo3_polars::derive::polars_expr;
use sha256::digest;
use std::fmt::Write;
use wyhash::wyhash as real_wyhash;

fn sha256_hash(value: &str, output: &mut String) {
    let hash = digest(value);
    write!(output, "{hash}").unwrap()
}

fn wyhash_hash(value: Option<&str>) -> Option<u64> {
    match value {
        None => None,
        Some(v) => Some(real_wyhash(v.as_bytes().as_ref(), 0)),
    }
}

fn geohash_encoder(
    lat: Option<f64>,
    long: Option<f64>,
    len: Option<i64>,
) -> PolarsResult<Option<String>> {
    match (lat, long) {
        (Some(lat), Some(long)) => match len {
            Some(len) => Ok(Some(
                encode(Coord { x: long, y: lat }, len as usize)
                    .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?,
            )),
            _ => Err(PolarsError::ComputeError(
                "Length may not be null".to_string().into(),
            )),
        },
        _ => Err(PolarsError::ComputeError(
            format!(
                "Coordinates cannot be null. 
        Provided latitude: {:?}, longitude: {:?}",
                lat, long
            )
            .into(),
        )),
    }
}

#[polars_expr(output_type=Utf8)]
fn sha256(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked = ca.apply_to_buffer(|value, output| sha256_hash(value, output));
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
        &DataType::Int64 => inputs[1].clone(),
        &DataType::Int32 => inputs[1].cast(&DataType::Int64)?,
        &DataType::Int16 => inputs[1].cast(&DataType::Int64)?,
        &DataType::Int8 => inputs[1].cast(&DataType::Int64)?,
        _ => polars_bail!(InvalidOperation:"Length input needs to be integer"),
    };
    let len = len.i64()?;

    let lat = ca.field_by_name("lat")?;
    let long = ca.field_by_name("long")?;
    let lat = match lat.dtype() {
        &DataType::Float32 => lat.cast(&DataType::Float64)?,
        &DataType::Float64 => lat,
        _ => polars_bail!(InvalidOperation:"Length input needs to be integer"),
    };

    let long = match long.dtype() {
        &DataType::Float32 => long.cast(&DataType::Float64)?,
        &DataType::Float64 => long,
        _ => polars_bail!(InvalidOperation:"Length input needs to be integer"),
    };

    let ca_lat = lat.f64()?;
    let ca_long = long.f64()?;

    try_ternary_elementwise(ca_lat, ca_long, len, geohash_encoder)
        .map(|ca: Utf8Chunked| ca.into_series())
}

pub fn geohash_output(_: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![Field::new("latitude", Float64), Field::new("longitude", Float64)];
    Ok(Field::new("coordinates", Struct(v)))
}

#[polars_expr(output_type_func=geohash_output)]
fn ghash_decode(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;

    let mut x_vec: Vec<Option<f64>> = Vec::new(); //latitude
    let mut y_vec: Vec<Option<f64>> = Vec::new(); //longitude

    for value in ca.into_iter() {
        match value {
            Some(value) => {
                let (cords, _, _) =
                    decode(value).map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
                let (x_value, y_value) = cords.x_y();
                x_vec.push(Some(x_value));
                y_vec.push(Some(y_value));
            }
            _ => {
                x_vec.push(None);
                y_vec.push(None);
            }
        }
    }

    let ca_lat = Series::new("latitude", x_vec);
    let ca_long = Series::new("longitude", y_vec);
    let out = StructChunked::new("coordinates", &[ca_lat, ca_long])?;

    Ok(out.into_series())
}
