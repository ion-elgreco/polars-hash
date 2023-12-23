use geohash::{decode, encode, neighbors, Coord};
use polars::{chunked_array::ops::arity::try_ternary_elementwise, prelude::*};
use polars_core::datatypes::{
    DataType::{Float64, Struct, Utf8},
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

fn geohash_decoder(ca: &ChunkedArray<Utf8Type>) -> PolarsResult<StructChunked> {
    let mut longitude: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("longitude", ca.len());
    let mut latitude: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("latitude", ca.len());

    for value in ca.into_iter() {
        match value {
            Some(value) => {
                let (cords, _, _) =
                    decode(value).map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
                let (x_value, y_value) = cords.x_y();
                longitude.append_value(x_value);
                latitude.append_value(y_value);
            }
            _ => {
                longitude.append_null();
                latitude.append_null();
            }
        }
    }
    let ser_long = longitude.finish().into_series();
    let ser_lat = latitude.finish().into_series();
    Ok(StructChunked::new("coordinates", &[ser_long, ser_lat])?)
}

fn geohash_neighbors(ca: &Utf8Chunked) -> PolarsResult<StructChunked> {
    let mut n_ca = Utf8ChunkedBuilder::new("n", ca.len(), ca.get_values_size());
    let mut ne_ca = Utf8ChunkedBuilder::new("ne", ca.len(), ca.get_values_size());
    let mut e_ca = Utf8ChunkedBuilder::new("e", ca.len(), ca.get_values_size());
    let mut se_ca = Utf8ChunkedBuilder::new("se", ca.len(), ca.get_values_size());
    let mut s_ca = Utf8ChunkedBuilder::new("s", ca.len(), ca.get_values_size());
    let mut sw_ca = Utf8ChunkedBuilder::new("sw", ca.len(), ca.get_values_size());
    let mut w_ca = Utf8ChunkedBuilder::new("w", ca.len(), ca.get_values_size());
    let mut nw_ca = Utf8ChunkedBuilder::new("nw", ca.len(), ca.get_values_size());

    for value in ca.into_iter() {
        match value {
            Some(value) => {
                let neighbors_result = neighbors(value)
                    .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
                n_ca.append_value(neighbors_result.n);
                ne_ca.append_value(neighbors_result.ne);
                e_ca.append_value(neighbors_result.e);
                se_ca.append_value(neighbors_result.se);
                s_ca.append_value(neighbors_result.s);
                sw_ca.append_value(neighbors_result.sw);
                w_ca.append_value(neighbors_result.w);
                nw_ca.append_value(neighbors_result.nw);
            }
            _ => {
                n_ca.append_null();
                ne_ca.append_null();
                e_ca.append_null();
                se_ca.append_null();
                s_ca.append_null();
                sw_ca.append_null();
                w_ca.append_null();
                nw_ca.append_null();
            }
        }
    }
    let ser_north = n_ca.finish().into_series();
    let ser_north_east = ne_ca.finish().into_series();
    let ser_east = e_ca.finish().into_series();
    let ser_south_east = se_ca.finish().into_series();
    let ser_south = s_ca.finish().into_series();
    let ser_south_west = sw_ca.finish().into_series();
    let ser_west = w_ca.finish().into_series();
    let ser_north_west = nw_ca.finish().into_series();

    Ok(StructChunked::new(
        "neighbors",
        &[
            ser_north,
            ser_north_east,
            ser_east,
            ser_south_east,
            ser_south,
            ser_south_west,
            ser_west,
            ser_north_west,
        ],
    )?)
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

    let lat = ca.field_by_name("latitude")?;
    let long = ca.field_by_name("longitude")?;
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
