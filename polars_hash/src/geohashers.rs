use crate::shared::string_struct;
use geohash::{decode, encode, neighbors, Coord};
use polars::prelude::*;

pub fn geohash_encoder(
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
        _ => Ok(None),
    }
}

pub fn geohash_decoder(ca: &StringChunked) -> PolarsResult<StructChunked> {
    let mut longitude: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("longitude".into(), ca.len());
    let mut latitude: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("latitude".into(), ca.len());

    for value in ca.iter() {
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
    StructChunked::from_series(ca.name().clone(), ca.len(), [ser_long, ser_lat].iter())
}

pub fn geohash_neighbors(ca: &StringChunked) -> PolarsResult<StructChunked> {
    string_struct(ca, ["n", "ne", "e", "se", "s", "sw", "w", "nw"], |value| {
        let n = neighbors(value).map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        Ok([
            Some(n.n),
            Some(n.ne),
            Some(n.e),
            Some(n.se),
            Some(n.s),
            Some(n.sw),
            Some(n.w),
            Some(n.nw),
        ])
    })
}
