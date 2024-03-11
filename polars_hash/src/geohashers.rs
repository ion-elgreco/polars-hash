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

pub fn geohash_decoder(ca: &StringChunked) -> PolarsResult<StructChunked> {
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
    StructChunked::new(ca.name(), &[ser_long, ser_lat])
}

pub fn geohash_neighbors(ca: &StringChunked) -> PolarsResult<StructChunked> {
    let mut n_ca = StringChunkedBuilder::new("n", ca.len());
    let mut ne_ca = StringChunkedBuilder::new("ne", ca.len());
    let mut e_ca = StringChunkedBuilder::new("e", ca.len());
    let mut se_ca = StringChunkedBuilder::new("se", ca.len());
    let mut s_ca = StringChunkedBuilder::new("s", ca.len());
    let mut sw_ca = StringChunkedBuilder::new("sw", ca.len());
    let mut w_ca = StringChunkedBuilder::new("w", ca.len());
    let mut nw_ca = StringChunkedBuilder::new("nw", ca.len());

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

    StructChunked::new(
        ca.name(),
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
    )
}
