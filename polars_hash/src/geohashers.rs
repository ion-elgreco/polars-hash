use geohash::{decode, encode, neighbors, Coord};
use polars::prelude::*;

// Helper function to encode with base 32
fn encode_base32(coord: Coord<f64>, len: usize) -> Result<String, PolarsError> {
    encode(coord, len).map_err(|e| PolarsError::ComputeError(e.to_string().into()))
}

// Helper function to encode with base 16
fn encode_base16(coord: Coord<f64>, len: usize) -> Result<String, PolarsError> {
    // Placeholder for actual base 16 encoding logic
    // Replace this with the actual base 16 encode function when available
    encode(coord, len).map_err(|e| PolarsError::ComputeError(e.to_string().into()))
}

// Function to select the appropriate encoding function based on the base
fn select_encode_function(base: i64) -> fn(Coord<f64>, usize) -> Result<String, PolarsError> {
    match base {
        32 => encode_base32,
        _ => encode_base16, // Default to base 16
    }
}

// Geohash encoder function
pub fn geohash_encoder(
    lat: Option<f64>,
    long: Option<f64>,
    len: Option<i64>,
) -> PolarsResult<Option<String>> {
    let base = 16; // Set default base to 16
    match (lat, long) {
        (Some(lat), Some(long)) => match len {
            Some(len) => {
                let coord = Coord { x: long, y: lat };
                let encode_fn = select_encode_function(base);
                let encoded = encode_fn(coord, len as usize)?;
                Ok(Some(encoded))
            }
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

// Geohash decoder function
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

// Geohash neighbors function
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
