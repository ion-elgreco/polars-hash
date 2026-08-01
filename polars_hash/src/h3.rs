use h3o::{LatLng, Resolution};
use polars::prelude::*;

fn get_resolution(resolution: i64) -> PolarsResult<Resolution> {
    match resolution {
        1 => Ok(Resolution::One),
        2 => Ok(Resolution::Two),
        3 => Ok(Resolution::Three),
        4 => Ok(Resolution::Four),
        5 => Ok(Resolution::Five),
        6 => Ok(Resolution::Six),
        7 => Ok(Resolution::Seven),
        8 => Ok(Resolution::Eight),
        9 => Ok(Resolution::Nine),
        10 => Ok(Resolution::Ten),
        11 => Ok(Resolution::Eleven),
        12 => Ok(Resolution::Twelve),
        13 => Ok(Resolution::Thirteen),
        14 => Ok(Resolution::Fourteen),
        15 => Ok(Resolution::Fifteen),
        _ => {
            polars_bail!(InvalidOperation: "expected resolution between 1 and 15, got {}", resolution)
        }
    }
}

pub fn h3_encoder(
    lat: Option<f64>,
    long: Option<f64>,
    len: Option<i64>,
) -> PolarsResult<Option<String>> {
    match (lat, long) {
        (Some(lat), Some(long)) => match len {
            Some(len) => {
                if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&long) {
                    polars_bail!(
                        ComputeError:
                        "invalid coordinate range: latitude {}, longitude {}", lat, long
                    )
                }
                let coord = LatLng::new(lat, long).map_err(|e| {
                    PolarsError::ComputeError(
                        format!("invalid coordinate: latitude {lat}, longitude {long}: {e}").into(),
                    )
                })?;
                Ok(Some(coord.to_cell(get_resolution(len)?).to_string()))
            }
            _ => Err(PolarsError::ComputeError(
                "Length may not be null".to_string().into(),
            )),
        },
        _ => Ok(None),
    }
}
