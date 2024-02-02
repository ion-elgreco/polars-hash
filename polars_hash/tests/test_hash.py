import polars as pl
import polars_hash as plh  # noqa: F401

from polars.testing import assert_frame_equal


def test_sha256():
    result = pl.select(pl.lit("hello_world").chash.sha2_256())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                ["35072c1ae546350e0bfa7ab11d49dc6f129e72ccd57ec7eb671225bbd197c8f1"],
                dtype=pl.Utf8,
            ),
        ]
    )
    assert_frame_equal(result, expected)


def test_wyhash():
    result = pl.select(pl.lit("hello_world").nchash.wyhash())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series("literal", [16737367591072095403], dtype=pl.UInt64),
        ]
    )

    assert_frame_equal(result, expected)


def test_geohash():
    df = pl.DataFrame(
        {"coord": [{"longitude": -120.6623, "latitude": 35.3003}]},
        schema={
            "coord": pl.Struct(
                [pl.Field("longitude", pl.Float64), pl.Field("latitude", pl.Float64)]
            ),
        },
    )

    result = df.select(pl.col("coord").geohash.from_coords(5))  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series("coord", ["9q60y"], dtype=pl.Utf8),
        ]
    )
    assert_frame_equal(result, expected)
    assert_frame_equal(
        df.select(pl.col("coord").geohash.from_coords(12).geohash.to_coords()),
        df,  # type: ignore
    )


def test_lazy_name():
    result = (
        pl.from_dicts({"h1": "sp1xk2m6194y"})
        .lazy()
        .with_columns(pl.col("h1").geohash.neighbors())
        .unnest("h1")
        .collect()
    )

    expected = pl.DataFrame(
        [
            pl.Series("n", ["sp1xk2m6194z"], dtype=pl.Utf8),
            pl.Series("ne", ["sp1xk2m6195p"], dtype=pl.Utf8),
            pl.Series("e", ["sp1xk2m6195n"], dtype=pl.Utf8),
            pl.Series("se", ["sp1xk2m6195j"], dtype=pl.Utf8),
            pl.Series("s", ["sp1xk2m6194v"], dtype=pl.Utf8),
            pl.Series("sw", ["sp1xk2m6194t"], dtype=pl.Utf8),
            pl.Series("w", ["sp1xk2m6194w"], dtype=pl.Utf8),
            pl.Series("nw", ["sp1xk2m6194x"], dtype=pl.Utf8),
        ]
    )

    assert_frame_equal(result, expected)


def test_geohash_13():
    result = (
        pl.from_dict(
            {"longitude": [90.6623, -120.6623], "latitude": [40.3003, 35.3003]}
        )
        .with_columns(geohash=pl.struct(["latitude", "longitude"]))
        .with_columns(plh.col("geohash").geohash.from_coords())  # type: ignore
    )

    expected = pl.DataFrame(
        [
            pl.Series("longitude", [90.6623, -120.6623], dtype=pl.Float64),
            pl.Series("latitude", [40.3003, 35.3003], dtype=pl.Float64),
            pl.Series("geohash", ["wp0mr06q28qt", "9q60y60rhsgg"], dtype=pl.String),
        ]
    )
    assert_frame_equal(result, expected)
