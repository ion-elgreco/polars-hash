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
