import polars as pl
from polars.testing import assert_frame_equal

import polars_hash as plh  # noqa: F401


def test_sha1():
    result = pl.select(pl.lit("hello_world").nchash.sha1())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                ["e4ecd6fc11898565af24977e992cea0c9c7b7025"],
                dtype=pl.Utf8,
            ),
        ]
    )
    assert_frame_equal(result, expected)


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


def test_wyhash_str():
    result = pl.select(pl.lit("hello_world").nchash.wyhash())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series("literal", [16737367591072095403], dtype=pl.UInt64),
        ]
    )

    assert_frame_equal(result, expected)


def test_wyhash_bytes():
    result = pl.select(pl.lit(b"my_bytes").nchash.wyhash())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series("literal", [5112362246832359110], dtype=pl.UInt64),
        ]
    )

    assert_frame_equal(result, expected)


def test_md5_str():
    result = pl.select(pl.lit("hello_world").nchash.md5())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series("literal", ["99b1ff8f11781541f7f89f9bd41c4a17"], dtype=pl.Utf8),
        ]
    )

    assert_frame_equal(result, expected)


def test_md5_bytes():
    result = pl.select(pl.lit(b"my_bytes").nchash.md5())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series("literal", ["4445d78d11baa258c5f4ac1b8d33b8ba"], dtype=pl.Utf8),
        ]
    )

    assert_frame_equal(result, expected)


def test_blake3_str():
    result = pl.select(pl.lit("hello_world").chash.blake3())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                ["9833e5324eb2400de814730f4e92810905351bc0451e10b75847210c1d7c37ed"],
                dtype=pl.Utf8,
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_blake3_bytes():
    result = pl.select(pl.lit(b"my_bytes").chash.blake3())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                ["4656d42e3468733c9316ef5d4e4488682fc41ad441644ca63cde6aced8378605"],
                dtype=pl.Utf8,
            ),
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


def test_h3():
    df = pl.DataFrame(
        {"coord": [{"longitude": -120.6623, "latitude": 35.3003}]},
        schema={
            "coord": pl.Struct(
                [pl.Field("longitude", pl.Float64), pl.Field("latitude", pl.Float64)]
            ),
        },
    )

    result = df.select(pl.col("coord").h3.from_coords(5))  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series("coord", ["8529adc7fffffff"], dtype=pl.Utf8),
        ]
    )
    assert_frame_equal(result, expected)


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
