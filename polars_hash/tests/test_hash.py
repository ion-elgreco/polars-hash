import polars as pl
import pytest
from polars.exceptions import ComputeError
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


def test_sha3_shake128():
    result = pl.select(pl.lit("hello_world").chash.sha3_shake128(length=10))  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                ["6b57b385e070e3534257"],
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


def test_murmurhash32():
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})
    result = df.select(pl.col("literal").nchash.murmur32())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [
                    3531928679,
                    None,
                    0,
                ],
                dtype=pl.UInt32,
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_murmurhash32_seeded():
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})
    result = df.select(plh.col("literal").nchash.murmur32(seed=42))

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [
                    259561949,
                    None,
                    142593372,
                ],
                dtype=pl.UInt32,
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_murmurhash128():
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})
    result = df.select(plh.col("literal").nchash.murmur128())

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [
                    b"\x98,\xf3\x9e\x1c\x1a\xa5]\x1b\x07\x97\x16\x07l\x8de",
                    None,
                    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
                ],
                dtype=pl.Binary,
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_xxhash32():
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})
    result = df.select(pl.col("literal").nchash.xxhash32())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [
                    1605956417,
                    None,
                    46947589,
                ],
                dtype=pl.UInt32,
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_xxhash64():
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})
    result = df.select(pl.col("literal").nchash.xxhash64())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [
                    5654987600477331689,
                    None,
                    17241709254077376921,
                ],
                dtype=pl.UInt64,
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_big():
    df = (
        pl.DataFrame({"a": ["asdfasdf" * 1_000_000]})
        .with_columns(pl.col("a").str.split(""))
        .explode("a")
    )
    print(df.select(plh.col("a").nchash.xxhash64()))


def test_xxhash32_seeded():
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})
    result = df.select(pl.col("literal").nchash.xxhash32(seed=42))  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [
                    1544934469,
                    None,
                    3586027192,
                ],
                dtype=pl.UInt32,
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_xxhash64_seeded():
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})
    result = df.select(pl.col("literal").nchash.xxhash64(seed=42))  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [
                    17477110538672341566,
                    None,
                    11002672306508523268,
                ],
                dtype=pl.UInt64,
            ),
        ]
    )

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("hash_fn_expr"),
    [
        plh.col("literal").nchash.xxhash32(seed=None),  # type: ignore
    ],
)
def test_forced_missing_seed_errors(hash_fn_expr):
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})

    with pytest.raises(ComputeError, match="expected u32"):
        df.select(hash_fn_expr)


def test_xxh3_64():
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})
    result = df.select(plh.col("literal").nchash.xxh3_64())

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [
                    7060460777671424209,
                    None,
                    3244421341483603138,
                ],
                dtype=pl.UInt64,
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_xxh3_64_seeded():
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})
    result = df.select(plh.col("literal").nchash.xxh3_64(seed=42))

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [
                    827481053383045869,
                    None,
                    12693748630217917650,
                ],
                dtype=pl.UInt64,
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_xxh3_128():
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})
    result = df.select(plh.col("literal").nchash.xxh3_128())

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [
                    b'\x03o\xfe!^\x18\xfbg"\xc6=\xaf^\x1c\xd3\xbe',
                    None,
                    b"\x7fI\x8dF$\xc3\x01`\xd8\x98G\x01\xd3\x06\xaa\x99",
                ],
                dtype=pl.Binary,
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_xxh3_128_seeded():
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})
    result = df.select(plh.col("literal").nchash.xxh3_128(seed=42))

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [
                    b"BM\xd8\x9d\x8dX]|k\xd9\xb9\xc0|\xea\xc7\xec",
                    None,
                    b"d\x91$\xfe\xe9\t\x1d</\xaf\xf73\xcd\n\xc2\x16",
                ],
                dtype=pl.Binary,
            ),
        ]
    )

    assert_frame_equal(result, expected)
