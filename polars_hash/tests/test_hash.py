from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.plugins import register_plugin_function
from polars.testing import assert_frame_equal, assert_series_equal

import polars_hash as plh


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


def test_hmac_sha256():
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})
    result = df.select(pl.col("literal").chash.hmac_sha256(key="secret"))  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [
                    "e0f5b5bb7264e77b340a55a694a6c9ca4edc035c394c703a0408f099563be1ca",
                    None,
                    "f9e66e179b6747ae54108f82f8ade8b3c25d76fd30afde6c395822c530196169",
                ],
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


def test_md5_bytes_null():
    df = pl.DataFrame({"b": pl.Series([b"my_bytes", None], dtype=pl.Binary)})
    result = df.select(pl.col("b").nchash.md5())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series("b", ["4445d78d11baa258c5f4ac1b8d33b8ba", None], dtype=pl.Utf8),
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


def test_blake3_bytes_null():
    df = pl.DataFrame({"b": pl.Series([b"my_bytes", None], dtype=pl.Binary)})
    result = df.select(pl.col("b").chash.blake3())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series(
                "b",
                [
                    "4656d42e3468733c9316ef5d4e4488682fc41ad441644ca63cde6aced8378605",
                    None,
                ],
                dtype=pl.Utf8,
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_farmhash32():
    result = pl.select(pl.lit("hello world").nchash.farmhash32())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series("literal", [430397466], dtype=pl.UInt32),
        ]
    )

    assert_frame_equal(result, expected)


def test_farmhash64():
    result = pl.select(pl.lit("hello world").nchash.farmhash64())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series("literal", [6381520714923946011], dtype=pl.UInt64),
        ]
    )

    assert_frame_equal(result, expected)


# Reference values from the `cityhash` package on PyPI, which wraps the C++
# implementation. The lengths straddle every dispatch branch, and the range where
# CityHash and FarmHash still return the same value (<= 12 bytes for the 32-bit
# pair, <= 32 for the 64-bit pair).
CITYHASH_VECTORS = [
    # value, cityhash32, cityhash64, cityhash64(seed=42), cityhash128
    (
        "hello_world",
        1719156559,
        15605398435621216523,
        10175920941468920074,
        133423608296839006301901834072762183026,
    ),
    (
        "abcdefghijklm",
        2011858552,
        8550978237989882775,
        5922989375063028602,
        24191345147165086804460642396259579660,
    ),
    (
        "the quick brown fox jumps over th",
        1180637772,
        16856596542782860373,
        2679240623542055829,
        336604998673075951095544934565492753941,
    ),
    (
        "0123456789" * 10,
        2906309322,
        15263031703162308175,
        4920090847931360695,
        66910926801977979717489458583538888857,
    ),
    (
        "polars-hash cityhash coverage row " * 5 + "0123456789" * 3,
        1390292832,
        6019340673294074119,
        1791122867898158287,
        246668851854637870664640075271962112941,
    ),
    # Hashed as its 22 UTF-8 bytes, not as 12 code points.
    (
        "\N{LATIN SMALL LETTER E WITH ACUTE}l\N{LATIN SMALL LETTER E WITH GRAVE}"
        "ve-\N{CJK UNIFIED IDEOGRAPH-65E5}\N{CJK UNIFIED IDEOGRAPH-672C}"
        "\N{CJK UNIFIED IDEOGRAPH-8A9E}-\N{PARTY POPPER}",
        1470863098,
        17522264240893443485,
        9690003126991708700,
        20224271564307957976476495106258650914,
    ),
]


@pytest.mark.parametrize(
    ("value", "c32", "c64", "c64_seeded", "c128"),
    CITYHASH_VECTORS,
    ids=lambda v: None if not isinstance(v, str) else f"len{len(v.encode())}",
)
def test_cityhash_matches_the_reference(value, c32, c64, c64_seeded, c128):
    df = pl.DataFrame({"literal": [value]})
    result = df.select(
        c32=plh.col("literal").nchash.cityhash32(),
        c64=plh.col("literal").nchash.cityhash64(),
        c64_seeded=plh.col("literal").nchash.cityhash64(seed=42),
        c128=plh.col("literal").nchash.cityhash128(),
    )

    assert result.row(0) == (c32, c64, c64_seeded, c128)


@pytest.mark.parametrize(
    ("expr", "dtype", "empty"),
    [
        (plh.col("literal").nchash.cityhash32(), pl.UInt32, 3696677242),
        (plh.col("literal").nchash.cityhash64(), pl.UInt64, 11160318154034397263),
        (
            plh.col("literal").nchash.cityhash64(seed=42),
            pl.UInt64,
            12207790695972129833,
        ),
        (
            plh.col("literal").nchash.cityhash128(),
            pl.UInt128,
            82332263323914296566372529678324145705,
        ),
    ],
    ids=["cityhash32", "cityhash64", "cityhash64_seeded", "cityhash128"],
)
def test_cityhash_null_and_empty(expr, dtype, empty):
    df = pl.DataFrame({"literal": [None, ""]})
    result = df.select(expr)

    assert_frame_equal(result, pl.DataFrame(pl.Series("literal", [None, empty], dtype)))


@pytest.mark.parametrize(
    ("city", "farm", "shared"),
    [("cityhash32", "farmhash32", 12), ("cityhash64", "farmhash64", 32)],
)
def test_cityhash_leaves_farmhash_above_the_shared_range(city, farm, shared):
    """FarmHash reuses CityHash on short input, so only longer input tells them apart."""
    df = pl.DataFrame({"literal": ["a" * shared, "a" * (shared + 1)]})
    result = df.select(
        city=getattr(plh.col("literal").nchash, city)(),
        farm=getattr(plh.col("literal").nchash, farm)(),
    )

    assert result["city"][0] == result["farm"][0]
    assert result["city"][1] != result["farm"][1]


def test_cityhash64_seed_zero_is_not_unseeded():
    """`CityHash64WithSeed(v, 0)` is its own hash, not `CityHash64(v)`."""
    df = pl.DataFrame({"literal": ["hello_world"]})
    result = df.select(
        seeded=plh.col("literal").nchash.cityhash64(seed=0),
        unseeded=plh.col("literal").nchash.cityhash64(),
    )

    assert result["seeded"].to_list() == [14430004998761670210]
    assert result["unseeded"].to_list() == [15605398435621216523]


def test_cityhash64_seed_zero_on_an_empty_string_is_zero():
    """`CityHash64("") == k2`, so `HashLen16(k2 - k2, 0)` really is 0."""
    df = pl.DataFrame({"literal": [""]})

    assert df.select(plh.col("literal").nchash.cityhash64(seed=0)).item() == 0


@pytest.mark.parametrize(
    "hash_fn", ["cityhash32", "cityhash64", "cityhash128", "farmhash32", "farmhash64"]
)
def test_cityhash_rejects_a_non_string_column(hash_fn):
    df = pl.DataFrame({"literal": [1, 2, 3]})

    with pytest.raises(ComputeError, match="expected `String`"):
        df.select(getattr(plh.col("literal").nchash, hash_fn)())


@pytest.mark.xfail(
    reason="Polars encodes UInt128 as the private Arrow type _plu128, which pyarrow "
    "rejects. Delete this xfail and the matching note in the cityhash128 docs when "
    "it starts passing — that is the blocker on murmur128 and xxh3_128 returning "
    "UInt128 too.",
    strict=True,
)
def test_cityhash128_reaches_arrow():
    df = pl.DataFrame({"literal": ["hello_world"]})

    df.select(plh.col("literal").nchash.cityhash128()).to_arrow()


# Expected values come from the reference implementations: the `cityhash` and
# `xxhash` packages on PyPI.
@pytest.mark.parametrize(
    ("hash_fn", "seed", "expected"),
    [
        ("cityhash64", 2**63 - 1, 813647477810708320),
        ("cityhash64", 2**63, 14000826912671887845),
        ("cityhash64", 2**64 - 1, 12146214194603664578),
        ("xxhash64", 2**63, 2037080936879071958),
        ("xxhash64", 2**64 - 1, 8264024218298755446),
        ("xxh3_64", 2**63, 13487197773793219251),
        ("xxh3_64", 2**64 - 1, 1513394910116877137),
    ],
)
def test_seed_accepts_the_whole_u64_range(hash_fn, seed, expected):
    """Seeds above `i64::MAX` reach the plugin, not just the lower half."""
    df = pl.DataFrame({"literal": ["hello_world"]})
    expr = getattr(plh.col("literal").nchash, hash_fn)(seed=seed)

    assert df.select(expr).item() == expected


@pytest.mark.parametrize("hash_fn", ["cityhash64", "xxhash64", "xxh3_64", "xxh3_128"])
@pytest.mark.parametrize("seed", [-1, 2**64, 2**70])
def test_seed_outside_the_u64_range_errors(hash_fn, seed):
    with pytest.raises(ValueError, match="seed must fit in a u64"):
        getattr(plh.col("literal").nchash, hash_fn)(seed=seed)


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


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [
        (91.0, -120.6623),
        (999.0, -120.6623),
        (-999.0, -120.6623),
        (35.3003, 181.0),
        (float("nan"), -120.6623),
        (float("inf"), -120.6623),
    ],
)
def test_h3_invalid_coords(latitude, longitude):
    df = pl.DataFrame(
        {"coord": [{"longitude": longitude, "latitude": latitude}]},
        schema={
            "coord": pl.Struct(
                [pl.Field("longitude", pl.Float64), pl.Field("latitude", pl.Float64)]
            ),
        },
    )

    with pytest.raises(ComputeError, match="invalid coordinate range"):
        df.select(pl.col("coord").h3.from_coords(5))  # type: ignore


@pytest.mark.parametrize(
    "dtype",
    [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64],
)
def test_length_arg_dtypes(dtype):
    """All three namespaces route the length/precision through `_length_expr`."""
    df = pl.DataFrame(
        {
            "latitude": [35.3003],
            "longitude": [-120.6623],
            "t": [datetime(2017, 2, 21, 20, 15, 13)],
        },
    ).with_columns(coord=pl.struct(["latitude", "longitude"]), n=pl.lit(5, dtype=dtype))

    assert df.select(pl.col("coord").geohash.from_coords("n")).to_series()[0] == "9q60y"  # type: ignore
    assert (
        df.select(pl.col("coord").h3.from_coords("n")).to_series()[0]  # type: ignore
        == "8529adc7fffffff"
    )
    assert df.select(plh.col("t").timehash.from_datetime("n")).to_series()[0] == "afccc"


def test_from_coords_null_coords():
    df = pl.DataFrame(
        {
            "latitude": pl.Series([35.3003, None], dtype=pl.Float64),
            "longitude": pl.Series([-120.6623, None], dtype=pl.Float64),
        }
    ).with_columns(coord=pl.struct(["latitude", "longitude"]))

    result = df.select(
        geohash=pl.col("coord").geohash.from_coords(5),  # type: ignore
        h3=pl.col("coord").h3.from_coords(5),  # type: ignore
    )

    expected = pl.DataFrame(
        [
            pl.Series("geohash", ["9q60y", None], dtype=pl.Utf8),
            pl.Series("h3", ["8529adc7fffffff", None], dtype=pl.Utf8),
        ]
    )
    assert_frame_equal(result, expected)


def test_uuid5_concat_null_default():
    df = pl.DataFrame({"id": ["abc-123"], "side": pl.Series([None], dtype=pl.Utf8)})

    expr = register_plugin_function(
        plugin_path=Path(plh.__file__).parent,
        function_name="uuid5_concat_default",
        args=[pl.col("id"), pl.col("side"), pl.lit(None, dtype=pl.Utf8)],
        is_elementwise=True,
    )

    with pytest.raises(ComputeError, match="Default value may not be null"):
        df.select(expr)


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


def test_timehash():
    df = pl.DataFrame({"t": [datetime(2017, 2, 21, 20, 15, 13)]})

    result = df.select(plh.col("t").timehash.from_datetime())

    expected = pl.DataFrame(
        [
            pl.Series("t", ["afcccc0e1b"], dtype=pl.Utf8),
        ]
    )
    assert_frame_equal(result, expected)


def test_timehash_to_datetime():
    result = pl.select(pl.lit("afcccc0e1b").timehash.to_datetime())  # type: ignore

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [datetime(2017, 2, 21, 20, 15, 11, 292315, tzinfo=timezone.utc)],
                dtype=pl.Datetime("us", "UTC"),
            ),
        ]
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [pl.Int8, pl.Int16, pl.UInt8, pl.UInt16])
def test_timehash_small_integer_input_dtypes(dtype):
    """These panic at the FFI boundary unless polars carries their dtype-* feature."""
    df = pl.DataFrame({"t": pl.Series([100], dtype=dtype)})

    assert df.select(plh.col("t").timehash.from_datetime(4)).to_series()[0] == "0000"


def test_timehash_to_datetime_is_utc():
    """Declared naive, the UTC instant reads as local time and compares wrong."""
    result = pl.select(pl.lit("afcccc0e1b").timehash.to_datetime())  # type: ignore

    assert result.schema["literal"] == pl.Datetime("us", "UTC")


def test_timehash_to_datetime_recovers_the_original_time_zone():
    """The zone cannot come from the hash, so UTC output makes the one step correct."""
    df = pl.DataFrame(
        {"t": [datetime(2017, 2, 21, 20, 15, 13, tzinfo=timezone.utc)]}
    ).with_columns(pl.col("t").dt.convert_time_zone("America/New_York"))

    hashed = df.select(plh.col("t").timehash.from_datetime(10))
    decoded = hashed.select(
        plh.col("t").timehash.to_datetime().dt.convert_time_zone("America/New_York")
    )

    assert decoded.schema["t"] == pl.Datetime("us", "America/New_York")
    # decode returns the window midpoint, so allow half a window (~1.9s at precision 10)
    assert abs((decoded["t"][0] - df["t"][0]).total_seconds()) < 2


def test_timehash_neighbors():
    result = (
        pl.from_dicts({"h1": "afcccc0e1b"})
        .lazy()
        .with_columns(plh.col("h1").timehash.neighbors())
        .unnest("h1")
        .collect()
    )

    expected = pl.DataFrame(
        [
            pl.Series("before", ["afcccc0e1a"], dtype=pl.Utf8),
            pl.Series("after", ["afcccc0e1c"], dtype=pl.Utf8),
        ]
    )
    assert_frame_equal(result, expected)


def test_timehash_range_edges_have_no_neighbor():
    df = pl.DataFrame({"h": ["0000", "ffff"]})

    result = df.select(plh.col("h").timehash.neighbors()).unnest("h")

    expected = pl.DataFrame(
        [
            pl.Series("before", [None, "fffe"], dtype=pl.Utf8),
            pl.Series("after", ["0001", None], dtype=pl.Utf8),
        ]
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (datetime(2017, 2, 21, 20, 15, 13), pl.Datetime("ms")),
        (datetime(2017, 2, 21, 20, 15, 13), pl.Datetime("us")),
        (datetime(2017, 2, 21, 20, 15, 13), pl.Datetime("ns")),
        (1487708113.0, pl.Float64),
        (1487708113, pl.Int64),
        (1487708113, pl.UInt32),
    ],
)
def test_timehash_input_dtypes(value, dtype):
    df = pl.DataFrame({"t": pl.Series([value], dtype=dtype)})

    assert (
        df.select(plh.col("t").timehash.from_datetime(10)).to_series()[0]
        == "afcccc0e1b"
    )


def test_timehash_time_unit_does_not_change_the_hash():
    """Nanoseconds run past 2^53, so converting to f64 first rounds to 256ns steps."""
    df = pl.DataFrame({"t": [datetime(2017, 2, 21, 20, 15, 14, 147738)]})

    micros = df.with_columns(pl.col("t").cast(pl.Datetime("us")))
    nanos = df.with_columns(pl.col("t").cast(pl.Datetime("ns")))

    assert (
        micros.select(plh.col("t").timehash.from_datetime(16)).to_series()[0]
        == nanos.select(plh.col("t").timehash.from_datetime(16)).to_series()[0]
    )


def test_timehash_time_zone_is_normalized():
    df = pl.DataFrame(
        {"t": [datetime(2017, 2, 21, 20, 15, 13, tzinfo=timezone.utc)]}
    ).with_columns(pl.col("t").dt.convert_time_zone("America/New_York"))

    assert (
        df.select(plh.col("t").timehash.from_datetime(10)).to_series()[0]
        == "afcccc0e1b"
    )


def test_timehash_date():
    df = pl.DataFrame({"t": [date(2017, 2, 21)]})

    assert (
        df.select(plh.col("t").timehash.from_datetime(8)).to_series()[0] == "afccbfaf"
    )


def test_timehash_one_window_shares_a_hash():
    """The feature's central promise. The precision-8 window is about 4 minutes."""
    base = datetime(2017, 2, 21, 20, 15, 13)
    df = pl.DataFrame(
        {"t": [base, base + timedelta(seconds=1), base + timedelta(seconds=2)]}
    )

    hashes = df.select(plh.col("t").timehash.from_datetime(8)).to_series().to_list()

    assert len(set(hashes)) == 1


def test_timehash_separate_windows_differ():
    """The other half of the promise: 10 minutes apart cannot share a 4 minute bin."""
    base = datetime(2017, 2, 21, 20, 15, 13)
    df = pl.DataFrame({"t": [base, base + timedelta(minutes=10)]})

    hashes = df.select(plh.col("t").timehash.from_datetime(8)).to_series().to_list()

    assert hashes[0] != hashes[1]


@pytest.mark.parametrize("precision", [4, 8, 10, 14, 16])
def test_timehash_decode_stays_inside_its_window(precision):
    """to_datetime returns the window midpoint, so re-encoding must land in the same
    window. Catches an interval shift that the single fixed instant would not."""
    base = datetime(2017, 2, 21, 20, 15, 13)
    df = pl.DataFrame(
        {
            "t": [
                base + timedelta(seconds=i * 37, microseconds=i * 811)
                for i in range(50)
            ]
        }
    )

    hashed = df.select(plh.col("t").timehash.from_datetime(precision))
    again = hashed.select(
        plh.col("t").timehash.to_datetime().timehash.from_datetime(precision)
    )

    assert again.to_series().to_list() == hashed.to_series().to_list()


def test_timehash_precision_per_row():
    df = pl.DataFrame(
        {"t": [datetime(2017, 2, 21, 20, 15, 13)] * 3, "n": [4, 8, 10]},
    )

    result = df.select(plh.col("t").timehash.from_datetime("n"))

    expected = pl.DataFrame(
        [
            pl.Series("t", ["afcc", "afcccc0e", "afcccc0e1b"], dtype=pl.Utf8),
        ]
    )
    assert_frame_equal(result, expected)


def test_timehash_scalar_timestamp_broadcasts():
    """One timestamp against a precision column must yield one hash per row."""
    df = pl.DataFrame({"n": [4, 8, 10]})

    result = df.select(
        pl.lit(datetime(2017, 2, 21, 20, 15, 13)).timehash.from_datetime("n").alias("h")
    )

    expected = pl.DataFrame(
        [pl.Series("h", ["afcc", "afcccc0e", "afcccc0e1b"], dtype=pl.Utf8)]
    )
    assert_frame_equal(result, expected)


def test_timehash_scalar_timestamp_broadcasts_in_with_columns():
    df = pl.DataFrame({"n": [4, 8, 10]})

    result = df.with_columns(
        h=pl.lit(datetime(2017, 2, 21, 20, 15, 13)).timehash.from_datetime("n")
    )

    assert result["h"].to_list() == ["afcc", "afcccc0e", "afcccc0e1b"]


def test_timehash_scalar_timestamp_broadcasts_across_chunks():
    """Zipping unequal lengths panics in chunk alignment instead of broadcasting."""
    df = pl.concat(
        [pl.DataFrame({"n": [4, 8]}), pl.DataFrame({"n": [10]})], rechunk=False
    )
    assert df["n"].n_chunks() == 2

    result = df.select(
        pl.lit(datetime(2017, 2, 21, 20, 15, 13)).timehash.from_datetime("n").alias("h")
    )

    assert result["h"].to_list() == ["afcc", "afcccc0e", "afcccc0e1b"]


def test_timehash_null_scalar_timestamp_broadcasts():
    df = pl.DataFrame({"n": [4, 8, 10]})

    result = df.select(
        pl.lit(None, dtype=pl.Datetime("us")).timehash.from_datetime("n").alias("h")
    )

    assert result["h"].to_list() == [None, None, None]


def test_timehash_length_mismatch_is_rejected():
    """Neither operand is a scalar, so erroring beats zipping down to the shorter."""
    df = pl.DataFrame({"n": [4, 8, 10]})
    timestamps = pl.Series([datetime(2017, 2, 21, 20, 15, 13)] * 2)

    with pytest.raises(ComputeError, match="expected equal lengths or a scalar"):
        df.select(pl.lit(timestamps).timehash.from_datetime("n"))


def test_timehash_null_dtype_column():
    """A scan that infers Null must behave like the same all-null data typed."""
    df = pl.DataFrame({"t": pl.Series([None, None], dtype=pl.Null)})

    assert df.select(plh.col("t").timehash.from_datetime(10)).to_series().to_list() == [
        None,
        None,
    ]
    assert df.select(plh.col("t").timehash.to_datetime()).to_series().to_list() == [
        None,
        None,
    ]
    assert df.select(plh.col("t").timehash.neighbors()).to_series().to_list() == [
        {"before": None, "after": None},
        {"before": None, "after": None},
    ]


def test_timehash_null():
    df = pl.DataFrame(
        {
            "t": pl.Series(
                [datetime(2017, 2, 21, 20, 15, 13), None], dtype=pl.Datetime("us")
            ),
            "h": pl.Series(["afcccc0e1b", None], dtype=pl.Utf8),
        }
    )

    result = df.select(
        encoded=plh.col("t").timehash.from_datetime(10),
        decoded=plh.col("h").timehash.to_datetime(),
        neighbors=plh.col("h").timehash.neighbors(),
    )

    expected = pl.DataFrame(
        [
            pl.Series("encoded", ["afcccc0e1b", None], dtype=pl.Utf8),
            pl.Series(
                "decoded",
                [datetime(2017, 2, 21, 20, 15, 11, 292315, tzinfo=timezone.utc), None],
                dtype=pl.Datetime("us", "UTC"),
            ),
            pl.Series(
                "neighbors",
                [
                    {"before": "afcccc0e1a", "after": "afcccc0e1c"},
                    {"before": None, "after": None},
                ],
                dtype=pl.Struct({"before": pl.Utf8, "after": pl.Utf8}),
            ),
        ]
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("precision", [-1, 0, 33])
def test_timehash_invalid_precision(precision):
    df = pl.DataFrame({"t": [datetime(2017, 2, 21, 20, 15, 13)]})

    with pytest.raises(ComputeError, match="expected precision between 1 and 32"):
        df.select(plh.col("t").timehash.from_datetime(precision))


@pytest.mark.parametrize("precision", [-1, 0, 33])
@pytest.mark.parametrize(
    "values", [pytest.param([None, None], id="all-null"), pytest.param([], id="empty")]
)
def test_timehash_invalid_precision_without_a_non_null_row(values, precision):
    """Precision is a static argument, so rejecting it must not depend on the data."""
    df = pl.DataFrame({"t": pl.Series(values, dtype=pl.Datetime("us"))})

    with pytest.raises(ComputeError, match="expected precision between 1 and 32"):
        df.select(plh.col("t").timehash.from_datetime(precision))


@pytest.mark.parametrize(
    "seconds",
    [-1.0, 4039372801.0, float("nan"), float("inf"), float("-inf")],
)
def test_timehash_out_of_range(seconds):
    df = pl.DataFrame({"t": pl.Series([seconds], dtype=pl.Float64)})

    with pytest.raises(ComputeError, match="invalid timestamp range"):
        df.select(plh.col("t").timehash.from_datetime(10))


@pytest.mark.parametrize(
    "seconds", [-1.0, 4039372801.0, float("nan"), float("inf"), float("-inf")]
)
def test_timehash_out_of_range_not_strict(seconds):
    """strict=False nulls an out-of-range row, matching polars' cast and str.to_date."""
    df = pl.DataFrame({"t": pl.Series([seconds], dtype=pl.Float64)})

    result = df.select(plh.col("t").timehash.from_datetime(10, strict=False))

    assert result.to_series().to_list() == [None]


def test_timehash_not_strict_keeps_the_valid_rows():
    df = pl.DataFrame({"t": pl.Series([1487708113.0, -1.0], dtype=pl.Float64)})

    result = df.select(plh.col("t").timehash.from_datetime(10, strict=False))

    assert result.to_series().to_list() == ["afcccc0e1b", None]


def test_timehash_not_strict_survives_when_then():
    """when/then evaluates both branches, so a guard alone cannot exclude bad rows."""
    df = pl.DataFrame({"t": [1.0e9, -5.0]})

    result = df.with_columns(
        h=pl.when(pl.col("t") >= 0).then(
            plh.col("t").timehash.from_datetime(10, strict=False)
        )
    )

    assert result["h"].to_list() == ["1fee011d0a", None]


def test_timehash_not_strict_still_rejects_invalid_precision():
    """Precision is a static argument, not per-row data, so strict does not soften it."""
    df = pl.DataFrame({"t": pl.Series([1487708113.0], dtype=pl.Float64)})

    with pytest.raises(ComputeError, match="expected precision between 1 and 32"):
        df.select(plh.col("t").timehash.from_datetime(99, strict=False))


@pytest.mark.parametrize(
    ("seconds", "expected"), [(0.0, "0000"), (4039372800.0, "ffff")]
)
def test_timehash_range_bounds_are_encodable(seconds, expected):
    df = pl.DataFrame({"t": pl.Series([seconds], dtype=pl.Float64)})

    assert df.select(plh.col("t").timehash.from_datetime(4)).to_series()[0] == expected


@pytest.mark.parametrize(
    "series",
    [
        pytest.param(pl.Series(["not a timestamp"]), id="String"),
        pytest.param(
            pl.Series([Decimal(1487708113)], dtype=pl.Decimal(20, 0)), id="Decimal"
        ),
        pytest.param(pl.Series([True]), id="Boolean"),
        pytest.param(pl.Series([timedelta(seconds=1)]), id="Duration"),
        pytest.param(pl.Series([time(12, 0)]), id="Time"),
        pytest.param(pl.Series([b"x"]), id="Binary"),
    ],
)
def test_timehash_invalid_input_dtype(series):
    df = pl.DataFrame({"t": series})

    with pytest.raises(ComputeError, match="timehash input needs to be"):
        df.select(plh.col("t").timehash.from_datetime(10))


def test_timehash_float32_is_rejected():
    """f32 spaces values 128s apart here, so 1487708113 is held as 1487708160."""
    df = pl.DataFrame({"t": pl.Series([1487708113.0], dtype=pl.Float32)})

    with pytest.raises(ComputeError, match="Float32 cannot hold epoch seconds"):
        df.select(plh.col("t").timehash.from_datetime(10))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "timehash may not be empty"),
        ("zzz", "invalid timehash character 'z'"),
        ("AFCC", "invalid timehash character 'A'"),
        ("afcc ", "invalid timehash character ' '"),
    ],
)
def test_timehash_invalid_hash(value, message):
    """Pin the exact message: every error this module emits contains "timehash"."""
    df = pl.DataFrame({"h": [value]})

    with pytest.raises(ComputeError, match=message):
        df.select(plh.col("h").timehash.to_datetime())
    with pytest.raises(ComputeError, match=message):
        df.select(plh.col("h").timehash.neighbors())


def test_timehash_multi_byte_hash_is_rejected():
    """Upstream panics on a multi-byte character instead of erroring."""
    df = pl.DataFrame({"h": ["é0"]})

    with pytest.raises(ComputeError, match="invalid timehash character"):
        df.select(plh.col("h").timehash.neighbors())


@pytest.mark.parametrize("value", ["a\x00b", "\x00"])
def test_timehash_nul_byte_hash_is_rejected(value):
    """A NUL reaches pyo3-polars inside the error text, which panics building it."""
    df = pl.DataFrame({"h": [value]})

    with pytest.raises(ComputeError, match="invalid timehash character"):
        df.select(plh.col("h").timehash.to_datetime())
    with pytest.raises(ComputeError, match="invalid timehash character"):
        df.select(plh.col("h").timehash.neighbors())


def test_uuid5_url():
    result = pl.select(pl.lit("https://example.com").uuidhash.uuid5("url"))
    expected = pl.DataFrame(
        [
            pl.Series(
                "literal", ["4fd35a71-71ef-5a55-a9d9-aa75c889a6d0"], dtype=pl.Utf8
            ),
        ]
    )
    assert_frame_equal(result, expected)


def test_uuid5_dns_null():
    df = pl.DataFrame({"literal": ["hello", None, "world"]})
    result = df.select(pl.col("literal").uuidhash.uuid5())
    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [
                    "9342d47a-1bab-5709-9869-c840b2eac501",
                    None,
                    "b3a4c24e-f57a-5448-b81b-a643f6768036",
                ],
                dtype=pl.Utf8,
            ),
        ]
    )
    assert_frame_equal(result, expected)


def test_uuid5_concat():
    df = pl.DataFrame({"id": ["abc-123"], "side": ["a"]})
    result = df.select(pl.col("id").uuidhash.uuid5_concat(pl.col("side")))
    expected = pl.DataFrame(
        [
            pl.Series("id", ["e89d330c-f123-519c-a7a1-e48e46f30ccf"], dtype=pl.Utf8),
        ]
    )
    assert_frame_equal(result, expected)


def test_uuid5_concat_with_default():
    df = pl.DataFrame({"id": ["abc-123"], "side": pl.Series([None], dtype=pl.Utf8)})
    result = df.select(pl.col("id").uuidhash.uuid5_concat(pl.col("side"), default="a"))
    expected = pl.DataFrame(
        [
            pl.Series("id", ["e89d330c-f123-519c-a7a1-e48e46f30ccf"], dtype=pl.Utf8),
        ]
    )
    assert_frame_equal(result, expected)


def test_uuid5_concat_broadcasts_a_literal():
    df = pl.DataFrame({"id": ["x", "y", "z"]})
    result = df.with_columns(out=plh.col("id").uuidhash.uuid5_concat(pl.lit("S")))

    expected = pl.Series(
        "out",
        [
            "186c8031-a217-596e-8c93-75b07f90af6f",
            "d0f9d2b4-7e90-5f93-acd1-7f16a3774ed4",
            "d3245ea2-4b8f-56e3-a9ea-9ba38b053026",
        ],
        dtype=pl.Utf8,
    )
    assert_series_equal(result["out"], expected)


def test_uuid5_concat_default_broadcasts_a_literal():
    df = pl.DataFrame({"id": ["x", "y", "z"]})
    result = df.with_columns(
        out=plh.col("id").uuidhash.uuid5_concat(
            pl.lit(None, dtype=pl.Utf8), default="D"
        )
    )

    expected = pl.Series(
        "out",
        [
            "3d0f0686-fd0b-54bf-b4db-811c1c2d6f7c",
            "1004bd8f-803b-5908-8c16-c2081cbf113c",
            "61390897-7d4b-5d1f-9639-8a57caf8216e",
        ],
        dtype=pl.Utf8,
    )
    assert_series_equal(result["out"], expected)


def test_uuid5_concat_broadcasts_the_first_column():
    df = pl.DataFrame({"side": ["S", "S", "S"]})
    result = df.with_columns(
        out=pl.lit("x").uuidhash.uuid5_concat(pl.col("side"))  # type: ignore
    )

    assert result["out"].to_list() == ["186c8031-a217-596e-8c93-75b07f90af6f"] * 3


@pytest.mark.parametrize("default", [None, "D"])
def test_uuid5_concat_rejects_a_length_mismatch(default):
    df = pl.DataFrame({"id": ["x", "y", "z"], "side": ["a", "b", "c"]})

    with pytest.raises(ComputeError, match="expected equal lengths or a scalar"):
        df.select(
            plh.col("id").uuidhash.uuid5_concat(pl.col("side").head(2), default=default)
        )
