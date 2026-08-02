import hashlib
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


# Reference values from the `gxhash` package on PyPI, which binds the same upstream
# crate through its own layer, so these pin the wiring rather than the algorithm. The
# lengths straddle every branch `compress_all` takes, up to its wide loop above 64 bytes.
GXHASH_VECTORS = [
    # value, gxhash32, gxhash64, gxhash128, gxhash64(seed=42)
    (
        "hello_world",
        2751540945,
        2180020304351407825,
        56218077491375249900279963678916292305,
        15254170022685821676,
    ),
    (
        "0123456789abcdef",
        1570350261,
        2930507594094716085,
        41598245394210107925662921412237173941,
        14005003272224318756,
    ),
    (
        "0123456789abcdefg",
        238226515,
        7750332876019928147,
        285233948970283975186386669336525999187,
        3032895827694623696,
    ),
    (
        "the quick brown fox jumps over th",
        4022630390,
        2551299749957629942,
        278207565320499808743461372539676424182,
        895124801849494436,
    ),
    (
        "0123456789" * 5 + "abcdefghijklm",
        469949192,
        7716185691520162568,
        336065415150536231499290114686378498824,
        3460190292077382754,
    ),
    (
        "0123456789" * 10,
        1184804170,
        2204170311184592202,
        159737905367241229606696689146798189898,
        2167010504938637199,
    ),
    (
        "0123456789" * 12 + "abcdefgh",
        29132972,
        16624339216108390572,
        144076341491644785736749709244136982700,
        14855682399770243763,
    ),
    (
        "polars-hash gxhash coverage row " * 5 + "0123456789" * 3,
        1826346443,
        333135472238515659,
        225759949245592036298123883189167183307,
        15651006582957085576,
    ),
    # Hashed as its 22 UTF-8 bytes, not as 12 code points.
    (
        "\N{LATIN SMALL LETTER E WITH ACUTE}l\N{LATIN SMALL LETTER E WITH GRAVE}"
        "ve-\N{CJK UNIFIED IDEOGRAPH-65E5}\N{CJK UNIFIED IDEOGRAPH-672C}"
        "\N{CJK UNIFIED IDEOGRAPH-8A9E}-\N{PARTY POPPER}",
        4087074997,
        9898532851104338101,
        81447829809662801153156452411172966581,
        17129501535291329433,
    ),
]


_BYTE_HASHERS = [
    ("chash", "sha2_224", {}),
    ("chash", "sha2_256", {}),
    ("chash", "sha2_384", {}),
    ("chash", "sha2_512", {}),
    ("chash", "sha3_224", {}),
    ("chash", "sha3_256", {}),
    ("chash", "sha3_384", {}),
    ("chash", "sha3_512", {}),
    ("chash", "sha3_shake128", {"length": 8}),
    ("chash", "blake3", {}),
    ("chash", "hmac_sha256", {"key": "secret"}),
    ("nchash", "sha1", {}),
    ("nchash", "md5", {}),
    ("nchash", "wyhash", {}),
    ("nchash", "murmur32", {}),
    ("nchash", "murmur128", {}),
    ("nchash", "xxhash32", {}),
    ("nchash", "xxhash64", {}),
    ("nchash", "xxh3_64", {}),
    ("nchash", "xxh3_128", {}),
    ("nchash", "farmhash32", {}),
    ("nchash", "farmhash64", {}),
    ("nchash", "cityhash32", {}),
    ("nchash", "cityhash64", {}),
    ("nchash", "cityhash64", {"seed": 7}),
    ("nchash", "cityhash128", {}),
    ("nchash", "gxhash32", {}),
    ("nchash", "gxhash64", {}),
    ("nchash", "gxhash128", {}),
    ("nchash", "gxhash64", {"seed": 7}),
    ("uuidhash", "uuid5", {}),
]


@pytest.mark.parametrize(
    ("value", "g32", "g64", "g128", "g64_seeded"),
    GXHASH_VECTORS,
    ids=lambda v: None if not isinstance(v, str) else f"len{len(v.encode())}",
)
def test_gxhash_matches_the_reference(value, g32, g64, g128, g64_seeded):
    df = pl.DataFrame({"literal": [value]})
    result = df.select(
        g32=plh.col("literal").nchash.gxhash32(),
        g64=plh.col("literal").nchash.gxhash64(),
        g128=plh.col("literal").nchash.gxhash128(),
        g64_seeded=plh.col("literal").nchash.gxhash64(seed=42),
    )

    assert result.row(0) == (g32, g64, g128, g64_seeded)


@pytest.mark.parametrize(
    ("expr", "dtype", "empty"),
    [
        (plh.col("literal").nchash.gxhash32(), pl.UInt32, 2533353535),
        (plh.col("literal").nchash.gxhash64(), pl.UInt64, 17210906488525023295),
        (
            plh.col("literal").nchash.gxhash128(),
            pl.UInt128,
            302767221070957831171542222971961600063,
        ),
        (
            plh.col("literal").nchash.gxhash64(seed=42),
            pl.UInt64,
            1387850744621952556,
        ),
    ],
    ids=["gxhash32", "gxhash64", "gxhash128", "gxhash64_seeded"],
)
def test_gxhash_null_and_empty(expr, dtype, empty):
    df = pl.DataFrame({"literal": [None, ""]})
    result = df.select(expr)

    assert_frame_equal(result, pl.DataFrame(pl.Series("literal", [None, empty], dtype)))


@pytest.mark.parametrize("value", ["", "hello_world", "0123456789" * 12 + "abcdefgh"])
def test_gxhash_widths_truncate_one_state(value):
    """All three widths read the same finalized state, so each is the next one's tail."""
    df = pl.DataFrame({"literal": [value]})
    g32, g64, g128 = df.select(
        g32=plh.col("literal").nchash.gxhash32(),
        g64=plh.col("literal").nchash.gxhash64(),
        g128=plh.col("literal").nchash.gxhash128(),
    ).row(0)

    assert g32 == g64 % 2**32
    assert g64 == g128 % 2**64


def test_gxhash_seed_changes_the_hash():
    """Unlike CityHash, gxhash has no unseeded form: seed 0 is the default, not a mode."""
    df = pl.DataFrame({"literal": ["hello_world"]})
    result = df.select(
        default=plh.col("literal").nchash.gxhash64(),
        zero=plh.col("literal").nchash.gxhash64(seed=0),
        other=plh.col("literal").nchash.gxhash64(seed=1),
    )

    assert result["default"].item() == result["zero"].item()
    assert result["other"].item() != result["zero"].item()


@pytest.mark.parametrize("hash_fn", ["gxhash32", "gxhash64", "gxhash128"])
def test_gxhash_rejects_a_non_string_column(hash_fn):
    df = pl.DataFrame({"literal": [1, 2, 3]})

    with pytest.raises(ComputeError, match="expected `String`"):
        df.select(getattr(plh.col("literal").nchash, hash_fn)())


@pytest.mark.parametrize(
    ("namespace", "method", "kwargs"),
    _BYTE_HASHERS,
    ids=[f"{m}{sorted(k)}" for _, m, k in _BYTE_HASHERS],
)
def test_binary_input_hashes_like_the_same_utf8_bytes(namespace, method, kwargs):
    """A hash reads bytes, so the dtype holding them may not change the digest."""
    df = pl.DataFrame({"s": ["hello_world", None], "b": [b"hello_world", None]})

    result = df.select(
        s=getattr(getattr(plh.col("s"), namespace), method)(**kwargs),
        b=getattr(getattr(plh.col("b"), namespace), method)(**kwargs),
    )

    assert_series_equal(result["s"], result["b"], check_names=False)


def test_binary_input_takes_bytes_that_are_not_utf8():
    """The reason Binary is worth accepting: these bytes have no string to stand in."""
    df = pl.DataFrame({"b": [b"\xff\xfe\x00"]})

    result = df.select(plh.col("b").chash.sha2_256())

    assert result.item() == hashlib.sha256(b"\xff\xfe\x00").hexdigest()


def test_a_hasher_names_both_dtypes_it_accepts():
    df = pl.DataFrame({"literal": [1, 2, 3]})

    with pytest.raises(ComputeError, match="expected `String` or `Binary` input"):
        df.select(plh.col("literal").chash.sha2_256())


@pytest.mark.parametrize(
    "series",
    [
        pytest.param(pl.Series([[1, 2]], dtype=pl.Array(pl.Int64, 2)), id="Array"),
        pytest.param(
            pl.Series([[[1.5, 2.5]]], dtype=pl.Array(pl.Array(pl.Float64, 2), 1)),
            id="nested_Array",
        ),
    ],
)
def test_a_rejected_dtype_raises_rather_than_aborting(series):
    """Polars panics while reading an arrow type whose dtype feature is off, and a
    panic across the plugin's C boundary aborts the process instead of raising."""
    df = pl.DataFrame({"literal": series})

    with pytest.raises(ComputeError, match="expected `String`"):
        df.select(plh.col("literal").chash.sha2_256())


# Expected values come from the reference implementations: the `cityhash`, `gxhash` and
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
        # gxhash seeds are `i64` upstream: -1, `i64::MIN` and `i64::MAX`.
        ("gxhash64", 2**64 - 1, 4810392188550196786),
        ("gxhash64", 2**63, 5168275476899811548),
        ("gxhash64", 2**63 - 1, 4596807014130615183),
    ],
)
def test_seed_accepts_the_whole_u64_range(hash_fn, seed, expected):
    """Seeds above `i64::MAX` reach the plugin, not just the lower half."""
    df = pl.DataFrame({"literal": ["hello_world"]})
    expr = getattr(plh.col("literal").nchash, hash_fn)(seed=seed)

    assert df.select(expr).item() == expected


@pytest.mark.parametrize(
    "hash_fn",
    [
        "cityhash64",
        "xxhash64",
        "xxh3_64",
        "xxh3_128",
        "gxhash32",
        "gxhash64",
        "gxhash128",
    ],
)
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


# Expected value from `mmh3.hash128` on PyPI. 0.8.0 changed this expression from
# Binary to UInt128, and the two halves are packed so the integer matches that
# reference; see `test_the_128_bit_digests_round_trip`.
def test_murmurhash128():
    df = pl.DataFrame({"literal": ["hello_world", None, ""]})
    result = df.select(plh.col("literal").nchash.murmur128())

    expected = pl.DataFrame(
        [
            pl.Series(
                "literal",
                [134986332493155497415370161450594282648, None, 0],
                dtype=pl.UInt128,
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
                    253649469245435599925940275794906345219,
                    None,
                    204254712233039002205064565430793619839,
                ],
                dtype=pl.UInt128,
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
                    314735830047873782861649874643137875266,
                    None,
                    30250540579776425168508632643632664932,
                ],
                dtype=pl.UInt128,
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_the_128_bit_digests_round_trip():
    """The two algorithms canonicalise their digest to bytes in opposite orders.

    MurmurHash3 writes two little-endian halves and XXH128 writes one big-endian
    value, so the integers here recover each digest through a different call. Both
    match their reference implementation; before 0.8.0 `xxh3_128()` returned the
    canonical bytes reversed.
    """
    df = pl.DataFrame({"literal": ["hello_world"]})
    murmur, xxh3 = df.select(
        murmur=plh.col("literal").nchash.murmur128(),
        xxh3=plh.col("literal").nchash.xxh3_128(),
    ).row(0)

    # mmh3.hash_bytes("hello_world", 0)
    assert murmur.to_bytes(16, "little").hex() == "982cf39e1c1aa55d1b079716076c8d65"
    # xxhash.xxh128_hexdigest("hello_world")
    assert f"{xxh3:032x}" == "bed31c5eaf3dc62267fb185e21fe6f03"


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


# `encode_rows` writes version 1 of the encoding. These bytes are frozen: a stored
# hash outlives the release that wrote it, so a change here needs a new version.
_ENCODINGS = [
    ("null", pl.Series([None], dtype=pl.Int64), "0d0100"),
    ("false", pl.Series([False]), "0d0101"),
    ("true", pl.Series([True]), "0d0102"),
    ("int_1", pl.Series([1], dtype=pl.Int64), "0d01030101"),
    ("int_minus_1", pl.Series([-1], dtype=pl.Int64), "0d01030001"),
    ("int_300", pl.Series([300], dtype=pl.Int64), "0d010301ac02"),
    ("float_1_5", pl.Series([1.5]), "0d01043ff8000000000000"),
    ("string_ab", pl.Series(["ab"]), "0d0105026162"),
    ("binary_ff", pl.Series([b"\xff"]), "0d010601ff"),
    ("date_day_1", pl.Series([date(1970, 1, 2)]), "0d01070101"),
    ("time_1ns", pl.Series([1], dtype=pl.Int64).cast(pl.Time), "0d01080101"),
    (
        "datetime_1000us",
        pl.Series([1000], dtype=pl.Int64).cast(pl.Datetime("us")),
        "0d010901c0843d",
    ),
    ("duration_1us", pl.Series([timedelta(microseconds=1)]), "0d010a01e807"),
    (
        "decimal_1_50",
        pl.Series([Decimal("1.50")], dtype=pl.Decimal(10, 2)),
        "0d010b010f01",
    ),
    ("list_1_2", pl.Series([[1, 2]]), "0d010c02030101030102"),
    ("struct_a_1", pl.Series([{"a": 1}]), "0d010d01030101"),
]


@pytest.mark.parametrize(
    ("series", "expected"),
    [pytest.param(s, e, id=name) for name, s, e in _ENCODINGS],
)
def test_encode_rows_writes_the_documented_bytes(series, expected):
    df = pl.DataFrame({"x": series})

    result = df.select(plh.encode_rows(pl.all()).alias("e"))

    assert result["e"][0].hex() == expected


def _encode(df: pl.DataFrame) -> list[bytes]:
    return df.select(plh.encode_rows(pl.all()).alias("e"))["e"].to_list()


def test_encode_rows_separates_columns_that_concat_str_runs_together():
    """`("ab", "c")` and `("a", "bc")` are one string but two rows."""
    df = pl.DataFrame({"a": ["ab", "a"], "b": ["c", "bc"]})

    assert _encode(df)[0] != _encode(df)[1]


def test_encode_rows_hashes_a_row_holding_a_list_and_a_struct():
    """The case from issue #43 that `concat_str` cannot reach at all."""
    df = pl.DataFrame(
        {
            "foo": ["hello_world"],
            "baz": [42],
            "qux": [[1, 2, 3]],
            "quux": [{"a": 1, "b": 2}],
        }
    )

    result = df.select(plh.encode_rows(pl.all()).chash.sha2_256())

    assert (
        result.item()
        == "7e9187dade49806c0ae6ec85ecfad6f2f99751be1cc650a03f6f8de0b404aec4"
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param(
            pl.Series([1], dtype=pl.Int8),
            pl.Series([1], dtype=pl.Int64),
            id="int_width",
        ),
        pytest.param(
            pl.Series([1], dtype=pl.UInt64),
            pl.Series([1], dtype=pl.Int64),
            id="int_sign",
        ),
        pytest.param(
            pl.Series([1.5], dtype=pl.Float32),
            pl.Series([1.5], dtype=pl.Float64),
            id="float_width",
        ),
        pytest.param(pl.Series([-0.0]), pl.Series([0.0]), id="negative_zero"),
        pytest.param(
            pl.Series([1000], dtype=pl.Int64).cast(pl.Datetime("ms")),
            pl.Series([1_000_000_000], dtype=pl.Int64).cast(pl.Datetime("ns")),
            id="datetime_unit",
        ),
        pytest.param(
            pl.Series([datetime(2020, 1, 1)]).dt.replace_time_zone("UTC"),
            pl.Series([datetime(2020, 1, 1)]),
            id="datetime_time_zone",
        ),
        pytest.param(
            pl.Series([timedelta(seconds=1)]).cast(pl.Duration("ms")),
            pl.Series([timedelta(seconds=1)]).cast(pl.Duration("ns")),
            id="duration_unit",
        ),
        pytest.param(
            pl.Series([Decimal("1.50")], dtype=pl.Decimal(10, 2)),
            pl.Series([Decimal("1.500")], dtype=pl.Decimal(10, 3)),
            id="decimal_scale",
        ),
        pytest.param(
            pl.Series(["a"], dtype=pl.Categorical), pl.Series(["a"]), id="categorical"
        ),
        pytest.param(
            pl.Series(["a"], dtype=pl.Enum(["z", "a"])), pl.Series(["a"]), id="enum"
        ),
        pytest.param(
            pl.Series([[1, 2]], dtype=pl.Array(pl.Int64, 2)),
            pl.Series([[1, 2]]),
            id="array_and_list",
        ),
    ],
)
def test_encode_rows_reads_a_value_by_what_it_means(left, right):
    """How polars holds a value is not part of the value."""
    assert _encode(pl.DataFrame({"x": left})) == _encode(pl.DataFrame({"x": right}))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param(
            pl.Series([None], dtype=pl.Utf8), pl.Series([""]), id="null_and_empty"
        ),
        pytest.param(
            pl.Series([None], dtype=pl.Int64),
            pl.Series([0], dtype=pl.Int64),
            id="null_and_zero",
        ),
        pytest.param(pl.Series([1], dtype=pl.Int64), pl.Series([1.0]), id="int_float"),
        pytest.param(
            pl.Series([date(2020, 1, 1)]),
            pl.Series([datetime(2020, 1, 1)]),
            id="date_datetime",
        ),
        pytest.param(
            pl.Series([None], dtype=pl.Struct({"a": pl.Int64})),
            pl.Series([{"a": None}], dtype=pl.Struct({"a": pl.Int64})),
            id="null_struct_and_struct_of_nulls",
        ),
        pytest.param(
            pl.Series([None], dtype=pl.List(pl.Int64)),
            pl.Series([[]], dtype=pl.List(pl.Int64)),
            id="null_list_and_empty_list",
        ),
    ],
)
def test_encode_rows_keeps_values_apart(left, right):
    assert _encode(pl.DataFrame({"x": left})) != _encode(pl.DataFrame({"x": right}))


def test_encode_rows_ignores_column_names():
    assert _encode(pl.DataFrame({"a": [1]})) == _encode(pl.DataFrame({"zzz": [1]}))


def test_encode_rows_reads_the_columns_in_order():
    assert _encode(pl.DataFrame({"a": [1], "b": [2]})) != _encode(
        pl.DataFrame({"a": [2], "b": [1]})
    )


def test_encode_rows_gives_a_row_with_a_null_a_hash_of_its_own():
    """`concat_str` returns null for the whole row instead. A null is a value."""
    df = pl.DataFrame({"a": ["x", None, "x"], "b": ["y", "y", None]})

    result = df.select(plh.encode_rows(pl.all()).chash.sha2_256().alias("h"))["h"]

    assert result.null_count() == 0
    assert result[1] != result[2]


def test_encode_rows_is_blind_to_chunking_and_slicing():
    """Offsets of a sliced list column do not start at zero."""
    df = pl.DataFrame({"x": [[1, 2], [3], [4, 5, 6]], "y": ["a", "b", "c"]})
    chunked = pl.concat([df[:1], df[1:]], rechunk=False)

    assert _encode(chunked) == _encode(df)
    assert _encode(df[1:]) == _encode(df)[1:]
    assert _encode(df.filter(pl.col("y") != "b")) == [_encode(df)[0], _encode(df)[2]]


def test_encode_rows_takes_named_columns_and_an_expression():
    df = pl.DataFrame({"a": [1], "b": ["x"], "c": [True]})

    assert (
        _encode(df)
        == df.select(plh.encode_rows("a", "b", "c").alias("e"))["e"].to_list()
    )


def test_encode_rows_rejects_an_encoding_it_cannot_write():
    df = pl.DataFrame({"a": [1]})

    with pytest.raises(ComputeError, match="unknown row encoding version 2"):
        df.select(plh.encode_rows(pl.all(), version=2))


@pytest.mark.parametrize(
    ("namespace", "method"),
    [("chash", "sha2_256"), ("nchash", "xxh3_64"), ("uuidhash", "uuid5")],
)
def test_encode_rows_feeds_any_hasher(namespace, method):
    df = pl.DataFrame({"a": [1, 1, 2], "b": [[1], [1], [2]]})

    result = df.select(
        getattr(getattr(plh.encode_rows(pl.all()), namespace), method)().alias("h")
    )["h"]

    assert result[0] == result[1]
    assert result[0] != result[2]


def test_hash_rows_adds_a_fingerprint_column():
    df = pl.DataFrame({"a": [1, 1, 2], "b": [[1], [1], [2]]})

    result = plh.hash_rows(df)

    assert result.columns == ["a", "b", "hash"]
    assert result["hash"][0] == result["hash"][1] != result["hash"][2]


def test_hash_rows_is_the_expression_it_says_it_is():
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    assert_frame_equal(
        plh.hash_rows(df),
        df.with_columns(plh.encode_rows(pl.all()).nchash.xxh3_64().alias("hash")),
    )


def test_hash_rows_keeps_a_lazyframe_lazy():
    frame = plh.hash_rows(pl.LazyFrame({"a": [1]}))

    assert isinstance(frame, pl.LazyFrame)
    assert frame.collect()["hash"].dtype == pl.UInt64


def test_hash_rows_reads_only_the_subset():
    df = pl.DataFrame({"a": [1, 1], "b": ["x", "y"]})

    result = plh.hash_rows(df, "a")

    assert result["hash"][0] == result["hash"][1]


@pytest.mark.parametrize(
    ("algorithm", "kwargs", "dtype"),
    [
        ("sha2_256", {}, pl.Utf8),
        ("hmac_sha256", {"key": "secret"}, pl.Utf8),
        ("cityhash128", {}, pl.UInt128),
        ("uuid5", {}, pl.Utf8),
    ],
)
def test_hash_rows_reaches_every_namespace(algorithm, kwargs, dtype):
    df = pl.DataFrame({"a": [1]})

    result = plh.hash_rows(df, algorithm=algorithm, name="h", **kwargs)

    assert result["h"].dtype == dtype


def test_hash_rows_names_the_column():
    df = pl.DataFrame({"a": [1]})

    assert plh.hash_rows(df, name="row_id").columns == ["a", "row_id"]


@pytest.mark.parametrize("algorithm", ["nope", "_expr"])
def test_hash_rows_rejects_an_algorithm_it_has_no_hasher_for(algorithm):
    df = pl.DataFrame({"a": [1]})

    with pytest.raises(ValueError, match="unknown algorithm"):
        plh.hash_rows(df, algorithm=algorithm)


def test_encode_rows_keeps_struct_shapes_apart():
    """A struct writes its field count, as a list writes its element count.

    Without it a struct field runs into the field beside it, and two rows with
    different schemas reach the same bytes.
    """
    wide = pl.DataFrame({"a": [{"x": 1, "y": 2}]})
    split = pl.DataFrame({"a": [{"x": 1}], "b": [2]})

    assert _encode(wide) != _encode(split)
