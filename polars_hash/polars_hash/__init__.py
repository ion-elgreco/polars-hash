from __future__ import annotations

import warnings
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, cast

import polars as pl
from polars.plugins import register_plugin_function

try:
    from polars._typing import IntoExpr, PolarsDataType
except ImportError:
    from polars.type_aliases import IntoExpr, PolarsDataType  # type: ignore[no-redef]

from polars_hash._internal import __version__ as __version__

_PLUGIN_PATH = Path(__file__).parent
_U64_MAX = 2**64 - 1


def _plugin(
    function_name: str,
    args: IntoExpr | list[IntoExpr],
    **kwargs: Any,
) -> pl.Expr:
    """Call one of the plugin's expressions.

    Every expression in this module is elementwise and lives in the same
    directory, so the call site only ever varies by name, arguments and kwargs.
    """
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name=function_name,
        args=args,
        is_elementwise=True,
        kwargs=kwargs or None,
    )


def _encode_u64_seed(seed: int) -> int:
    """Map a `u64` seed onto the `i64` range that plugin kwargs travel in.

    Kwargs reach the plugin as a pickle, whose integers are `i64`, so a seed
    above `i64::MAX` has to cross as its two's-complement counterpart.
    """
    if not 0 <= seed <= _U64_MAX:
        raise ValueError(f"seed must fit in a u64, got {seed}")
    return seed - 2**64 if seed >= 2**63 else seed


@pl.api.register_expr_namespace("chash")
class CryptographicHashingNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def sha256(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha256 from SHA-2 family."""
        warnings.warn(
            "Call to deprecated method chash.sha256. Use chash.sha2_256() instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return _plugin("sha2_256", self._expr)

    def sha2_256(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha256 from SHA-2 family."""
        return _plugin("sha2_256", self._expr)

    def sha2_512(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha512 from SHA-2 family."""
        return _plugin("sha2_512", self._expr)

    def sha2_384(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha384 from SHA-2 family."""
        return _plugin("sha2_384", self._expr)

    def sha2_224(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha224 from SHA-2 family."""
        return _plugin("sha2_224", self._expr)

    def sha3_256(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha256 from SHA-3 family."""
        return _plugin("sha3_256", self._expr)

    def sha3_512(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha512 from SHA-3 family."""
        return _plugin("sha3_512", self._expr)

    def sha3_384(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha384 from SHA-3 family."""
        return _plugin("sha3_384", self._expr)

    def sha3_224(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha224 from SHA-3 family."""
        return _plugin("sha3_224", self._expr)

    def sha3_shake128(self, *, length: int) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with shake128 from SHA-3 family."""
        return _plugin("sha3_shake128", self._expr, length=length)

    def blake3(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with blake3."""
        return _plugin("blake3", self._expr)

    def hmac_sha256(self, *, key: str) -> pl.Expr:
        """Takes Utf8 as input and returns hex-encoded HMAC-SHA256 string."""
        return _plugin("hmac_sha256", self._expr, key=key)


@pl.api.register_expr_namespace("nchash")
class NonCryptographicHashingNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def wyhash(self) -> pl.Expr:
        """Takes Bytes or Utf8 as input and returns uint64 hash with wyhash."""
        return _plugin("wyhash", self._expr)

    def sha1(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha1."""
        return _plugin("sha1", self._expr)

    def md5(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with md5."""
        return _plugin("md5", self._expr)

    def murmur32(self, *, seed: int = 0) -> pl.Expr:
        """Takes Utf8 as input and returns uint32 hash with murmur32."""
        return _plugin("murmur32", self._expr, seed=seed)

    def murmur128(self, *, seed: int = 0) -> pl.Expr:
        """Takes Utf8 as input and returns uint128 hash with murmur128."""
        return _plugin("murmur128", self._expr, seed=seed)

    def xxhash32(self, *, seed: int = 0) -> pl.Expr:
        """Takes Utf8 as input and returns uint32 hash with xxhash32."""
        return _plugin("xxhash32", self._expr, seed=seed)

    def xxhash64(self, *, seed: int = 0) -> pl.Expr:
        """Takes Utf8 as input and returns uint64 hash with xxhash64."""
        return _plugin("xxhash64", self._expr, seed=_encode_u64_seed(seed))

    def xxh3_64(self, *, seed: int = 0) -> pl.Expr:
        """Takes Utf8 as input and returns uint64 hash with XXH3 64bit."""
        return _plugin("xxh3_64", self._expr, seed=_encode_u64_seed(seed))

    def xxh3_128(self, *, seed: int = 0) -> pl.Expr:
        """Takes Utf8 as input and returns uint128 hash with XXH3 128bit."""
        return _plugin("xxh3_128", self._expr, seed=_encode_u64_seed(seed))

    def farmhash32(self) -> pl.Expr:
        """Takes Utf8 as input and returns uint32 hash with FarmHash fingerprint32."""
        return _plugin("farmhash32", self._expr)

    def farmhash64(self) -> pl.Expr:
        """Takes Utf8 as input and returns uint64 hash with FarmHash fingerprint64."""
        return _plugin("farmhash64", self._expr)

    def cityhash32(self) -> pl.Expr:
        """Takes Utf8 as input and returns uint32 hash with CityHash32."""
        return _plugin("cityhash32", self._expr)

    def cityhash64(self, *, seed: int | None = None) -> pl.Expr:
        """Takes Utf8 as input and returns uint64 hash with CityHash64.

        Without a seed this is `CityHash64`, with one `CityHash64WithSeed` — a
        different value even for `seed=0`.
        """
        if seed is None:
            return _plugin("cityhash64", self._expr)

        return _plugin("cityhash64_with_seed", self._expr, seed=_encode_u64_seed(seed))

    def cityhash128(self) -> pl.Expr:
        """Takes Utf8 as input and returns uint128 hash with CityHash128."""
        return _plugin("cityhash128", self._expr)

    def gxhash32(self, *, seed: int = 0) -> pl.Expr:
        """Takes Utf8 as input and returns uint32 hash with GxHash."""
        return _plugin("gxhash32", self._expr, seed=_encode_u64_seed(seed))

    def gxhash64(self, *, seed: int = 0) -> pl.Expr:
        """Takes Utf8 as input and returns uint64 hash with GxHash."""
        return _plugin("gxhash64", self._expr, seed=_encode_u64_seed(seed))

    def gxhash128(self, *, seed: int = 0) -> pl.Expr:
        """Takes Utf8 as input and returns uint128 hash with GxHash."""
        return _plugin("gxhash128", self._expr, seed=_encode_u64_seed(seed))


def _length_expr(length: int | str | pl.Expr) -> pl.Expr:
    if isinstance(length, str):
        expr = pl.col(length)
    elif isinstance(length, pl.Expr):
        expr = length
    else:
        expr = pl.lit(length)
    return expr.cast(pl.Int64)


@pl.api.register_expr_namespace("geohash")
class GeoHashingNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def to_coords(self) -> pl.Expr:
        """Takes Utf8 as input and returns a struct of the coordinates."""
        return _plugin("ghash_decode", self._expr)

    def from_coords(self, len: int | str | pl.Expr = 12) -> pl.Expr:
        """Takes Struct with latitude, longitude as input and returns utf8 hash using geohash."""
        return _plugin("ghash_encode", [self._expr, _length_expr(len)])

    def neighbors(self) -> pl.Expr:
        """Takes Utf8 hash as input and returns a struct of the neighbors."""
        return _plugin("ghash_neighbors", self._expr)


@pl.api.register_expr_namespace("h3")
class H3NameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def from_coords(self, len: int = 12) -> pl.Expr:
        """Takes Struct with latitude, longitude as input and returns utf8 H3 spatial index."""
        return _plugin("h3_encode", [self._expr, _length_expr(len)])


@pl.api.register_expr_namespace("timehash")
class TimeHashingNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def from_datetime(
        self, precision: int | str | pl.Expr = 10, *, strict: bool = True
    ) -> pl.Expr:
        """Takes Datetime, Date or epoch seconds as input and returns utf8 hash using timehash.

        Timestamps must fall between 1970-01-01 and 2098-01-01. Higher precision
        means a shorter window: 10 covers about 4 seconds, 8 about 4 minutes.

        Precision may be 1 to 32 and defaults to 10, but past about 18 the hash stops
        changing for present-day timestamps and the extra characters are padding.

        Epoch seconds may be Float64 or any integer type; Float32 cannot hold one
        closely enough to land in the right window.

        With ``strict=False`` a timestamp outside the range yields null instead of
        raising. A ``when``/``then`` guard cannot do this, because polars evaluates
        both branches over the whole column. Precision stays strict either way.
        """
        return _plugin(
            "thash_encode", [self._expr, _length_expr(precision)], strict=strict
        )

    def to_datetime(self) -> pl.Expr:
        """Takes Utf8 hash as input and returns the midpoint of its window as Datetime.

        The hash holds an instant, not a wall clock, so the zone is not recoverable.
        The result is UTC; use ``.dt.convert_time_zone(tz)`` for another zone.
        """
        return _plugin("thash_decode", self._expr)

    def neighbors(self) -> pl.Expr:
        """Takes Utf8 hash as input and returns a struct of the preceding and succeeding hash."""
        return _plugin("thash_neighbors", self._expr)


class UUIDNamespace(str, Enum):
    """Standard namespace for UUID(v5) generation"""

    DNS = "dns"
    URL = "url"
    OID = "oid"
    X500 = "x500"


@pl.api.register_expr_namespace("uuidhash")
class UUIDHashNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def uuid5(self, namespace: UUIDNamespace | str = UUIDNamespace.DNS) -> pl.Expr:
        """Generate UUID5 from string input using specified namespace.

        Args: namespace:
        UUIDNamespace.{DNS | URL | OID | X500} or a custom UUID string.

        Returns:
            Expression producing UUID5 strings.
        """
        return _plugin("uuid5", [self._expr, pl.lit(namespace)])

    def uuid5_concat(self, other: pl.Expr, default: str | None = None) -> pl.Expr:
        """Concatenate two columns and generate UUID5 using DNS namespace.

        Args:
            other: Second column to concatenate.
            default: Value to use when other is null. If None, null is treated as empty string.

        Returns:
            Expression producing UUID5 strings.
        """
        if default is not None:
            return _plugin("uuid5_concat_default", [self._expr, other, pl.lit(default)])

        return _plugin("uuid5_concat", [self._expr, other])


class HExpr(pl.Expr):
    @property
    def chash(self) -> CryptographicHashingNameSpace:
        return CryptographicHashingNameSpace(self)

    @property
    def nchash(self) -> NonCryptographicHashingNameSpace:
        return NonCryptographicHashingNameSpace(self)

    @property
    def geohash(self) -> GeoHashingNameSpace:
        return GeoHashingNameSpace(self)

    @property
    def h3(self) -> H3NameSpace:
        return H3NameSpace(self)

    @property
    def timehash(self) -> TimeHashingNameSpace:
        return TimeHashingNameSpace(self)

    @property
    def uuidhash(self) -> UUIDHashNameSpace:
        return UUIDHashNameSpace(self)


class HashColumn(Protocol):
    def __call__(
        self,
        name: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType],
        *more_names: str | PolarsDataType,
    ) -> HExpr: ...

    def __getattr__(self, name: str) -> pl.Expr: ...

    @property
    def chash(self) -> CryptographicHashingNameSpace: ...

    @property
    def nchash(self) -> NonCryptographicHashingNameSpace: ...

    @property
    def geohash(self) -> GeoHashingNameSpace: ...

    @property
    def timehash(self) -> TimeHashingNameSpace: ...

    @property
    def uuidhash(self) -> UUIDHashNameSpace: ...


class HashConcatStr(Protocol):
    def __call__(
        self,
        exprs: IntoExpr | Iterable[IntoExpr],
        *more_exprs: IntoExpr,
        separator: str = "",
        ignore_nulls: bool = False,
    ) -> HExpr: ...

    def __getattr__(self, name: str) -> pl.Expr: ...

    @property
    def chash(self) -> CryptographicHashingNameSpace: ...

    @property
    def nchash(self) -> NonCryptographicHashingNameSpace: ...

    @property
    def uuidhash(self) -> UUIDHashNameSpace: ...


col = cast(HashColumn, pl.col)
concat_str = cast(HashConcatStr, pl.concat_str)


__all__ = ["UUIDNamespace", "__version__", "col", "concat_str"]
