from __future__ import annotations

import warnings
from typing import Iterable, Protocol, cast

import polars as pl
from polars.type_aliases import IntoExpr, PolarsDataType
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr
from polars.utils.udfs import _get_shared_lib_location

from ._internal import __version__ as __version__

lib = _get_shared_lib_location(__file__)


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
        return self._expr.register_plugin(
            lib=lib,
            symbol="sha2_256",
            is_elementwise=True,
        )

    def sha2_256(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha256 from SHA-2 family."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="sha2_256",
            is_elementwise=True,
        )

    def sha2_512(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha512 from SHA-2 family."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="sha2_512",
            is_elementwise=True,
        )

    def sha2_384(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha384 from SHA-2 family."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="sha2_384",
            is_elementwise=True,
        )

    def sha2_224(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha224 from SHA-2 family."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="sha2_224",
            is_elementwise=True,
        )

    def sha3_256(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha256 from SHA-3 family."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="sha3_256",
            is_elementwise=True,
        )

    def sha3_512(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha512 from SHA-3 family."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="sha3_512",
            is_elementwise=True,
        )

    def sha3_384(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha384 from SHA-3 family."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="sha3_384",
            is_elementwise=True,
        )

    def sha3_224(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha224 from SHA-3 family."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="sha3_224",
            is_elementwise=True,
        )

    def blake3(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with blake3."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="blake3",
            is_elementwise=True,
        )


@pl.api.register_expr_namespace("nchash")
class NonCryptographicHashingNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def wyhash(self) -> pl.Expr:
        """Takes Utf8 as input and returns uint64 hash with wyhash."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="wyhash",
            is_elementwise=True,
        )

    def sha1(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha1."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="sha1",
            is_elementwise=True,
        )

    def md5(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with md5."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="md5",
            is_elementwise=True,
        )


@pl.api.register_expr_namespace("geohash")
class GeoHashingNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def to_coords(self) -> pl.Expr:
        """Takes Utf8 as input and returns a struct of the coordinates."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="ghash_decode",
            is_elementwise=True,
        )

    def from_coords(self, len: int | str | pl.Expr = 12) -> pl.Expr:
        """Takes Struct with latitude, longitude as input and returns utf8 hash using geohash."""
        len_expr = wrap_expr(parse_as_expression(len))
        return self._expr.register_plugin(
            lib=lib,
            args=[len_expr],
            symbol="ghash_encode",
            is_elementwise=True,
        )

    def neighbors(self) -> pl.Expr:
        """Takes Utf8 hash as input and returns a struct of the neighbors."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="ghash_neighbors",
            is_elementwise=True,
        )


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


class HashConcatStr(Protocol):
    def __call__(
        self,
        exprs: IntoExpr | Iterable[IntoExpr],
        *more_exprs: IntoExpr,
        seperator: str = "",
    ) -> HExpr: ...

    def __getattr__(self, name: str) -> pl.Expr: ...

    @property
    def chash(self) -> CryptographicHashingNameSpace: ...

    @property
    def nchash(self) -> NonCryptographicHashingNameSpace: ...


col = cast(HashColumn, pl.col)
concat_str = cast(HashConcatStr, pl.concat_str)


__all__ = ["col", "concat_str"]
