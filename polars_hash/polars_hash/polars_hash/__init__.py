import polars as pl
from polars.utils.udfs import _get_shared_lib_location
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr
from typing import Protocol, Iterable, cast
from polars.type_aliases import PolarsDataType, IntoExpr

lib = _get_shared_lib_location(__file__)

__version__ = "0.2.4"


@pl.api.register_expr_namespace("chash")
class CryptographicHashingNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def sha256(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha256."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="sha256",
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
    ) -> HExpr:
        ...

    def __getattr__(self, name: str) -> pl.Expr:
        ...

    @property
    def chash(self) -> CryptographicHashingNameSpace:
        ...

    @property
    def nchash(self) -> NonCryptographicHashingNameSpace:
        ...

    @property
    def geohash(self) -> GeoHashingNameSpace:
        ...


class HashConcatStr(Protocol):
    def __call__(
        self,
        exprs: IntoExpr | Iterable[IntoExpr],
        *more_exprs: IntoExpr,
        seperator: str = "",
    ) -> HExpr:
        ...

    def __getattr__(self, name: str) -> pl.Expr:
        ...

    @property
    def chash(self) -> CryptographicHashingNameSpace:
        ...

    @property
    def nchash(self) -> NonCryptographicHashingNameSpace:
        ...


col = cast(HashColumn, pl.col)
concat_str = cast(HashConcatStr, pl.concat_str)


__all__ = ["col", "concat_str"]
