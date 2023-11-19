import polars as pl
from polars.utils.udfs import _get_shared_lib_location
from typing import Protocol, Iterable, cast
from polars.type_aliases import PolarsDataType, IntoExpr

lib = _get_shared_lib_location(__file__)

__version__ = "0.1.2"


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
        """Takes Utf8 as input and returns uint hash with wyhash."""
        return self._expr.register_plugin(
            lib=lib,
            symbol="wyhash",
            is_elementwise=True,
        )


class HExpr(pl.Expr):
    @property
    def chash(self) -> CryptographicHashingNameSpace:
        return CryptographicHashingNameSpace(self)

    @property
    def nchash(self) -> NonCryptographicHashingNameSpace:
        return NonCryptographicHashingNameSpace(self)


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
