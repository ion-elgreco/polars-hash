from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Protocol, cast

import polars as pl
from polars.plugins import register_plugin_function
from polars.type_aliases import IntoExpr, PolarsDataType

from polars_hash._internal import __version__ as __version__


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
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="sha2_256",
            args=self._expr,
            is_elementwise=True,
        )

    def sha2_256(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha256 from SHA-2 family."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="sha2_256",
            args=self._expr,
            is_elementwise=True,
        )

    def sha2_512(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha512 from SHA-2 family."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="sha2_512",
            args=self._expr,
            is_elementwise=True,
        )

    def sha2_384(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha384 from SHA-2 family."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="sha2_384",
            args=self._expr,
            is_elementwise=True,
        )

    def sha2_224(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha224 from SHA-2 family."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="sha2_224",
            args=self._expr,
            is_elementwise=True,
        )

    def sha3_256(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha256 from SHA-3 family."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="sha3_256",
            args=self._expr,
            is_elementwise=True,
        )

    def sha3_512(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha512 from SHA-3 family."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="sha3_512",
            args=self._expr,
            is_elementwise=True,
        )

    def sha3_384(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha384 from SHA-3 family."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="sha3_384",
            args=self._expr,
            is_elementwise=True,
        )

    def sha3_224(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha224 from SHA-3 family."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="sha3_224",
            args=self._expr,
            is_elementwise=True,
        )

    def blake3(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with blake3."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="blake3",
            args=self._expr,
            is_elementwise=True,
        )


@pl.api.register_expr_namespace("nchash")
class NonCryptographicHashingNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def wyhash(self) -> pl.Expr:
        """Takes Bytes or Utf8 as input and returns uint64 hash with wyhash."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="wyhash",
            args=self._expr,
            is_elementwise=True,
        )

    def sha1(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with sha1."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="sha1",
            args=self._expr,
            is_elementwise=True,
        )

    def md5(self) -> pl.Expr:
        """Takes Utf8 as input and returns utf8 hash with md5."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="md5",
            args=self._expr,
            is_elementwise=True,
        )


@pl.api.register_expr_namespace("geohash")
class GeoHashingNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def to_coords(self) -> pl.Expr:
        """Takes Utf8 as input and returns a struct of the coordinates."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="ghash_decode",
            args=self._expr,
            is_elementwise=True,
        )

    def from_coords(self, len: int | str | pl.Expr = 12) -> pl.Expr:
        """Takes Struct with latitude, longitude as input and returns utf8 hash using geohash."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, len],
            function_name="ghash_encode",
            is_elementwise=True,
        )

    def neighbors(self) -> pl.Expr:
        """Takes Utf8 hash as input and returns a struct of the neighbors."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="ghash_neighbors",
            args=self._expr,
            is_elementwise=True,
        )


@pl.api.register_expr_namespace("h3")
class H3NameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def from_coords(self, len: int = 12) -> pl.Expr:
        """Takes Struct with latitude, longitude as input and returns utf8 H3 spatial index."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, len],
            function_name="h3_encode",
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

    @property
    def h3(self) -> H3NameSpace:
        return H3NameSpace(self)


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


__all__ = ["col", "concat_str", "__version__"]
