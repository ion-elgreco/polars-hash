import polars as pl
from polars.utils.udfs import _get_shared_lib_location

lib = _get_shared_lib_location(__file__)

__version__ = "0.1.0"

@pl.api.register_expr_namespace("stbl_hash")
class HashingAlgorithms:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def sha256(self) -> pl.Expr:  # type: ignore
        return self._expr._register_plugin(
            lib=lib,
            symbol="sha256",
            is_elementwise=True,
        )
