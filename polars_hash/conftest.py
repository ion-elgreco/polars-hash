"""Shared fixtures for the test suite.

The docstrings in `polars_hash` are the API reference that mkdocstrings renders,
and `--doctest-modules` runs their examples. Those examples are written the way a
user writes them, with `pl` and `plh` imported, so the doctest namespace has to
supply both names.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

import polars_hash as plh


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace: dict[str, Any]) -> None:
    doctest_namespace["pl"] = pl
    doctest_namespace["plh"] = plh
