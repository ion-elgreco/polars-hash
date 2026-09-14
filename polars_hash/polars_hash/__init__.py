"""Stable non-cryptographic and cryptographic hash functions for Polars.

Importing this package registers seven expression namespaces on `pl.Expr`:
`chash`, `nchash`, `bytes`, `geohash`, `h3`, `timehash` and `uuidhash`. It also exports
[`col`][polars_hash.col] and [`concat_str`][polars_hash.concat_str], which are
typed wrappers around `pl.col` and `pl.concat_str`, and
[`hash_rows`][polars_hash.hash_rows], which hashes a whole row.

Examples:
    >>> df = pl.DataFrame({"foo": ["hello_world"]})
    >>> df.select(plh.col("foo").chash.sha2_256()).item()
    '35072c1ae546350e0bfa7ab11d49dc6f129e72ccd57ec7eb671225bbd197c8f1'
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, cast

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
    """Cryptographic hash functions, on `pl.Expr` as `.chash`.

    Every expression here accepts Utf8 or Binary and gives a lowercase
    hexadecimal string. A digest reads bytes, so the data type of the input does
    not change the result. A null input gives a null output.
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def sha2_256(self) -> pl.Expr:
        """SHA-256 from the SHA-2 family.

        Returns:
            Utf8 with 64 characters.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").chash.sha2_256()).item()
            '35072c1ae546350e0bfa7ab11d49dc6f129e72ccd57ec7eb671225bbd197c8f1'
        """
        return _plugin("sha2_256", self._expr)

    def sha2_512(self) -> pl.Expr:
        """SHA-512 from the SHA-2 family.

        Returns:
            Utf8 with 128 characters.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").chash.sha2_512()).item()[:32]
            '94f427efefa74c1230c3e93c35104dcb'
        """
        return _plugin("sha2_512", self._expr)

    def sha2_384(self) -> pl.Expr:
        """SHA-384 from the SHA-2 family.

        Returns:
            Utf8 with 96 characters.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").chash.sha2_384()).item()[:32]
            '7f251a65acbe92af4c6a6d624c0860d9'
        """
        return _plugin("sha2_384", self._expr)

    def sha2_224(self) -> pl.Expr:
        """SHA-224 from the SHA-2 family.

        Returns:
            Utf8 with 56 characters.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").chash.sha2_224()).item()
            '69c9392f54e5a0e0fff8945e9ed6475ef89236092a52b2005776912c'
        """
        return _plugin("sha2_224", self._expr)

    def sha3_256(self) -> pl.Expr:
        """SHA3-256 from the SHA-3 family.

        The digest has the size of [`sha2_256()`][polars_hash.CryptographicHashingNameSpace.sha2_256]
        and a different construction. The two expressions do not give the same value.

        Returns:
            Utf8 with 64 characters.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").chash.sha3_256()).item()
            'fed30406b832b6c457e1e3605016eadfe7b57074c050e16ce2321de734ab29f4'
        """
        return _plugin("sha3_256", self._expr)

    def sha3_512(self) -> pl.Expr:
        """SHA3-512 from the SHA-3 family.

        Returns:
            Utf8 with 128 characters.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").chash.sha3_512()).item()[:32]
            '3d96f9b16a74980badc6aa05f8f102d7'
        """
        return _plugin("sha3_512", self._expr)

    def sha3_384(self) -> pl.Expr:
        """SHA3-384 from the SHA-3 family.

        Returns:
            Utf8 with 96 characters.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").chash.sha3_384()).item()[:32]
            'd407e9fb45a350dce0d557f4d3d514f0'
        """
        return _plugin("sha3_384", self._expr)

    def sha3_224(self) -> pl.Expr:
        """SHA3-224 from the SHA-3 (Keccak) family.

        Returns:
            Utf8 with 56 characters.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").chash.sha3_224()).item()
            'e24c066a49e260ba46a7b73d5d2374bfe86670be8ebbdf547bfce343'
        """
        return _plugin("sha3_224", self._expr)

    def sha3_shake128(self, *, length: int) -> pl.Expr:
        """SHAKE128, the extendable-output function from the SHA-3 family.

        Every other digest in this namespace has a fixed size. Here you set the
        size.

        Args:
            length: The digest size in bytes. The hexadecimal output has two
                characters for each byte. A `length` of 0 gives an empty string.
                A negative `length` raises `ComputeError`.

        Returns:
            Utf8 with `2 × length` characters.

        Note:
            A short output is the start of a longer output for the same input.
            If you cut a long digest to *n* bytes, you get the digest that
            `length=n` gives.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").chash.sha3_shake128(length=10)).item()
            '6b57b385e070e3534257'
        """
        return _plugin("sha3_shake128", self._expr, length=length)

    def blake3(self) -> pl.Expr:
        """BLAKE3 with the default 256-bit output.

        BLAKE3 is much faster than SHA-2. Use it for large quantities of data.

        Returns:
            Utf8 with 64 characters.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").chash.blake3()).item()
            '9833e5324eb2400de814730f4e92810905351bc0451e10b75847210c1d7c37ed'

            Each expression in this namespace also hashes the bytes of a Binary
            column:

            >>> pl.select(pl.lit(b"my_bytes").chash.blake3()).item()
            '4656d42e3468733c9316ef5d4e4488682fc41ad441644ca63cde6aced8378605'
        """
        return _plugin("blake3", self._expr)

    def hmac_sha256(self, *, key: str) -> pl.Expr:
        """Keyed HMAC-SHA256 (RFC 2104).

        The digest is a function of the input and the key. One input with two
        different keys gives two different digests.

        Args:
            key: The key. It can have any length, and an empty key is permitted.
                polars-hash expands the key one time for each expression, not one
                time for each row.

        Returns:
            Utf8 with 64 characters.

        Warning:
            polars-hash writes `key` into the keyword arguments of the
            expression. The key therefore appears in the output of `explain()`
            and in each plan that you cache or write to a log.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").chash.hmac_sha256(key="secret")).item()
            'e0f5b5bb7264e77b340a55a694a6c9ca4edc035c394c703a0408f099563be1ca'
        """
        return _plugin("hmac_sha256", self._expr, key=key)


@pl.api.register_expr_namespace("nchash")
class NonCryptographicHashingNameSpace:
    """Non-cryptographic hash functions, on `pl.Expr` as `.nchash`.

    These expressions are fast and their output is stable. Each one accepts Utf8
    or Binary. A digest reads bytes, so the data type of the input does not
    change the result, and a Utf8 column gives the digest of its UTF-8 bytes.
    A null input gives a null output. Any other input type raises
    ``ComputeError: expected `String` or `Binary` input``.

    Every expression with a `UInt128` output also takes `return_binary=True`,
    which writes the same hash as 16 Binary bytes.

    Warning:
        Polars encodes `UInt128` as a private Arrow type, so `to_arrow()` and
        `to_pandas()` raise `ArrowInvalid` on such a column, and `to_numpy()`
        fails. `write_parquet`, `write_ipc`, joins, `group_by` and sorting all
        work. Set `return_binary=True` if the column has to leave Polars:
        Binary travels everywhere. A cast to `pl.Binary` is not the same thing.
        It writes the decimal digits of the integer, not its 16 bytes.
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def wyhash(self) -> pl.Expr:
        """wyhash with 64-bit output.

        This expression is very fast. You cannot set the seed. It is always 0.

        Returns:
            UInt64.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.wyhash()).item()
            16737367591072095403

            Each expression in this namespace also hashes the bytes of a Binary
            column:

            >>> pl.select(pl.lit(b"my_bytes").nchash.wyhash()).item()
            5112362246832359110
        """
        return _plugin("wyhash", self._expr)

    def sha1(self) -> pl.Expr:
        """SHA-1, hexadecimal.

        Returns:
            Utf8 with 40 characters.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.sha1()).item()
            'e4ecd6fc11898565af24977e992cea0c9c7b7025'
        """
        return _plugin("sha1", self._expr)

    def md5(self) -> pl.Expr:
        """MD5, hexadecimal.

        Returns:
            Utf8 with 32 characters.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.md5()).item()
            '99b1ff8f11781541f7f89f9bd41c4a17'
        """
        return _plugin("md5", self._expr)

    def murmur32(self, *, seed: int = 0) -> pl.Expr:
        """MurmurHash3, x86 32-bit variant.

        Many systems have an implementation of this algorithm, for example Spark,
        Kafka and bloom filter libraries.

        Args:
            seed: A value in the range of a `u32`, that is 0 to 4294967295.

        Returns:
            UInt32.

        Note:
            With the default seed, an empty string gives 0. With a different
            seed, an empty string gives a value that is not 0. This is the
            correct MurmurHash3 result. It is not a null value in the output.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.murmur32()).item()
            3531928679
            >>> df.select(plh.col("foo").nchash.murmur32(seed=42)).item()
            259561949
        """
        return _plugin("murmur32", self._expr, seed=seed)

    def murmur128(self, *, seed: int = 0, return_binary: bool = False) -> pl.Expr:
        """MurmurHash3, x64 128-bit variant.

        The integer matches `mmh3.hash128(..., signed=False)`.

        Args:
            seed: A value in the range of a `u32`. The 128-bit variant also uses
                a 32-bit seed.
            return_binary: Write the hash as the 16 digest bytes. MurmurHash3
                writes its digest as two little-endian halves, so these are the
                bytes `mmh3.hash_bytes()` gives. Use this where the target of a
                write has no 128-bit integer. The bytes and the integer hold the
                same hash.

        Returns:
            UInt128, or Binary with `return_binary=True`.

        Warning:
            Releases up to 0.7.0 returned the 16 digest bytes, and 0.8.0 changed
            the output to `UInt128`. Only the container changed.
            `int.from_bytes(old, "little")` converts a stored value. From 0.9.1,
            `return_binary=True` gives the 0.7.0 bytes again, byte for byte.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.murmur128()).item()
            134986332493155497415370161450594282648
            >>> df.select(plh.col("foo").nchash.murmur128(seed=42)).item()
            128378975539535818103252123378652633995
            >>> df.select(
            ...     plh.col("foo").nchash.murmur128(return_binary=True).bin.encode("hex")
            ... ).item()
            '982cf39e1c1aa55d1b079716076c8d65'
        """
        return _plugin("murmur128", self._expr, seed=seed, return_binary=return_binary)

    def xxhash32(self, *, seed: int = 0) -> pl.Expr:
        """XXH32, the original 32-bit xxHash.

        Args:
            seed: A value in the range of a `u32`. A value outside that range,
                or `None`, raises `expected u32`.

        Returns:
            UInt32.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.xxhash32()).item()
            1605956417
            >>> df.select(plh.col("foo").nchash.xxhash32(seed=42)).item()
            1544934469
        """
        return _plugin("xxhash32", self._expr, seed=seed)

    def xxhash64(self, *, seed: int = 0) -> pl.Expr:
        """XXH64, the 64-bit xxHash.

        [`xxh3_64()`][polars_hash.NonCryptographicHashingNameSpace.xxh3_64] is
        faster. Use `xxhash64()` when you must get the same values as a different
        system.

        Args:
            seed: A value in the range of a `u64`.

        Returns:
            UInt64.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.xxhash64()).item()
            5654987600477331689
            >>> df.select(plh.col("foo").nchash.xxhash64(seed=42)).item()
            17477110538672341566
        """
        return _plugin("xxhash64", self._expr, seed=_encode_u64_seed(seed))

    def xxh3_64(self, *, seed: int = 0) -> pl.Expr:
        """XXH3 with 64-bit output.

        For usual string lengths, this is the fastest expression in the
        namespace.

        Args:
            seed: A value in the range of a `u64`.

        Returns:
            UInt64.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.xxh3_64()).item()
            7060460777671424209
            >>> df.select(plh.col("foo").nchash.xxh3_64(seed=42)).item()
            827481053383045869
        """
        return _plugin("xxh3_64", self._expr, seed=_encode_u64_seed(seed))

    def xxh3_128(
        self,
        *,
        seed: int = 0,
        return_binary: bool = False,
        byte_order: Literal["little", "big"] | None = None,
    ) -> pl.Expr:
        """XXH3 with 128-bit output.

        The integer matches `xxhash.xxh128_intdigest()`. Formatted as 32
        hexadecimal digits with `f"{value:032x}"`, it is the canonical XXH128
        digest, the string `xxh128_hexdigest()` gives.

        Args:
            seed: A value in the range of a `u64`.
            return_binary: Write the hash as 16 Binary bytes. Use this where the
                target of a write has no 128-bit integer.
            byte_order: How those bytes read, with `return_binary=True`. `"big"`
                is the digest XXH3 itself writes, the bytes
                `xxhash.xxh128_digest()` gives. `"little"` is the integer least
                significant byte first, which is what releases up to 0.7.0 wrote.
                The default is `"little"` and it warns, because the compatible
                order is not the canonical one. Name an order to accept it
                silently.

        Returns:
            UInt128, or Binary with `return_binary=True`.

        Warning:
            0.8.0 changed this output from Binary to `UInt128`. Up to 0.7.0 the
            16 bytes held the value in the reverse of the canonical order. The
            integer now agrees with the reference. To read data hashed by an
            older release, reverse the old bytes:
            `int.from_bytes(old, "little")`.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.xxh3_128()).item()
            253649469245435599925940275794906345219
            >>> f"{253649469245435599925940275794906345219:032x}"
            'bed31c5eaf3dc62267fb185e21fe6f03'
            >>> df.select(plh.col("foo").nchash.xxh3_128(seed=42)).item()
            314735830047873782861649874643137875266

            The canonical digest bytes, and the bytes of releases up to 0.7.0:

            >>> df.select(
            ...     plh.col("foo")
            ...     .nchash.xxh3_128(return_binary=True, byte_order="big")
            ...     .bin.encode("hex")
            ... ).item()
            'bed31c5eaf3dc62267fb185e21fe6f03'
            >>> df.select(
            ...     plh.col("foo")
            ...     .nchash.xxh3_128(return_binary=True, byte_order="little")
            ...     .bin.encode("hex")
            ... ).item()
            '036ffe215e18fb6722c63daf5e1cd3be'
        """
        if byte_order not in (None, "little", "big"):
            msg = f"`byte_order` must be 'little' or 'big', got {byte_order!r}"
            raise ValueError(msg)
        if return_binary and byte_order is None:
            warnings.warn(
                "xxh3_128 binary defaults to little-endian for compatibility with older polars-hash versions. "
                "XXH3 canonicalises big-endian. Set byte_order to silence this.",
                UserWarning,
                stacklevel=2,
            )
        return _plugin(
            "xxh3_128",
            self._expr,
            seed=_encode_u64_seed(seed),
            return_binary=return_binary,
            big_endian=byte_order == "big",
        )

    def farmhash32(self) -> pl.Expr:
        """Google FarmHash `fingerprint32`.

        The fingerprint functions give the same value on all platforms. BigQuery
        uses them for its `FARM_FINGERPRINT` function. This expression has no
        seed.

        Returns:
            UInt32.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello world"]})
            >>> df.select(plh.col("foo").nchash.farmhash32()).item()
            430397466
        """
        return _plugin("farmhash32", self._expr)

    def farmhash64(self) -> pl.Expr:
        """Google FarmHash `fingerprint64`, the 64-bit fingerprint.

        This expression has no seed.

        Returns:
            UInt64.

        Note:
            The `FARM_FINGERPRINT` function in BigQuery gives the same 64 bits as
            a signed `INT64`. To compare the two results, use `.cast(pl.Int64)`
            on the polars-hash output.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello world"]})
            >>> df.select(plh.col("foo").nchash.farmhash64()).item()
            6381520714923946011
        """
        return _plugin("farmhash64", self._expr)

    def cityhash32(self) -> pl.Expr:
        """Google CityHash `CityHash32`, from CityHash v1.1.1.

        FarmHash replaced CityHash, so use
        [`farmhash32()`][polars_hash.NonCryptographicHashingNameSpace.farmhash32]
        for new work. Use `cityhash32()` when you must get the same values as a
        different system. This expression has no seed. `CityHash32` takes none.

        Returns:
            UInt32.

        Warning:
            Every CityHash expression here gives the values of v1.1.1, the last
            release Google published. `CityHash64` changed during the v1.0 series
            and `CityHash128` changed after v1.0.3, so a system built on an
            earlier release gives a different value for the same input. Check
            which release the other system uses before you compare.

        Note:
            FarmHash reuses CityHash for short input, so `cityhash32()` and
            `farmhash32()` give the same value for input up to 12 bytes, as do
            `cityhash64()` and `farmhash64()` up to 32 bytes. They part above
            those lengths. The equal values are not a defect.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.cityhash32()).item()
            1719156559
        """
        return _plugin("cityhash32", self._expr)

    def cityhash64(self, *, seed: int | None = None) -> pl.Expr:
        """Google CityHash `CityHash64`, from CityHash v1.1.1.

        Args:
            seed: A value in the range of a `u64`, or `None` for the unseeded
                algorithm.

        Returns:
            UInt64.

        Warning:
            A seed of 0 is not the same as no seed. Without a seed this
            expression is `CityHash64`. With one it is `CityHash64WithSeed`, a
            separate function that gives a different value for every seed, 0
            included. This is why the seed defaults to `None`.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.cityhash64()).item()
            15605398435621216523
            >>> df.select(plh.col("foo").nchash.cityhash64(seed=42)).item()
            10175920941468920074
        """
        if seed is None:
            return _plugin("cityhash64", self._expr)

        return _plugin("cityhash64_with_seed", self._expr, seed=_encode_u64_seed(seed))

    def cityhash128(self, *, return_binary: bool = False) -> pl.Expr:
        """Google CityHash `CityHash128`, from CityHash v1.1.1.

        `CityHash128WithSeed` is not wrapped, so this expression has no seed.

        Args:
            return_binary: Write the hash as 16 Binary bytes, least significant
                byte first. The bytes and the integer hold the same hash.

        Returns:
            UInt128, or Binary with `return_binary=True`.

        Note:
            C++ returns `CityHash128` as a pair. This expression packs it the way
            `python-cityhash` does, `Uint128Low64(h) << 64 | Uint128High64(h)`,
            so the C++ *low* word is the *high* half of the integer. A system
            that composes the halves the other way round, or that stores the raw
            16 bytes, needs a word swap before the values compare equal.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.cityhash128()).item()
            133423608296839006301901834072762183026
        """
        return _plugin("cityhash128", self._expr, return_binary=return_binary)

    def crc32c(
        self,
        *,
        return_binary: bool = False,
        byte_order: Literal["little", "big"] = "little",
    ) -> pl.Expr:
        """CRC-32C (Castagnoli), the variant iSCSI, SCTP and libcsp use.

        This is a checksum, not a general-purpose hash. It is fast to compute and
        it detects the usual transmission and storage errors, but it does not
        resist a deliberate collision.

        Args:
            return_binary: Write the checksum as 4 Binary bytes.
            byte_order: The order of those bytes. `"little"` is the integer's own
                bytes. `"big"` is network byte order, the order a checksum field
                on the wire usually has.

        Returns:
            UInt32, or Binary with `return_binary=True`.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.crc32c()).item()
            1680342080
            >>> df.select(
            ...     plh.col("foo")
            ...     .nchash.crc32c(return_binary=True, byte_order="big")
            ...     .bin.encode("hex")
            ... ).item()
            '6427fc40'
        """
        if byte_order not in ("little", "big"):
            msg = f"`byte_order` must be 'little' or 'big', got {byte_order!r}"
            raise ValueError(msg)
        return _plugin(
            "crc32c",
            self._expr,
            return_binary=return_binary,
            big_endian=byte_order == "big",
        )

    def gxhash32(self, *, seed: int = 0) -> pl.Expr:
        """GxHash with 32-bit output.

        GxHash reaches its speed through the AES block cipher, which the CPU runs
        as a single instruction. It is the fastest expression in the namespace
        for input above about a hundred bytes. Below that,
        [`xxh3_64()`][polars_hash.NonCryptographicHashingNameSpace.xxh3_64] is
        the one to beat.

        Args:
            seed: A value in the range of a `u64`. GxHash takes an `i64` seed
                upstream, and this namespace presents every 64-bit seed as a
                `u64`, so a seed at or above `2**63` is the upstream seed minus
                `2**64`. Both reach the same 64 bits.

        Returns:
            UInt32.

        Warning:
            GxHash needs a CPU with AES instructions. The algorithm has no
            software fallback. The published wheels cover x86, x86-64 and
            aarch64, and every one of them is built with the instructions
            enabled, so a CPU without them stops the process the moment a GxHash
            expression runs. On x86 the instructions arrived with Westmere in
            2010 and every processor since has them. On ARM they are an optional
            extension: Apple silicon and server parts have them, and some small
            boards, such as the Raspberry Pi 4, do not.

            The instructions are enabled for the whole build and not for GxHash
            alone, so on x86 `ahash`, which the [`h3`](h3.md) namespace
            pulls in, switches to its AES-NI implementation as well, and the
            `h3` expressions come to need them too. On aarch64 `ahash` keeps its portable path, so there GxHash
            is the only namespace affected. There are no `linux-armv7` or
            `linux-ppc64le` wheels from 0.8.0 on, because GxHash cannot be built
            for either.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.gxhash32()).item()
            2751540945
            >>> df.select(plh.col("foo").nchash.gxhash32(seed=42)).item()
            3382299372
        """
        return _plugin("gxhash32", self._expr, seed=_encode_u64_seed(seed))

    def gxhash64(self, *, seed: int = 0) -> pl.Expr:
        """GxHash with 64-bit output.

        [`gxhash32()`][polars_hash.NonCryptographicHashingNameSpace.gxhash32]
        gives the requirements of the algorithm and the meaning of the seed.

        Args:
            seed: A value in the range of a `u64`. Every GxHash expression is
                seeded and the default seed is 0. Unlike
                [`cityhash64()`][polars_hash.NonCryptographicHashingNameSpace.cityhash64],
                there is no unseeded form to differ from.

        Returns:
            UInt64.

        Warning:
            GxHash holds its output stable across platforms, but only within a
            major version. polars-hash pins GxHash 3 exactly, so the values here
            do not change without a release that says so. A system on GxHash 2
            gives different values for the same input and seed.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.gxhash64()).item()
            2180020304351407825
            >>> df.select(plh.col("foo").nchash.gxhash64(seed=42)).item()
            15254170022685821676
        """
        return _plugin("gxhash64", self._expr, seed=_encode_u64_seed(seed))

    def gxhash128(self, *, seed: int = 0, return_binary: bool = False) -> pl.Expr:
        """GxHash with 128-bit output.

        [`gxhash32()`][polars_hash.NonCryptographicHashingNameSpace.gxhash32]
        gives the requirements of the algorithm.

        Args:
            seed: A value in the range of a `u64`.
            return_binary: Write the hash as 16 Binary bytes, least significant
                byte first. The bytes and the integer hold the same hash.

        Returns:
            UInt128, or Binary with `return_binary=True`.

        Note:
            GxHash builds a single 128-bit state and each width reads the low
            part of it, so `gxhash32()` is the low 32 bits of `gxhash64()`, which
            is the low 64 bits of `gxhash128()`. Ask for the width you need. The
            narrow ones cost no less than the wide one.

        Examples:
            >>> df = pl.DataFrame({"foo": ["hello_world"]})
            >>> df.select(plh.col("foo").nchash.gxhash128()).item()
            56218077491375249900279963678916292305
            >>> df.select(plh.col("foo").nchash.gxhash128(seed=42)).item()
            11136336363892181958542060125951740652
        """
        return _plugin(
            "gxhash128",
            self._expr,
            seed=_encode_u64_seed(seed),
            return_binary=return_binary,
        )


@pl.api.register_expr_namespace("bytes")
class BytesNameSpace:
    """Byte encoding, on `pl.Expr` as `.bytes`.

    These expressions change a value into its own bytes, so you can send the
    result to any hasher in [`nchash`](non-cryptographic.md) or
    [`chash`](cryptographic.md), or write it out directly. Each one accepts
    Boolean, Int8/16/32/64, UInt8/16/32/64, Float32/64, Utf8 or Binary.

    Each type keeps its own width: Int8 makes 1 byte, Int32 makes 4, Float64
    makes 8, and so on. The two expressions differ only in the order of the bytes
    of that width. A value that is already bytes, Utf8 or Binary, has no byte
    order of its own, and it passes through unchanged either way.

    Tip:
        This namespace encodes a value. It does not hash one. Send the result to a
        hasher to get a hash of the value itself, and not of a string form of it:
        `plh.col("id").cast(pl.Int64).bytes.to_le().nchash.murmur32()`.
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def to_le(self) -> pl.Expr:
        r"""Encode the value as its own bytes, least significant byte first.

        Returns:
            Binary, of the width of the input type. Boolean and the two 8-bit
                integer types write one byte, Boolean as `0x00` or `0x01`. There
                is no second byte to order, so `to_le()` and `to_be()` agree on
                those three types. Utf8 writes its raw UTF-8 bytes and Binary
                passes through, and the byte order changes neither.

        Raises:
            ComputeError: The input is a type this namespace does not accept, for
                example Date or Decimal. The message is ``expected a numeric,
                Boolean, String or Binary input, got `date` ``.

        Note:
            To get a different width, cast first.
            `plh.col("x").cast(pl.Int64).bytes.to_le()` widens a narrower integer
            to 8 bytes before it encodes, sign-extended as any polars numeric cast
            is.

        Examples:
            >>> df = pl.DataFrame({"literal": [1]}, schema={"literal": pl.Int32})
            >>> df.select(plh.col("literal").bytes.to_le()).item()
            b'\x01\x00\x00\x00'
        """
        return _plugin("bytes_to_le", self._expr)

    def to_be(self) -> pl.Expr:
        r"""Encode the value as its own bytes, most significant byte first.

        Everything on [`to_le()`][polars_hash.BytesNameSpace.to_le] applies here,
        except for the order of the bytes.

        Returns:
            Binary, of the width of the input type.

        Raises:
            ComputeError: The input is a type this namespace does not accept.

        Examples:
            >>> df = pl.DataFrame({"literal": [1]}, schema={"literal": pl.Int32})
            >>> df.select(plh.col("literal").bytes.to_be()).item()
            b'\x00\x00\x00\x01'
        """
        return _plugin("bytes_to_be", self._expr)


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
    """Geohash encode, decode and neighbors, on `pl.Expr` as `.geohash`.

    A geohash is a base-32 string. It gives the name of a rectangular cell on the
    earth. The prefixes are hierarchical: cell `9q60y` contains every point whose
    geohash starts with `9q60y`. A `starts_with` filter is therefore a query for
    a rectangular area, and two geohashes with a long identical prefix are near
    to each other.

    The expressions that take coordinates read a Struct with a `latitude` field
    and a `longitude` field. Both fields must be Float32 or Float64, and
    polars-hash casts Float32 to Float64. It finds the fields by name, so their
    order is not important, and it ignores the other fields.
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def to_coords(self) -> pl.Expr:
        """Decode a geohash string to the coordinates of the cell center.

        If you encode a coordinate and then decode the geohash, the result is
        different from the initial coordinate. The difference is less than the
        size of the cell.

        Returns:
            A Struct with two Float64 fields, `longitude` first and `latitude`
                second.

        Note:
            [`from_coords()`][polars_hash.GeoHashingNameSpace.from_coords] finds
            the fields of its input struct by name, but `to_coords()` writes
            `longitude` before `latitude`. Use `unnest`, or select the fields by
            name and not by position.

        Raises:
            ComputeError: The string is not a geohash. A null row gives a struct
                of two null fields, and raises nothing.

        Examples:
            >>> df = pl.DataFrame({"h": ["9q60y60rhs"]})
            >>> coords = df.select(plh.col("h").geohash.to_coords()).item()
            >>> round(coords["longitude"], 4), round(coords["latitude"], 4)
            (-120.6623, 35.3003)
        """
        return _plugin("ghash_decode", self._expr)

    def from_coords(self, len: int | str | pl.Expr = 12) -> pl.Expr:
        """Encode a coordinate struct to a geohash string.

        Args:
            len: The number of characters, from 1 to 12. An `int` applies to
                every row. polars-hash reads a `str` as a column name and a
                `pl.Expr` as an expression, and the precision is then different
                for each row. All the integer types are permitted, signed and
                unsigned, and polars-hash casts the value to Int64. It also
                accepts a float and truncates it, so `5.9` gives 5.

        Returns:
            Utf8.

        Raises:
            ComputeError: `len` is less than 1 or more than 12
                (`Invalid length specified: 13. Accepted values are between 1
                and 12, inclusive`); `len` is null (`Length may not be null`); a
                coordinate is outside its range
                (`invalid coordinate range: COORD(-120.6623 91.0)`); or a
                coordinate field is not a float
                (`Latitude input needs to be float`). A null latitude or a null
                longitude gives null for that row, and raises nothing.

        Note:
            Each character adds about 5 bits, so the cells become small quickly.
            A `len` of 1 covers about 5000 × 5000 km, 3 covers 156 × 156 km, 5
            covers 4.9 × 4.9 km, 7 covers 153 × 153 m, 9 covers 4.8 × 4.8 m, and
            12 covers 3.7 × 1.9 cm.

        Examples:
            >>> df = pl.DataFrame(
            ...     {"coord": [{"longitude": -120.6623, "latitude": 35.3003}]},
            ...     schema={
            ...         "coord": pl.Struct(
            ...             [
            ...                 pl.Field("longitude", pl.Float64),
            ...                 pl.Field("latitude", pl.Float64),
            ...             ]
            ...         )
            ...     },
            ... )
            >>> df.select(plh.col("coord").geohash.from_coords(5)).item()
            '9q60y'

            A column name or an expression gives one precision for each row:

            >>> df.with_columns(n=pl.lit(3)).select(
            ...     plh.col("coord").geohash.from_coords("n")
            ... ).item()
            '9q6'
        """
        return _plugin("ghash_encode", [self._expr, _length_expr(len)])

    def neighbors(self) -> pl.Expr:
        """Give the eight geohash cells around a cell.

        The neighbor cells have the precision of the input cell. Use this
        expression to find near points: a point near the edge of a cell can have
        near points in an adjacent cell, and a search over the cell and its eight
        neighbors also finds those points.

        Returns:
            A Struct with eight Utf8 fields in this order: `n`, `ne`, `e`, `se`,
                `s`, `sw`, `w`, `nw`.

        Raises:
            ComputeError: The string is not a geohash. A null row gives a struct
                of eight null fields, and raises nothing.

        Examples:
            >>> df = pl.DataFrame({"h": ["sp1xk2m6194y"]})
            >>> df.select(plh.col("h").geohash.neighbors()).item()["n"]
            'sp1xk2m6194z'
        """
        return _plugin("ghash_neighbors", self._expr)


@pl.api.register_expr_namespace("h3")
class H3NameSpace:
    """The H3 cell index, on `pl.Expr` as `.h3`.

    H3 is the hierarchical hexagonal grid from Uber. Refer to the
    [H3 website](https://h3geo.org/). H3 divides the earth into hexagons, where a
    geohash uses rectangles. Each hexagonal cell has six neighbors and the
    distance to each neighbor is the same, so H3 is better than a geohash for
    aggregation and area analysis. A geohash keeps the more simple property that
    a prefix gives a rectangular area.

    Note:
        This namespace encodes only. To decode a cell index or to find the
        neighbors of a cell, use the
        [h3 Python package](https://pypi.org/project/h3/) on the output column.
        The [`geohash`](geohash.md) namespace has both operations.
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def from_coords(self, len: int = 12) -> pl.Expr:
        """Encode a coordinate struct to an H3 cell index.

        The input is the Struct that
        [`geohash`](geohash.md) reads, a `latitude` field
        and a `longitude` field, both float.

        Args:
            len: The H3 resolution, from 1 to 15. A column name or a `pl.Expr`
                also works at run time and gives one resolution for each row, but
                the type hint does not show this yet. polars-hash casts the value
                to Int64. Resolution 0 is not permitted, although H3 has it: it
                has 122 base cells and is not a usual level for aggregation.

        Returns:
            Utf8, the standard lowercase hexadecimal cell index with 15
                characters. The value holds the resolution, so two indexes of
                different resolutions are always different.

        Raises:
            ComputeError: `len` is less than 1 or more than 15
                (`expected resolution between 1 and 15, got 16`); `len` is null
                (`Length may not be null`); a coordinate is outside its range or
                is NaN or infinite
                (`invalid coordinate range: latitude 91, longitude -120.6623`);
                or a coordinate field is not a float
                (`Latitude input needs to be float`). A null latitude or a null
                longitude gives null for that row, and raises nothing.

        Note:
            The approximate average edge length of a hexagon: resolution 1 gives
            483 km, 3 gives 69 km, 5 gives 9.9 km, 7 gives 1.4 km, 9 gives 200 m,
            11 gives 29 m, 13 gives 4.1 m, and 15 gives 0.6 m.

        Examples:
            >>> df = pl.DataFrame(
            ...     {"coord": [{"longitude": -120.6623, "latitude": 35.3003}]},
            ...     schema={
            ...         "coord": pl.Struct(
            ...             [
            ...                 pl.Field("longitude", pl.Float64),
            ...                 pl.Field("latitude", pl.Float64),
            ...             ]
            ...         )
            ...     },
            ... )
            >>> df.select(plh.col("coord").h3.from_coords(5)).item()
            '8529adc7fffffff'
        """
        return _plugin("h3_encode", [self._expr, _length_expr(len)])


@pl.api.register_expr_namespace("timehash")
class TimeHashingNameSpace:
    """Time buckets, on `pl.Expr` as `.timehash`.

    A timehash is a short string that names the window of time an instant falls
    in. Two instants in the same window get the same hash, and a shorter hash
    names a wider window, so a comparison of prefixes is a coarser bucket.
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def from_datetime(
        self, precision: int | str | pl.Expr = 10, *, strict: bool = True
    ) -> pl.Expr:
        """Encode an instant to the timehash of the window that holds it.

        Datetime and Date columns work directly. Epoch seconds may be Float64 or
        any integer type. Float32 cannot hold one closely enough to land in the
        right window. The timestamp must fall between 1970-01-01 and 2098-01-01.

        Args:
            precision: The number of characters in the hash, from 1 to 32. A
                higher precision means a shorter window: 10 covers about 4
                seconds and 8 about 4 minutes. Past about 18 the hash stops
                changing for a present-day timestamp, and the extra characters
                are padding.
            strict: With `False`, a timestamp outside the range gives null
                instead of raising. Precision stays strict either way.

        Returns:
            Utf8.

        Tip:
            A `when`/`then` guard cannot skip an out-of-range timestamp, because
            polars evaluates both branches over the whole column. Use
            `strict=False` instead.

        Examples:
            >>> from datetime import datetime
            >>> df = pl.DataFrame({"t": [datetime(2024, 5, 17, 12, 30, 45)]})
            >>> df.select(plh.col("t").timehash.from_datetime()).item()
            'bb1c00aaf0'

            A lower precision gives a shorter hash and a wider window:

            >>> df.select(plh.col("t").timehash.from_datetime(8)).item()
            'bb1c00aa'
        """
        return _plugin(
            "thash_encode", [self._expr, _length_expr(precision)], strict=strict
        )

    def to_datetime(self) -> pl.Expr:
        """Decode a timehash to the midpoint of the window it names.

        Returns:
            Datetime in microseconds, UTC.

        Note:
            The hash holds an instant and not a wall clock, so the original time
            zone is gone. Use `.dt.convert_time_zone(tz)` for another zone. The
            midpoint is not the instant you encoded. It is the center of the
            window that instant fell in, so a round trip is exact only up to the
            precision you used.

        Examples:
            >>> df = pl.DataFrame({"h": ["bb1c00aaf0"]})
            >>> df.select(
            ...     plh.col("h").timehash.to_datetime().dt.strftime("%Y-%m-%d %H:%M:%S")
            ... ).item()
            '2024-05-17 12:30:46'
        """
        return _plugin("thash_decode", self._expr)

    def neighbors(self) -> pl.Expr:
        """Give the windows on either side of the one the hash names.

        Returns:
            A Struct with the Utf8 fields `before` and `after`.

        Examples:
            >>> df = pl.DataFrame({"h": ["bb1c00aaf0"]})
            >>> df.select(plh.col("h").timehash.neighbors()).item()
            {'before': 'bb1c00aaef', 'after': 'bb1c00aaf1'}
        """
        return _plugin("thash_neighbors", self._expr)


class UUIDNamespace(str, Enum):
    """The four RFC 4122 namespaces for a UUID v5.

    This is a `str` enum, so a member and its value are equivalent arguments.
    `DNS` names a fully qualified domain name, `URL` a URL, `OID` an ISO object
    identifier, and `X500` an X.500 distinguished name.

    Examples:
        >>> df = pl.DataFrame({"foo": ["https://example.com"]})
        >>> df.select(plh.col("foo").uuidhash.uuid5(plh.UUIDNamespace.URL)).item()
        '4fd35a71-71ef-5a55-a9d9-aa75c889a6d0'
        >>> df.select(plh.col("foo").uuidhash.uuid5("url")).item()
        '4fd35a71-71ef-5a55-a9d9-aa75c889a6d0'
    """

    DNS = "dns"
    URL = "url"
    OID = "oid"
    X500 = "x500"


@pl.api.register_expr_namespace("uuidhash")
class UUIDHashNameSpace:
    """Deterministic UUID version 5, on `pl.Expr` as `.uuidhash`.

    A v5 UUID is a SHA-1 digest of a namespace UUID and a name, in UUID format.
    The result is deterministic: the same namespace and the same name always give
    the same UUID, so a v5 UUID is a key for a value you have. A null input gives
    a null output.
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def uuid5(self, namespace: UUIDNamespace | str = UUIDNamespace.DNS) -> pl.Expr:
        """Make a UUID v5 from a Utf8 or Binary column.

        Args:
            namespace: A [standard namespace][polars_hash.UUIDNamespace]:
                `"dns"`, `"url"`, `"oid"` or `"x500"`. Uppercase and lowercase
                letters are equivalent. Every other string is a custom namespace
                UUID, and two different namespaces give two different UUIDs for
                the same input.

        Returns:
            Utf8, the 36-character format with hyphens.

        Raises:
            ComputeError: `namespace` is null (`Namespace must be provided`), or
                it is neither a standard name nor a correct UUID
                (`Invalid namespace '{value}': {reason}`).

        Examples:
            >>> df = pl.DataFrame({"literal": ["hello", None, "world"]})
            >>> df.select(plh.col("literal").uuidhash.uuid5()).to_series().to_list()
            ['9342d47a-1bab-5709-9869-c840b2eac501', None, 'b3a4c24e-f57a-5448-b81b-a643f6768036']

            A custom namespace:

            >>> tenant = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
            >>> df.select(plh.col("literal").uuidhash.uuid5(tenant)).item(0, 0)
            'f9f6bc57-bc58-5993-8b8d-ee0ddf417610'
        """
        return _plugin("uuid5", [self._expr, pl.lit(namespace)])

    def uuid5_concat(self, other: pl.Expr, default: str | None = None) -> pl.Expr:
        """Concatenate two Utf8 columns and make a UUID v5 in the DNS namespace.

        Use this to make a key from two columns with one expression, in place of
        a `concat_str` and a [`uuid5`][polars_hash.UUIDHashNameSpace.uuid5].

        The two columns are not equivalent. A null in the first column gives
        null. A null in `other` gives the UUID of the first value and `default`,
        or of the first value alone when `default` is `None`.

        Args:
            other: The second column, which polars-hash puts after the first. It
                must be Utf8, and polars-hash casts it to Utf8 first if you set
                `default`.
            default: The value that replaces a null in `other`. With `None`, a
                null in `other` becomes an empty string.

        Returns:
            Utf8.

        Raises:
            ComputeError: `default` is null (`Default value may not be null`).

        Note:
            This expression adds no separator, so `("ab", "c")` and `("a", "bc")`
            give the same UUID. If your data can have this condition, make the
            key with a separator that the data does not contain, with
            [`concat_str`][polars_hash.concat_str]:
            `plh.concat_str("id", "side", separator="|").uuidhash.uuid5()`. That
            form also lets you select the namespace.

        Examples:
            >>> df = pl.DataFrame({"id": ["abc-123"], "side": ["a"]})
            >>> df.select(plh.col("id").uuidhash.uuid5_concat(pl.col("side"))).item()
            'e89d330c-f123-519c-a7a1-e48e46f30ccf'

            `default` gives a null in `other` the same result as the value
            itself:

            >>> df = pl.DataFrame(
            ...     {"id": ["abc-123"], "side": pl.Series([None], dtype=pl.Utf8)}
            ... )
            >>> df.select(
            ...     plh.col("id").uuidhash.uuid5_concat(pl.col("side"), default="a")
            ... ).item()
            'e89d330c-f123-519c-a7a1-e48e46f30ccf'
        """
        if default is not None:
            return _plugin("uuid5_concat_default", [self._expr, other, pl.lit(default)])

        return _plugin("uuid5_concat", [self._expr, other])


class HExpr(pl.Expr):
    """A `pl.Expr` that declares the polars-hash namespaces.

    [`col`][polars_hash.col] and [`concat_str`][polars_hash.concat_str] return
    this class. Use it as the type annotation when you pass one of their
    expressions between functions.
    """

    @property
    def chash(self) -> CryptographicHashingNameSpace:
        return CryptographicHashingNameSpace(self)

    @property
    def nchash(self) -> NonCryptographicHashingNameSpace:
        return NonCryptographicHashingNameSpace(self)

    @property
    def bytes(self) -> BytesNameSpace:
        return BytesNameSpace(self)

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
"""`pl.col`, with the polars-hash namespaces declared for a type checker.

Both functions work at run time. `plh.col` is a typed wrapper that declares the
namespaces, so mypy and Pyright accept `.chash.sha2_256()` on its result. With
`pl.col` they report an error and need a `# type: ignore` comment.

Examples:
    >>> df = pl.DataFrame({"foo": ["hello_world"]})
    >>> df.select(plh.col("foo").chash.sha2_256()).item()
    '35072c1ae546350e0bfa7ab11d49dc6f129e72ccd57ec7eb671225bbd197c8f1'
"""

concat_str = cast(HashConcatStr, pl.concat_str)
"""`pl.concat_str`, with the polars-hash namespaces declared for a type checker.

Use it to hash more than one column. Give a `separator` value: without one,
`("ab", "c")` and `("a", "bc")` give the same hash. To hash a whole row of any
type, use [`hash_rows`][polars_hash.hash_rows] instead.

The default value of `ignore_nulls` in `pl.concat_str` is `False`, so one null
input makes the concatenation null and the hash is also null. To get a value
instead, set `ignore_nulls=True` or replace the null values first.

Examples:
    >>> df = pl.DataFrame({"foo": ["hello_world"], "bar": ["today"]})
    >>> df.select(plh.concat_str("foo", "bar", separator="|").chash.sha2_256()).item()
    'e65103da8dabb65a3ebd4204dbc01f0d2d5eb685a1fb039518302f9fc2fc0b73'
"""


def _row_fields(
    exprs: IntoExpr | Iterable[IntoExpr],
    more_exprs: tuple[IntoExpr, ...],
) -> list[IntoExpr]:
    """Gives the columns of a row one name each.

    A struct needs one name for each field, but a row does not. `hash_rows(col("a"),
    col("a"))` and two columns that make one name are both a row of two values. This
    function adds a suffix to each column after the first. The encoder does not read
    the names, and therefore the suffix does not change the bytes.

    The first column keeps its name, because the name of the output comes from it.
    """
    if isinstance(exprs, (str, pl.Expr, pl.Series)) or not isinstance(exprs, Iterable):
        columns: list[IntoExpr] = [exprs]
    else:
        columns = list(exprs)
    columns += more_exprs

    # `pl.col` and `pl.lit` accept the same values as the struct. A wildcard needs
    # `name.suffix`, because `alias` gives all its columns one name.
    named: list[IntoExpr] = columns[:1]
    for position, column in enumerate(columns[1:], start=1):
        if isinstance(column, str):
            column = pl.col(column)
        elif not isinstance(column, pl.Expr):
            column = pl.lit(column)
        named.append(column.name.suffix(f"__polars_hash_{position}"))
    return named


def hash_rows(
    exprs: IntoExpr | Iterable[IntoExpr],
    *more_exprs: IntoExpr,
    version: int = 1,
) -> HExpr:
    r"""Change each row into Binary, for use with any hasher in this package.

    A hash of joined columns is not sufficient. The rows `("ab", "c")` and
    `("a", "bc")` make the same string, one null makes the full row null, and a
    List, an Array or a Struct column has no string form. This function gives
    each row bytes that no other row can make, and a hasher then reads those
    bytes.

    The encoder reads the meaning of a value and not the polars storage of it, so
    an Int32 and the Int64 next to it make the same hash. It does not read the
    column names, so a new name for a column keeps its hash, but a new order of
    the columns does not. A null is one of the values that the encoding writes,
    so a row with a null also has a hash. The
    [encoding][row-encoding] gives all the rules.

    Args:
        exprs: The columns to encode, in the order of the row. This argument
            accepts all that `pl.struct` accepts, and also selectors.
        *more_exprs: More columns, as positional arguments.
        version: The encoding to write. Version 1 does not change. To use a later
            version, give its number.

    Returns:
        An expression that makes Binary. There is one value for each row, and no
            value is null.

    Note:
        The output column keeps the name of the first column, as `pl.struct`,
        `pl.concat_str` and each `*_horizontal` expression do, so `with_columns`
        replaces that column. Use `.alias()` to keep it.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["hello_world"],
        ...         "bar": [42],
        ...         "baz": [[1, 2, 3]],
        ...         "qux": [{"a": 1}],
        ...     }
        ... )
        >>> df.select(plh.hash_rows(pl.all()).chash.sha2_256()).item()
        '9055866af8d3c113e0a8fdb729ce8e6fa67ed5f6f51efa8235a588e88ea972f4'

        You can keep, compare or store the bytes:

        >>> df.select(plh.hash_rows(pl.all())).item()
        b'\r\x04\x05\x0bhello_world\x03\x01*\x0c\x03\x03\x01\x01\x03\x01\x02\x03\x01\x03\r\x01\x03\x01\x01'

        Column names also work, and they set the order of the row:

        >>> df.select(plh.hash_rows("foo", "bar").nchash.xxh3_64()).item()
        9123089596710669414
    """
    # A plugin cannot expand a wildcard, but `pl.concat_str` can. `pl.all()` makes a
    # copy of the call for each column. It does not send all the columns to one call.
    # The struct sends a full row as one input.
    #
    # The output keeps the name of the first column, as `pl.struct`, `pl.concat_str`
    # and each `*_horizontal` expression do. A constant name such as `row` would also
    # replace a column, and it would replace one that the caller did not expect.
    return cast(
        HExpr,
        _plugin(
            "encode_rows", pl.struct(_row_fields(exprs, more_exprs)), version=version
        ),
    )


__all__ = ["UUIDNamespace", "__version__", "col", "concat_str", "hash_rows"]
