use polars::prelude::*;

/// Encode a column into its native-width bytes, little-endian if `big_endian` is
/// `false` and big-endian otherwise.
///
/// A `String` or `Binary` value has no endianness of its own, it passes through
/// unchanged either way. Every other supported type keeps its own width: `Int8`
/// becomes 1 byte, `Int32` becomes 4, `Float64` becomes 8, and so on. A caller who
/// wants a different width casts before calling this, the same way any
/// other numeric cast in Polars works.
pub fn encode_bytes(s: &Series, big_endian: bool) -> PolarsResult<BinaryChunked> {
    match s.dtype() {
        // A single byte has no order of its own, so Boolean and the two 8-bit
        // integer types ignore `big_endian`.
        DataType::Boolean => {
            let ca = s.cast(&DataType::UInt8)?;
            Ok(binary_from_chunked(ca.u8()?, |v: u8| [v]))
        }
        DataType::Int8 => Ok(binary_from_chunked(s.i8()?, |v: i8| [v as u8])),
        DataType::UInt8 => Ok(binary_from_chunked(s.u8()?, |v: u8| [v])),
        DataType::Int16 => Ok(endian_encode(s.i16()?, big_endian)),
        DataType::UInt16 => Ok(endian_encode(s.u16()?, big_endian)),
        DataType::Int32 => Ok(endian_encode(s.i32()?, big_endian)),
        DataType::UInt32 => Ok(endian_encode(s.u32()?, big_endian)),
        DataType::Int64 => Ok(endian_encode(s.i64()?, big_endian)),
        DataType::UInt64 => Ok(endian_encode(s.u64()?, big_endian)),
        DataType::Float32 => Ok(endian_encode(s.f32()?, big_endian)),
        DataType::Float64 => Ok(endian_encode(s.f64()?, big_endian)),
        DataType::String => {
            let ca = s.str()?;
            let mut builder = BinaryChunkedBuilder::new(s.name().clone(), s.len());
            for value in ca.iter() {
                match value {
                    Some(value) => builder.append_value(value.as_bytes()),
                    None => builder.append_null(),
                }
            }
            Ok(builder.finish())
        }
        DataType::Binary => Ok(s.binary()?.clone()),
        dtype => polars_bail!(
            InvalidOperation: "expected a numeric, Boolean, String or Binary input, got `{}`", dtype
        ),
    }
}

fn binary_from_chunked<T, F, const N: usize>(ca: &ChunkedArray<T>, op: F) -> BinaryChunked
where
    T: PolarsNumericType,
    F: Fn(T::Native) -> [u8; N],
{
    let mut builder = BinaryChunkedBuilder::new(ca.name().clone(), ca.len());
    for value in ca.iter() {
        match value {
            Some(value) => builder.append_value(op(value)),
            None => builder.append_null(),
        }
    }
    builder.finish()
}

/// Bridges [`binary_from_chunked`] to the native `to_le_bytes` and
/// `to_be_bytes` methods every integer and float primitive have.
///  They don't have any standard trait, so [`impl_endian_bytes`] writes the needed one.
trait EndianBytes<const N: usize> {
    fn to_le(self) -> [u8; N];
    fn to_be(self) -> [u8; N];
}

macro_rules! impl_endian_bytes {
    ($($t:ty => $n:literal),* $(,)?) => {
        $(
            impl EndianBytes<$n> for $t {
                fn to_le(self) -> [u8; $n] {
                    self.to_le_bytes()
                }
                fn to_be(self) -> [u8; $n] {
                    self.to_be_bytes()
                }
            }
        )*
    };
}

impl_endian_bytes!(i16 => 2, u16 => 2, i32 => 4, u32 => 4, i64 => 8, u64 => 8, f32 => 4, f64 => 8);

fn endian_encode<T, const N: usize>(ca: &ChunkedArray<T>, big_endian: bool) -> BinaryChunked
where
    T: PolarsNumericType,
    T::Native: EndianBytes<N>,
{
    if big_endian {
        binary_from_chunked(ca, |v: T::Native| v.to_be())
    } else {
        binary_from_chunked(ca, |v: T::Native| v.to_le())
    }
}
