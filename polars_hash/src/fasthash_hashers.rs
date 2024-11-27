use fasthash::{murmur3, xx};

pub fn murmurhash3_32(value: Option<&str>) -> Option<u32> {
    value.map(|v| murmur3::hash32(v.as_bytes()))
}

pub fn murmurhash3_128(value: Option<&str>) -> Option<Vec<u8>> {
    value.map(|v| murmur3::hash128(v.as_bytes()).to_le_bytes().to_vec())
}

pub fn xxhash_32(value: Option<&str>) -> Option<u32> {
    value.map(|v| xx::hash32(v.as_bytes()))
}

pub fn xxhash_64(value: Option<&str>) -> Option<u64> {
    value.map(|v| xx::hash64(v.as_bytes()))
}
