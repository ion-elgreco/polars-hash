use fasthash::{murmur3, xx};

pub fn murmurhash3_32(value: Option<&str>, seed: u32) -> Option<u32> {
    value.map(|v| murmur3::hash32_with_seed(v.as_bytes(), seed))
}

pub fn murmurhash3_128(value: Option<&str>, seed: u32) -> Option<Vec<u8>> {
    value.map(|v| {
        murmur3::hash128_with_seed(v.as_bytes(), seed)
            .to_le_bytes()
            .to_vec()
    })
}

pub fn xxhash_32(value: Option<&str>, seed: u32) -> Option<u32> {
    value.map(|v| xx::hash32_with_seed(v.as_bytes(), seed))
}

pub fn xxhash_64(value: Option<&str>, seed: u64) -> Option<u64> {
    value.map(|v| xx::hash64_with_seed(v.as_bytes(), seed))
}
