use mur3::murmurhash3_x64_128;
use mur3::murmurhash3_x86_32;
use xxhash_rust::xxh32::xxh32;
use xxhash_rust::xxh64::xxh64;

pub fn murmurhash3_32(value: Option<&str>, seed: u32) -> Option<u32> {
    value.map(|v| murmurhash3_x86_32(v.as_bytes(), seed))
}

pub fn murmurhash3_128(value: Option<&str>, seed: u32) -> Option<Vec<u8>> {
    value.map(|v| {
        let mut result = Vec::new();
        let hash = murmurhash3_x64_128(v.as_bytes(), seed);

        result.extend_from_slice(hash.0.to_le_bytes().as_ref());
        result.extend_from_slice(hash.1.to_le_bytes().as_ref());

        result
    })
}

pub fn xxhash_32(value: Option<&str>, seed: u32) -> Option<u32> {
    value.map(|v| xxh32(v.as_bytes(), seed))
}

pub fn xxhash_64(value: Option<&str>, seed: u64) -> Option<u64> {
    value.map(|v| xxh64(v.as_bytes(), seed))
}
