use xxhash_rust::xxh32::xxh32;
use xxhash_rust::xxh64::xxh64;

pub fn xxhash_32(value: Option<&str>, seed: u32) -> Option<u32> {
    value.map(|v| xxh32(v.as_bytes(), seed))
}

pub fn xxhash_64(value: Option<&str>, seed: u64) -> Option<u64> {
    value.map(|v| xxh64(v.as_bytes(), seed))
}
