use xxhash_rust::xxh3::xxh3_128_with_seed;
use xxhash_rust::xxh3::xxh3_64_with_seed;
use xxhash_rust::xxh32::xxh32;
use xxhash_rust::xxh64::xxh64;

pub fn xxhash_32(value: &str, seed: u32) -> u32 {
    xxh32(value.as_bytes(), seed)
}

pub fn xxhash_64(value: &str, seed: u64) -> u64 {
    xxh64(value.as_bytes(), seed)
}

pub fn xxhash3_64(value: &str, seed: u64) -> u64 {
    xxh3_64_with_seed(value.as_bytes(), seed)
}

pub fn xxhash3_128(value: &str, seed: u64) -> u128 {
    xxh3_128_with_seed(value.as_bytes(), seed)
}
