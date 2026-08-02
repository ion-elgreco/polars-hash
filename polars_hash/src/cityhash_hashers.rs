//! `cityhasher` is built without default features: they swap bounds checks for `unsafe`.
use cityhash_rs::cityhash_110_128;

pub fn cityhash_32(value: Option<&str>) -> Option<u32> {
    value.map(|v| cityhasher::hash(v.as_bytes()))
}

pub fn cityhash_64(value: Option<&str>) -> Option<u64> {
    value.map(|v| cityhasher::hash(v.as_bytes()))
}

pub fn cityhash_64_with_seed(value: Option<&str>, seed: u64) -> Option<u64> {
    value.map(|v| cityhasher::hash_with_seed(v.as_bytes(), seed))
}

pub fn cityhash_128(value: Option<&str>) -> Option<u128> {
    value.map(|v| cityhash_110_128(v.as_bytes()))
}
