use mur3::murmurhash3_x64_128;
use mur3::murmurhash3_x86_32;

pub fn murmurhash3_32(value: &str, seed: u32) -> u32 {
    murmurhash3_x86_32(value.as_bytes(), seed)
}

pub fn murmurhash3_128(value: &str, seed: u32) -> Vec<u8> {
    let mut result = Vec::new();
    let hash = murmurhash3_x64_128(value.as_bytes(), seed);

    result.extend_from_slice(hash.0.to_le_bytes().as_ref());
    result.extend_from_slice(hash.1.to_le_bytes().as_ref());

    result
}
