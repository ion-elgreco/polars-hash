use mur3::murmurhash3_x64_128;
use mur3::murmurhash3_x86_32;

pub fn murmurhash3_32(value: &[u8], seed: u32) -> u32 {
    murmurhash3_x86_32(value, seed)
}

/// The two halves are packed the way `mmh3.hash128` packs them, which is the digest
/// MurmurHash3 writes -- `h1` then `h2`, each little-endian -- read as one integer.
pub fn murmurhash3_128(value: &[u8], seed: u32) -> u128 {
    let (h1, h2) = murmurhash3_x64_128(value, seed);

    (h1 as u128) | ((h2 as u128) << 64)
}
