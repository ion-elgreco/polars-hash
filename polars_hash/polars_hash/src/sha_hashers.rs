use sha1::{Sha1};
use sha2::{Digest, Sha224, Sha256, Sha384, Sha512};
use sha3::{Sha3_224, Sha3_256, Sha3_384, Sha3_512};
use std::fmt::Write;

pub fn sha1_hash(value: &str, output: &mut String) {
    let hash = Sha1::digest(value);
    write!(output, "{:x}", hash).unwrap()
}

pub fn sha2_256_hash(value: &str, output: &mut String) {
    let hash = Sha256::digest(value);
    write!(output, "{:x}", hash).unwrap()
}

pub fn sha2_512_hash(value: &str, output: &mut String) {
    let hash = Sha512::digest(value);
    write!(output, "{:x}", hash).unwrap()
}

pub fn sha2_384_hash(value: &str, output: &mut String) {
    let hash = Sha384::digest(value);
    write!(output, "{:x}", hash).unwrap()
}

pub fn sha2_224_hash(value: &str, output: &mut String) {
    let hash = Sha224::digest(value);
    write!(output, "{:x}", hash).unwrap()
}

pub fn sha3_256_hash(value: &str, output: &mut String) {
    let hash = Sha3_256::digest(value);
    write!(output, "{:x}", hash).unwrap()
}

pub fn sha3_512_hash(value: &str, output: &mut String) {
    let hash = Sha3_512::digest(value);
    write!(output, "{:x}", hash).unwrap()
}

pub fn sha3_384_hash(value: &str, output: &mut String) {
    let hash = Sha3_384::digest(value);
    write!(output, "{:x}", hash).unwrap()
}

pub fn sha3_224_hash(value: &str, output: &mut String) {
    let hash = Sha3_224::digest(value);
    write!(output, "{:x}", hash).unwrap()
}
