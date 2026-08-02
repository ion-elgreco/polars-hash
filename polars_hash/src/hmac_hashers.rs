use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::fmt::Write;

pub type HmacSha256 = Hmac<Sha256>;

pub fn hmac_sha256_hash(value: &[u8], output: &mut String, keyed_mac: &HmacSha256) {
    let mut mac = keyed_mac.clone();
    mac.update(value);
    write!(output, "{}", hex::encode(mac.finalize().into_bytes())).unwrap()
}
