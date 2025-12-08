use uuid::Uuid;

pub fn uuid5_dns_hash(value: &str, output: &mut String) {
    output.push_str(&Uuid::new_v5(&Uuid::NAMESPACE_DNS, value.as_bytes()).hyphenated().to_string())
}

pub fn uuid5_url_hash(value: &str, output: &mut String) {
    output.push_str(&Uuid::new_v5(&Uuid::NAMESPACE_URL, value.as_bytes()).hyphenated().to_string())
}

pub fn uuid5_oid_hash(value: &str, output: &mut String) {
    output.push_str(&Uuid::new_v5(&Uuid::NAMESPACE_OID, value.as_bytes()).hyphenated().to_string())
}

pub fn uuid5_x500_hash(value: &str, output: &mut String) {
    output.push_str(&Uuid::new_v5(&Uuid::NAMESPACE_X500, value.as_bytes()).hyphenated().to_string())
}
