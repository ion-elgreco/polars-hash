mod expressions;
mod geohashers;
mod h3;
mod murmurhash_hashers;
mod sha_hashers;
mod xxhash_hashers;

use pyo3::types::PyModule;
use pyo3::{pymodule, Bound, PyResult, Python};
use pyo3_polars::PolarsAllocator;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pymodule]
fn _internal(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
