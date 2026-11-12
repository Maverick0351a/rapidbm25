//! rapidbm25 native core.
//!
//! The static index is a bit-for-bit rank_bm25 scorer. New pieces (live index,
//! tokenizer, batch scoring, on-disk format) sit beside it and call the same
//! scoring kernels. No fused multiply-add is used.

mod index;
mod live;
mod persist;
mod score;
mod tokenize;

use pyo3::prelude::*;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<index::Index>()?;
    m.add_class::<live::LiveIndex>()?;
    m.add_class::<tokenize::Tokenizer>()?;
    m.add_function(wrap_pyfunction!(score::top_k_of, m)?)?;
    Ok(())
}
