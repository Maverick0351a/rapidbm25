//! Immutable, rank_bm25-compatible index.
//!
//! `fixes = false` (the default) keeps the original control flow and the
//! original IEEE operation order. `fixes = true` opts into the documented
//! rank_bm25 bugfixes and is never used by the contract replay.

use crate::persist::{self, Columns};
use crate::score::{
    absent_uniform, compute_norm, okapi_idf, score_into, top_k, variant_idf, QueryDedup, Variant,
};
use memmap2::Mmap;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::path::Path;

pub(crate) struct Owned {
    pub(crate) doc_len: Vec<u64>,
    pub(crate) terms: Vec<String>,
    pub(crate) lookup: FxHashMap<String, u32>,
    pub(crate) idf: Vec<f64>,
    pub(crate) offsets: Vec<usize>,
    pub(crate) post_doc: Vec<u32>,
    pub(crate) post_tf: Vec<f64>,
    pub(crate) norm: Vec<f64>,
    pub(crate) doc_terms_off: Vec<usize>,
    pub(crate) doc_terms: Vec<(u32, u32)>,
}

pub(crate) struct Mapped {
    pub(crate) mmap: Mmap,
    pub(crate) terms: Vec<String>,
    pub(crate) lookup: FxHashMap<String, u32>,
    pub(crate) doc_len_off: usize,
    pub(crate) idf_off: usize,
    pub(crate) offsets_off: usize,
    pub(crate) post_doc_off: usize,
    pub(crate) post_tf_off: usize,
    pub(crate) norm_off: usize,
    pub(crate) doc_terms_off_off: usize,
    pub(crate) doc_terms_tid_off: usize,
    pub(crate) doc_terms_tf_off: usize,
    pub(crate) n_terms: usize,
    pub(crate) n_posts: usize,
    pub(crate) n_doc_terms: usize,
}

impl Mapped {
    /// # Safety invariants
    /// The file is little-endian and every numeric section starts on an
    /// 8-byte boundary. The mapping's base is page-aligned, so an 8-aligned
    /// file offset is a valid `u64`/`f64` pointer. This is only constructed
    /// on 64-bit little-endian hosts, where `usize` and `u64` share a
    /// representation, so the offset table can be read as `&[usize]`.
    fn bytes(&self, off: usize, nbytes: usize) -> &[u8] {
        &self.mmap[off..off + nbytes]
    }
    fn u64s(&self, off: usize, n: usize) -> &[u64] {
        if n == 0 {
            return &[];
        }
        let bytes = self.bytes(off, n * 8);
        debug_assert_eq!(bytes.as_ptr() as usize % 8, 0);
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, n) }
    }
    fn u32s(&self, off: usize, n: usize) -> &[u32] {
        if n == 0 {
            return &[];
        }
        let bytes = self.bytes(off, n * 4);
        debug_assert_eq!(bytes.as_ptr() as usize % 4, 0);
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, n) }
    }
    fn f64s(&self, off: usize, n: usize) -> &[f64] {
        let u = self.u64s(off, n);
        if u.is_empty() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(u.as_ptr() as *const f64, u.len()) }
    }
    fn usizes(&self, off: usize, n: usize) -> &[usize] {
        let u = self.u64s(off, n);
        if u.is_empty() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(u.as_ptr() as *const usize, u.len()) }
    }
}

pub(crate) enum Data {
    Owned(Owned),
    Mapped(Mapped),
}

#[pyclass(frozen, module = "rapidbm25._native")]
pub struct Index {
    pub(crate) variant: Variant,
    pub(crate) k1: f64,
    pub(crate) b: f64,
    pub(crate) epsilon: f64,
    pub(crate) delta: f64,
    pub(crate) fixes: bool,
    pub(crate) n_docs: usize,
    pub(crate) avgdl: f64,
    pub(crate) average_idf: f64,
    pub(crate) absent_partial: Option<f64>,
    pub(crate) mapped: bool,
    pub(crate) data: Data,
}

impl Index {
    fn terms(&self) -> &[String] {
        match &self.data {
            Data::Owned(o) => &o.terms,
            Data::Mapped(m) => &m.terms,
        }
    }
    fn lookup(&self) -> &FxHashMap<String, u32> {
        match &self.data {
            Data::Owned(o) => &o.lookup,
            Data::Mapped(m) => &m.lookup,
        }
    }
    fn idf(&self) -> &[f64] {
        match &self.data {
            Data::Owned(o) => &o.idf,
            Data::Mapped(m) => m.f64s(m.idf_off, m.n_terms),
        }
    }
    fn doc_len(&self) -> &[u64] {
        match &self.data {
            Data::Owned(o) => &o.doc_len,
            Data::Mapped(m) => m.u64s(m.doc_len_off, self.n_docs),
        }
    }
    fn offsets(&self) -> &[usize] {
        match &self.data {
            Data::Owned(o) => &o.offsets,
            Data::Mapped(m) => m.usizes(m.offsets_off, m.n_terms + 1),
        }
    }
    fn post_doc(&self) -> &[u32] {
        match &self.data {
            Data::Owned(o) => &o.post_doc,
            Data::Mapped(m) => m.u32s(m.post_doc_off, m.n_posts),
        }
    }
    fn post_tf(&self) -> &[f64] {
        match &self.data {
            Data::Owned(o) => &o.post_tf,
            Data::Mapped(m) => m.f64s(m.post_tf_off, m.n_posts),
        }
    }
    fn norm(&self) -> &[f64] {
        match &self.data {
            Data::Owned(o) => &o.norm,
            Data::Mapped(m) => m.f64s(m.norm_off, self.n_docs),
        }
    }
    fn doc_terms_off(&self) -> &[usize] {
        match &self.data {
            Data::Owned(o) => &o.doc_terms_off,
            Data::Mapped(m) => m.usizes(m.doc_terms_off_off, self.n_docs + 1),
        }
    }
    fn doc_term(&self, i: usize) -> (u32, u32) {
        match &self.data {
            Data::Owned(o) => o.doc_terms[i],
            Data::Mapped(m) => {
                let tid = m.u32s(m.doc_terms_tid_off, m.n_doc_terms)[i];
                let tf = m.u32s(m.doc_terms_tf_off, m.n_doc_terms)[i];
                (tid, tf)
            }
        }
    }
    fn n_doc_terms(&self) -> usize {
        match &self.data {
            Data::Owned(o) => o.doc_terms.len(),
            Data::Mapped(m) => m.n_doc_terms,
        }
    }

    pub(crate) fn columns(&self) -> Columns {
        let mut doc_terms_tid = Vec::with_capacity(self.n_doc_terms());
        let mut doc_terms_tf = Vec::with_capacity(self.n_doc_terms());
        for i in 0..self.n_doc_terms() {
            let (t, c) = self.doc_term(i);
            doc_terms_tid.push(t);
            doc_terms_tf.push(c);
        }
        Columns {
            doc_len: self.doc_len().to_vec(),
            terms: self.terms().to_vec(),
            idf: self.idf().to_vec(),
            offsets: self.offsets().iter().map(|&x| x as u64).collect(),
            post_doc: self.post_doc().to_vec(),
            post_tf: self.post_tf().to_vec(),
            norm: self.norm().to_vec(),
            doc_terms_off: self.doc_terms_off().iter().map(|&x| x as u64).collect(),
            doc_terms_tid,
            doc_terms_tf,
        }
    }

    fn idf_of(&self, q: &str) -> (Option<u32>, f64) {
        match self.lookup().get(q) {
            Some(&t) => (Some(t), self.idf()[t as usize]),
            None => (None, 0.0),
        }
    }

    fn score_one(&self, query: &[(Option<u32>, f64)]) -> Vec<f64> {
        let mut s = vec![0.0f64; self.n_docs];
        score_into(
            self.variant,
            self.k1,
            self.delta,
            self.n_docs,
            self.norm(),
            self.offsets(),
            self.post_doc(),
            self.post_tf(),
            self.absent_partial,
            query,
            &mut s,
        );
        s
    }

    fn resolve_query(&self, query: &Bound<'_, PyAny>) -> PyResult<Vec<(Option<u32>, f64)>> {
        let mut out = Vec::new();
        let mut dedup = QueryDedup::default();
        for item in query.try_iter()? {
            let item = item?;
            let (tid_idf, token) = match item.cast::<PyString>() {
                Ok(s) => {
                    let s = s.to_str()?;
                    (self.idf_of(s), Some(s.to_owned()))
                }
                // non-string tokens can never be in a str vocabulary
                Err(_) => ((None, 0.0), None),
            };
            if dedup.allow(self.fixes, tid_idf.0, token.as_deref()) {
                out.push(tid_idf);
            }
        }
        Ok(out)
    }

    fn empty(variant: Variant, k1: f64, b: f64, epsilon: f64, delta: f64) -> Self {
        Self {
            variant,
            k1,
            b,
            epsilon,
            delta,
            fixes: true,
            n_docs: 0,
            avgdl: 0.0,
            average_idf: if variant == Variant::Okapi { 0.0 } else { f64::NAN },
            absent_partial: None,
            mapped: false,
            data: Data::Owned(Owned {
                doc_len: Vec::new(),
                terms: Vec::new(),
                lookup: FxHashMap::default(),
                idf: Vec::new(),
                offsets: vec![0],
                post_doc: Vec::new(),
                post_tf: Vec::new(),
                norm: Vec::new(),
                doc_terms_off: vec![0],
                doc_terms: Vec::new(),
            }),
        }
    }
}

#[pymethods]
impl Index {
    #[new]
    #[pyo3(signature = (corpus, variant, k1, b, epsilon=0.25, delta=0.5, fixes=false))]
    fn new(
        corpus: &Bound<'_, PyAny>,
        variant: &str,
        k1: f64,
        b: f64,
        epsilon: f64,
        delta: f64,
        fixes: bool,
    ) -> PyResult<Self> {
        let variant = Variant::parse(variant)?;
        let mut lookup: FxHashMap<String, u32> = FxHashMap::default();
        let mut terms: Vec<String> = Vec::new();
        let mut nd: Vec<u64> = Vec::new();
        let mut doc_len: Vec<u64> = Vec::new();
        let mut doc_terms_off: Vec<usize> = vec![0];
        let mut doc_terms: Vec<(u32, u32)> = Vec::new();
        // scratch: term id -> position in current doc's term list (+1), 0 = absent
        let mut slot: Vec<u32> = Vec::new();
        let mut total: u64 = 0;

        for doc in corpus.try_iter()? {
            let doc = doc?;
            let start = doc_terms.len();
            let mut len: u64 = 0;
            for tok in doc.try_iter()? {
                let tok = tok?;
                let s = tok.cast::<PyString>().map_err(|_| {
                    PyTypeError::new_err("rapidbm25 tokens must be str")
                })?;
                let s = s.to_str()?;
                let tid = match lookup.get(s) {
                    Some(&t) => t,
                    None => {
                        let t = terms.len() as u32;
                        lookup.insert(s.to_owned(), t);
                        terms.push(s.to_owned());
                        nd.push(0);
                        slot.push(0);
                        t
                    }
                };
                let sl = slot[tid as usize];
                if sl == 0 {
                    doc_terms.push((tid, 1));
                    slot[tid as usize] = (doc_terms.len() - start) as u32;
                } else {
                    doc_terms[start + sl as usize - 1].1 += 1;
                }
                len += 1;
            }
            for &(t, _) in &doc_terms[start..] {
                slot[t as usize] = 0;
                nd[t as usize] += 1;
            }
            doc_len.push(len);
            total += len;
            doc_terms_off.push(doc_terms.len());
        }

        let n_docs = doc_len.len();
        if n_docs == 0 {
            if fixes {
                // rank_bm25 #36: an empty corpus is a valid (empty) index.
                return Ok(Self::empty(variant, k1, b, epsilon, delta));
            }
            return Err(PyZeroDivisionError::new_err("division by zero (empty corpus)"));
        }
        let avgdl = total as f64 / n_docs as f64;
        let n = n_docs as f64;

        let (idf, average_idf) = match variant {
            Variant::Okapi => okapi_idf(&nd, n, epsilon, fixes)?,
            other => (variant_idf(other, &nd, n), f64::NAN),
        };

        // CSR postings (doc ids ascending because we walk docs in order)
        let mut offsets = vec![0usize; terms.len() + 1];
        for &(t, _) in &doc_terms {
            offsets[t as usize + 1] += 1;
        }
        for i in 0..terms.len() {
            offsets[i + 1] += offsets[i];
        }
        let mut cursor = offsets.clone();
        let mut post_doc = vec![0u32; doc_terms.len()];
        let mut post_tf = vec![0f64; doc_terms.len()];
        for d in 0..n_docs {
            for &(t, c) in &doc_terms[doc_terms_off[d]..doc_terms_off[d + 1]] {
                let p = cursor[t as usize];
                post_doc[p] = d as u32;
                post_tf[p] = c as f64;
                cursor[t as usize] = p + 1;
            }
        }

        let norm = compute_norm(variant, &doc_len, avgdl, b, k1);
        let absent_partial = absent_uniform(variant, &norm, k1);

        Ok(Self {
            variant,
            k1,
            b,
            epsilon,
            delta,
            fixes,
            n_docs,
            avgdl,
            average_idf,
            absent_partial,
            mapped: false,
            data: Data::Owned(Owned {
                doc_len,
                terms,
                lookup,
                idf,
                offsets,
                post_doc,
                post_tf,
                norm,
                doc_terms_off,
                doc_terms,
            }),
        })
    }

    #[getter]
    fn corpus_size(&self) -> usize {
        self.n_docs
    }
    #[getter]
    fn avgdl(&self) -> f64 {
        self.avgdl
    }
    #[getter]
    fn average_idf(&self) -> f64 {
        self.average_idf
    }
    #[getter]
    fn vocab_size(&self) -> usize {
        self.terms().len()
    }
    #[getter]
    fn dense_fallback(&self) -> bool {
        self.absent_partial.is_none()
    }
    #[getter]
    fn k1(&self) -> f64 {
        self.k1
    }
    #[getter]
    fn b(&self) -> f64 {
        self.b
    }
    #[getter]
    fn epsilon(&self) -> f64 {
        self.epsilon
    }
    #[getter]
    fn delta(&self) -> f64 {
        self.delta
    }
    #[getter]
    fn fixes(&self) -> bool {
        self.fixes
    }
    #[getter]
    fn compat(&self) -> &'static str {
        self.variant.compat()
    }
    #[getter]
    fn class_name(&self) -> &'static str {
        self.variant.class_name()
    }
    #[getter]
    fn variant(&self) -> &'static str {
        self.variant.as_str()
    }
    #[getter]
    fn mapped(&self) -> bool {
        self.mapped
    }

    fn doc_len_list(&self) -> Vec<u64> {
        self.doc_len().to_vec()
    }

    fn idf_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let idf = self.idf();
        for (t, v) in self.terms().iter().zip(idf) {
            d.set_item(t, *v)?;
        }
        Ok(d)
    }

    fn doc_freqs_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        let off = self.doc_terms_off();
        let terms = self.terms();
        for d in 0..self.n_docs {
            let m = PyDict::new(py);
            for i in off[d]..off[d + 1] {
                let (t, c) = self.doc_term(i);
                m.set_item(&terms[t as usize], c)?;
            }
            out.append(m)?;
        }
        Ok(out)
    }

    /// Full score vector (float64, length corpus_size).
    fn get_scores<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let q = self.resolve_query(query)?;
        let score = py.detach(|| self.score_one(&q));
        Ok(PyArray1::from_vec(py, score))
    }

    /// Score many queries at once. The GIL is released; queries run on a thread pool.
    /// Shape is `(n_queries, corpus_size)`. Row `i` equals `get_scores(queries[i])`.
    fn get_scores_many<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut qs = Vec::new();
        for q in queries.try_iter()? {
            qs.push(self.resolve_query(&q?)?);
        }
        if qs.is_empty() {
            return Ok(PyArray2::zeros(py, [0, self.n_docs], false));
        }
        let rows: Vec<Vec<f64>> = py.detach(|| qs.par_iter().map(|q| self.score_one(q)).collect());
        PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Top-k doc ids and their scores: score descending, ties by lower doc id.
    fn top_k<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
        let q = self.resolve_query(query)?;
        let (ids, vals) = py.detach(|| {
            let s = self.score_one(&q);
            let ids = top_k(&s, k);
            let vals: Vec<f64> = ids.iter().map(|&i| s[i as usize]).collect();
            (
                ids.into_iter().map(|i| i as i64).collect::<Vec<i64>>(),
                vals,
            )
        });
        Ok((ids.to_pyarray(py), vals.to_pyarray(py)))
    }

    /// Top-k for many queries. GIL released; one thread per query.
    fn top_n_many<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let mut qs = Vec::new();
        for q in queries.try_iter()? {
            qs.push(self.resolve_query(&q?)?);
        }
        let rows: Vec<(Vec<i64>, Vec<f64>)> = py.detach(|| {
            qs.par_iter()
                .map(|q| {
                    let s = self.score_one(q);
                    let ids = top_k(&s, k);
                    let vals: Vec<f64> = ids.iter().map(|&i| s[i as usize]).collect();
                    (ids.into_iter().map(|i| i as i64).collect(), vals)
                })
                .collect()
        });
        let out = PyList::empty(py);
        for (ids, vals) in rows {
            out.append((ids.to_pyarray(py), vals.to_pyarray(py)))?;
        }
        Ok(out)
    }

    /// Write the versioned on-disk format. Numeric columns are raw little-endian.
    fn save(&self, path: &str) -> PyResult<()> {
        let bytes = persist::dump_static(self)?;
        persist::write_atomic(Path::new(path), &bytes)
    }

    /// Load a static index. `mmap=false` (the default) copies the columns.
    /// `mmap=true` maps the file on 64-bit little-endian hosts and is unsafe
    /// if that file is truncated or written in place. The Python classes pass
    /// the flag through; their default is also a copy.
    #[staticmethod]
    #[pyo3(signature = (path, mmap=false))]
    fn load(path: &str, mmap: bool) -> PyResult<Self> {
        if mmap {
            persist::load_static_mmap(Path::new(path))
        } else {
            let bytes = std::fs::read(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
            persist::load_static_bytes(&bytes)
        }
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = persist::dump_static(self)?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    fn from_bytes(data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        persist::load_static_bytes(data.as_bytes())
    }
}
