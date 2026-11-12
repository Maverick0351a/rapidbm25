//! rapidbm25 native core.
//!
//! Written from the behavior spec in `SPEC.md`. The goal is bit-for-bit equal
//! scores to the reference Python formulas, so every expression below keeps
//! the same IEEE-754 operation order as the spec. No fused multiply-add is used.

use numpy::{PyArray1, ToPyArray};
use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    Okapi,
    /// BM25L as in rank_bm25 git 47aa3dd
    L,
    /// BM25L as released in rank_bm25 0.2.2 on PyPI (extra tf factor)
    L022,
    /// BM25+ as in git 47aa3dd: idf = ln(N + 1) - ln(df)
    Plus,
    /// BM25+ as released in 0.2.2: idf = ln((N + 1) / df)
    Plus022,
}

impl Variant {
    fn parse(s: &str) -> PyResult<Self> {
        match s {
            "okapi" => Ok(Variant::Okapi),
            "l" => Ok(Variant::L),
            "l022" => Ok(Variant::L022),
            "plus" => Ok(Variant::Plus),
            "plus022" => Ok(Variant::Plus022),
            _ => Err(PyValueError::new_err(
                "variant must be 'okapi', 'l', 'l022', 'plus' or 'plus022'",
            )),
        }
    }
}

#[pyclass(module = "rapidbm25._native")]
pub struct Index {
    variant: Variant,
    k1: f64,
    delta: f64,
    n_docs: usize,
    avgdl: f64,
    average_idf: f64,
    doc_len: Vec<u64>,
    // vocabulary in first-seen order
    terms: Vec<String>,
    lookup: FxHashMap<String, u32>,
    idf: Vec<f64>,
    // CSR postings, doc ids ascending inside each term
    offsets: Vec<usize>,
    post_doc: Vec<u32>,
    post_tf: Vec<f64>,
    // per-document length factor:
    //   okapi / plus: k1 * ((1 - b) + b * dl / avgdl)
    //   l:            (1 - b) + b * dl / avgdl
    norm: Vec<f64>,
    // per-document "term absent" partial (see SPEC.md §4); `None` when it is
    // not the same bit pattern for every document, in which case scoring uses
    // the dense path for every term.
    absent_partial: Option<f64>,
    // per-doc term order (first occurrence) for rebuilding doc_freqs lazily
    doc_terms_off: Vec<usize>,
    doc_terms: Vec<(u32, u32)>,
}

#[inline(always)]
fn contrib(v: Variant, idf: f64, tf: f64, norm: f64, k1: f64, delta: f64) -> f64 {
    let k1p1 = k1 + 1.0;
    match v {
        // idf * (tf * (k1 + 1) / (tf + norm))
        Variant::Okapi => idf * (tf * k1p1 / (tf + norm)),
        // ((idf * (k1 + 1)) * (ctd + delta)) / ((k1 + ctd) + delta), ctd = tf / norm
        Variant::L => {
            let ctd = tf / norm;
            (idf * k1p1) * (ctd + delta) / (k1 + ctd + delta)
        }
        // (((idf * tf) * (k1 + 1)) * (ctd + delta)) / ((k1 + ctd) + delta)
        Variant::L022 => {
            let ctd = tf / norm;
            idf * tf * k1p1 * (ctd + delta) / (k1 + ctd + delta)
        }
        // idf * (delta + (tf * (k1 + 1)) / (norm + tf))
        Variant::Plus | Variant::Plus022 => idf * (delta + (tf * k1p1) / (norm + tf)),
    }
}

#[inline(always)]
fn absent_part(v: Variant, norm: f64, k1: f64) -> f64 {
    match v {
        Variant::Okapi => 0.0 * (k1 + 1.0) / (0.0 + norm),
        Variant::L | Variant::L022 => 0.0 / norm,
        Variant::Plus | Variant::Plus022 => (0.0 * (k1 + 1.0)) / (norm + 0.0),
    }
}

impl Index {
    fn idf_of(&self, q: &str) -> (Option<u32>, f64) {
        match self.lookup.get(q) {
            Some(&t) => (Some(t), self.idf[t as usize]),
            None => (None, 0.0),
        }
    }

    fn score_into(&self, query: &[(Option<u32>, f64)], score: &mut [f64]) {
        let (v, k1, delta) = (self.variant, self.k1, self.delta);
        for &(tid, idf) in query {
            let (s, e) = match tid {
                Some(t) => (self.offsets[t as usize], self.offsets[t as usize + 1]),
                None => (0, 0),
            };
            let docs = &self.post_doc[s..e];
            let tfs = &self.post_tf[s..e];
            match self.absent_partial {
                Some(_) => {
                    // value every document without this term receives
                    let c = contrib(v, idf, 0.0, self.norm.first().copied().unwrap_or(1.0), k1, delta);
                    if c == 0.0 {
                        // adding +/-0.0 never changes a score (scores start at +0.0)
                        for (d, tf) in docs.iter().zip(tfs) {
                            let d = *d as usize;
                            score[d] += contrib(v, idf, *tf, self.norm[d], k1, delta);
                        }
                    } else {
                        let mut p = 0usize;
                        for d in 0..self.n_docs {
                            if p < docs.len() && docs[p] as usize == d {
                                score[d] += contrib(v, idf, tfs[p], self.norm[d], k1, delta);
                                p += 1;
                            } else {
                                score[d] += c;
                            }
                        }
                    }
                }
                None => {
                    let mut p = 0usize;
                    for d in 0..self.n_docs {
                        let tf = if p < docs.len() && docs[p] as usize == d {
                            p += 1;
                            tfs[p - 1]
                        } else {
                            0.0
                        };
                        score[d] += contrib(v, idf, tf, self.norm[d], k1, delta);
                    }
                }
            }
        }
    }

    fn resolve_query(&self, query: &Bound<'_, PyAny>) -> PyResult<Vec<(Option<u32>, f64)>> {
        let mut out = Vec::new();
        for item in query.try_iter()? {
            let item = item?;
            let r = match item.cast::<PyString>() {
                Ok(s) => self.idf_of(s.to_str()?),
                // non-string tokens can never be in a str vocabulary
                Err(_) => (None, 0.0),
            };
            out.push(r);
        }
        Ok(out)
    }
}

/// Descending score, then ascending doc id. NaN sorts last.
#[inline(always)]
fn rank_cmp(scores: &[f64], a: u32, b: u32) -> Ordering {
    let (x, y) = (scores[a as usize], scores[b as usize]);
    match (x.is_nan(), y.is_nan()) {
        (true, true) => a.cmp(&b),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => y.partial_cmp(&x).unwrap().then(a.cmp(&b)),
    }
}

fn top_k(scores: &[f64], k: usize) -> Vec<u32> {
    let n = scores.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    // Fast path: if at least k docs score > 0, the top k are all among them
    // (everything else is <= 0 or NaN and ranks lower). Exact, any variant.
    let mut idx: Vec<u32> = scores
        .iter()
        .enumerate()
        .filter(|(_, &x)| x > 0.0)
        .map(|(i, _)| i as u32)
        .collect();
    if idx.len() < k {
        idx = (0..n as u32).collect();
    }
    let n = idx.len();
    if k < n {
        idx.select_nth_unstable_by(k - 1, |a, b| rank_cmp(scores, *a, *b));
        idx.truncate(k);
    }
    idx.sort_unstable_by(|a, b| rank_cmp(scores, *a, *b));
    idx
}

#[pymethods]
impl Index {
    #[new]
    #[pyo3(signature = (corpus, variant, k1, b, epsilon=0.25, delta=0.5))]
    fn new(
        corpus: &Bound<'_, PyAny>,
        variant: &str,
        k1: f64,
        b: f64,
        epsilon: f64,
        delta: f64,
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
                    pyo3::exceptions::PyTypeError::new_err("rapidbm25 tokens must be str")
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
            return Err(PyZeroDivisionError::new_err("division by zero (empty corpus)"));
        }
        let avgdl = total as f64 / n_docs as f64;
        let n = n_docs as f64;

        // IDF, same formulas and summation order as SPEC.md §3
        let mut idf: Vec<f64> = Vec::with_capacity(terms.len());
        let mut average_idf = f64::NAN;
        match variant {
            Variant::Okapi => {
                let mut sum = 0.0f64;
                let mut negative = Vec::new();
                for (t, &f) in nd.iter().enumerate() {
                    let f = f as f64;
                    let v = (n - f + 0.5).ln() - (f + 0.5).ln();
                    idf.push(v);
                    sum += v;
                    if v < 0.0 {
                        negative.push(t);
                    }
                }
                if terms.is_empty() {
                    return Err(PyZeroDivisionError::new_err("division by zero (empty vocabulary)"));
                }
                average_idf = sum / terms.len() as f64;
                let eps = epsilon * average_idf;
                for t in negative {
                    idf[t] = eps;
                }
            }
            Variant::L | Variant::L022 => {
                for &f in &nd {
                    idf.push((n + 1.0).ln() - (f as f64 + 0.5).ln());
                }
            }
            Variant::Plus022 => {
                for &f in &nd {
                    idf.push(((n + 1.0) / f as f64).ln());
                }
            }
            Variant::Plus => {
                for &f in &nd {
                    idf.push((n + 1.0).ln() - (f as f64).ln());
                }
            }
        }

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

        let one_minus_b = 1.0 - b;
        let norm: Vec<f64> = doc_len
            .iter()
            .map(|&dl| {
                let base = one_minus_b + b * dl as f64 / avgdl;
                match variant {
                    Variant::L | Variant::L022 => base,
                    _ => k1 * base,
                }
            })
            .collect();

        let first = absent_part(variant, norm[0], k1);
        let uniform = first.is_finite()
            && norm
                .iter()
                .all(|&x| absent_part(variant, x, k1).to_bits() == first.to_bits());

        Ok(Self {
            variant,
            k1,
            delta,
            n_docs,
            avgdl,
            average_idf,
            doc_len,
            terms,
            lookup,
            idf,
            offsets,
            post_doc,
            post_tf,
            norm,
            absent_partial: if uniform { Some(first) } else { None },
            doc_terms_off,
            doc_terms,
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
        self.terms.len()
    }
    #[getter]
    fn dense_fallback(&self) -> bool {
        self.absent_partial.is_none()
    }

    fn doc_len_list(&self) -> Vec<u64> {
        self.doc_len.clone()
    }

    fn idf_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (t, v) in self.terms.iter().zip(&self.idf) {
            d.set_item(t, *v)?;
        }
        Ok(d)
    }

    fn doc_freqs_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for d in 0..self.n_docs {
            let m = PyDict::new(py);
            for &(t, c) in &self.doc_terms[self.doc_terms_off[d]..self.doc_terms_off[d + 1]] {
                m.set_item(&self.terms[t as usize], c)?;
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
        let score = py.detach(|| {
            let mut s = vec![0.0f64; self.n_docs];
            self.score_into(&q, &mut s);
            s
        });
        Ok(PyArray1::from_vec(py, score))
    }

    /// Top-k doc ids and their scores: score descending, ties by lower doc id.
    /// Uses a partial select (O(N) + O(k log k)) instead of a full sort.
    fn top_k<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
        let q = self.resolve_query(query)?;
        let (ids, vals) = py.detach(|| {
            let mut s = vec![0.0f64; self.n_docs];
            self.score_into(&q, &mut s);
            let ids = top_k(&s, k);
            let vals: Vec<f64> = ids.iter().map(|&i| s[i as usize]).collect();
            (ids.into_iter().map(|i| i as i64).collect::<Vec<i64>>(), vals)
        });
        Ok((ids.to_pyarray(py), vals.to_pyarray(py)))
    }
}

/// Partial-select top-k over any score vector (score desc, id asc).
#[pyfunction]
fn top_k_of<'py>(
    py: Python<'py>,
    scores: numpy::PyReadonlyArray1<'py, f64>,
    k: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let s = scores.as_slice()?;
    let ids = top_k(s, k);
    Ok(ids.into_iter().map(|i| i as i64).collect::<Vec<_>>().to_pyarray(py))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    m.add_function(wrap_pyfunction!(top_k_of, m)?)?;
    Ok(())
}
