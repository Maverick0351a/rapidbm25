//! Scoring kernels shared by the static index and the live index.
//!
//! Every arithmetic expression below keeps the IEEE-754 operation order of the
//! rank_bm25 formulas. Do not introduce `mul_add`, reassociation, or a second
//! copy of these expressions.

use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use rustc_hash::FxHashSet;
use std::cmp::Ordering;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Variant {
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
    pub(crate) fn parse(s: &str) -> PyResult<Self> {
        match s {
            "okapi" => Ok(Variant::Okapi),
            "l" => Ok(Variant::L),
            "l022" => Ok(Variant::L022),
            "plus" => Ok(Variant::Plus),
            "plus022" => Ok(Variant::Plus022),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "variant must be 'okapi', 'l', 'l022', 'plus' or 'plus022'",
            )),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Variant::Okapi => "okapi",
            Variant::L => "l",
            Variant::L022 => "l022",
            Variant::Plus => "plus",
            Variant::Plus022 => "plus022",
        }
    }

    pub(crate) fn class_name(self) -> &'static str {
        match self {
            Variant::Okapi => "BM25Okapi",
            Variant::L | Variant::L022 => "BM25L",
            Variant::Plus | Variant::Plus022 => "BM25Plus",
        }
    }

    pub(crate) fn compat(self) -> &'static str {
        match self {
            Variant::L | Variant::Plus => "47aa3dd",
            _ => "0.2.2",
        }
    }
}

#[inline(always)]
pub(crate) fn contrib(v: Variant, idf: f64, tf: f64, norm: f64, k1: f64, delta: f64) -> f64 {
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
pub(crate) fn absent_part(v: Variant, norm: f64, k1: f64) -> f64 {
    match v {
        Variant::Okapi => 0.0 * (k1 + 1.0) / (0.0 + norm),
        Variant::L | Variant::L022 => 0.0 / norm,
        Variant::Plus | Variant::Plus022 => (0.0 * (k1 + 1.0)) / (norm + 0.0),
    }
}

pub(crate) fn absent_uniform(variant: Variant, norm: &[f64], k1: f64) -> Option<f64> {
    if norm.is_empty() {
        return None;
    }
    let first = absent_part(variant, norm[0], k1);
    let uniform = first.is_finite()
        && norm
            .iter()
            .all(|&x| absent_part(variant, x, k1).to_bits() == first.to_bits());
    if uniform {
        Some(first)
    } else {
        None
    }
}

pub(crate) fn compute_norm(variant: Variant, doc_len: &[u64], avgdl: f64, b: f64, k1: f64) -> Vec<f64> {
    let one_minus_b = 1.0 - b;
    doc_len
        .iter()
        .map(|&dl| {
            let base = one_minus_b + b * dl as f64 / avgdl;
            match variant {
                Variant::L | Variant::L022 => base,
                _ => k1 * base,
            }
        })
        .collect()
}

/// Okapi idf, same summation order as the static builder.
/// `dfs` is in first-seen term order. Returns `(idf_in_that_order, average_idf)`.
pub(crate) fn okapi_idf(dfs: &[u64], n_docs: f64, epsilon: f64, fixes: bool) -> PyResult<(Vec<f64>, f64)> {
    if !fixes {
        // Verbatim non-fixes path (empty vocabulary is an error).
        if dfs.is_empty() {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "division by zero (empty vocabulary)",
            ));
        }
        let mut idf = Vec::with_capacity(dfs.len());
        let mut sum = 0.0f64;
        let mut negative = Vec::new();
        for (t, &f) in dfs.iter().enumerate() {
            let f = f as f64;
            let v = (n_docs - f + 0.5).ln() - (f + 0.5).ln();
            idf.push(v);
            sum += v;
            if v < 0.0 {
                negative.push(t);
            }
        }
        let average_idf = sum / dfs.len() as f64;
        let eps = epsilon * average_idf;
        for t in negative {
            idf[t] = eps;
        }
        return Ok((idf, average_idf));
    }
    if dfs.is_empty() {
        return Ok((Vec::new(), 0.0));
    }
    let mut idf = Vec::with_capacity(dfs.len());
    let mut sum = 0.0f64;
    let mut negative = Vec::new();
    let mut zeros = Vec::new();
    for (t, &f) in dfs.iter().enumerate() {
        let f = f as f64;
        let v = (n_docs - f + 0.5).ln() - (f + 0.5).ln();
        idf.push(v);
        sum += v;
        if v < 0.0 {
            negative.push(t);
        } else if v == 0.0 {
            zeros.push(t);
        }
    }
    let average_idf = sum / dfs.len() as f64;
    let eps = epsilon * average_idf;
    for t in negative {
        idf[t] = eps;
    }
    // rank_bm25 #39: a term in exactly half the documents has idf 0, which ranks
    // below the epsilon floor given to terms in *more* than half. Only lift exact
    // zeros, and only when the floor is positive, so every other score stays put
    // and a non-positive floor cannot rank a match below a non-match.
    if eps > 0.0 {
        for t in zeros {
            idf[t] = eps;
        }
    }
    Ok((idf, average_idf))
}

pub(crate) fn variant_idf(variant: Variant, dfs: &[u64], n_docs: f64) -> Vec<f64> {
    let n = n_docs;
    match variant {
        Variant::Okapi => unreachable!("okapi uses okapi_idf"),
        Variant::L | Variant::L022 => dfs
            .iter()
            .map(|&f| (n + 1.0).ln() - (f as f64 + 0.5).ln())
            .collect(),
        Variant::Plus022 => dfs
            .iter()
            .map(|&f| ((n + 1.0) / f as f64).ln())
            .collect(),
        Variant::Plus => dfs
            .iter()
            .map(|&f| (n + 1.0).ln() - (f as f64).ln())
            .collect(),
    }
}

/// Same loop as the original `Index::score_into`. `offsets[t]` / `offsets[t+1]`
/// select `post_doc` / `post_tf`, which are ascending doc ids.
pub(crate) fn score_into(
    variant: Variant,
    k1: f64,
    delta: f64,
    n_docs: usize,
    norm: &[f64],
    offsets: &[usize],
    post_doc: &[u32],
    post_tf: &[f64],
    absent_partial: Option<f64>,
    query: &[(Option<u32>, f64)],
    score: &mut [f64],
) {
    let (v, k1, delta) = (variant, k1, delta);
    for &(tid, idf) in query {
        let (s, e) = match tid {
            Some(t) => (offsets[t as usize], offsets[t as usize + 1]),
            None => (0, 0),
        };
        let docs = &post_doc[s..e];
        let tfs = &post_tf[s..e];
        match absent_partial {
            Some(_) => {
                // value every document without this term receives
                let c = contrib(v, idf, 0.0, norm.first().copied().unwrap_or(1.0), k1, delta);
                if c == 0.0 {
                    // adding +/-0.0 never changes a score (scores start at +0.0)
                    for (d, tf) in docs.iter().zip(tfs) {
                        let d = *d as usize;
                        score[d] += contrib(v, idf, *tf, norm[d], k1, delta);
                    }
                } else {
                    let mut p = 0usize;
                    for d in 0..n_docs {
                        if p < docs.len() && docs[p] as usize == d {
                            score[d] += contrib(v, idf, tfs[p], norm[d], k1, delta);
                            p += 1;
                        } else {
                            score[d] += c;
                        }
                    }
                }
            }
            None => {
                let mut p = 0usize;
                for d in 0..n_docs {
                    let tf = if p < docs.len() && docs[p] as usize == d {
                        p += 1;
                        tfs[p - 1]
                    } else {
                        0.0
                    };
                    score[d] += contrib(v, idf, tf, norm[d], k1, delta);
                }
            }
        }
    }
}

/// Query-term dedupe for `fixes=True` (rank_bm25 #19): first occurrence wins.
#[derive(Default)]
pub(crate) struct QueryDedup {
    terms: FxHashSet<u32>,
    unknown: FxHashSet<String>,
    nonstring: bool,
}

impl QueryDedup {
    pub(crate) fn allow(&mut self, fixes: bool, tid: Option<u32>, token: Option<&str>) -> bool {
        if !fixes {
            return true;
        }
        match tid {
            Some(t) => self.terms.insert(t),
            None => match token {
                Some(s) => self.unknown.insert(s.to_owned()),
                None => {
                    if self.nonstring {
                        false
                    } else {
                        self.nonstring = true;
                        true
                    }
                }
            },
        }
    }
}

/// Descending score, then ascending doc id. NaN sorts last.
#[inline(always)]
pub(crate) fn rank_cmp(scores: &[f64], a: u32, b: u32) -> Ordering {
    let (x, y) = (scores[a as usize], scores[b as usize]);
    match (x.is_nan(), y.is_nan()) {
        (true, true) => a.cmp(&b),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => y.partial_cmp(&x).unwrap().then(a.cmp(&b)),
    }
}

pub(crate) fn top_k(scores: &[f64], k: usize) -> Vec<u32> {
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

/// `top_k` restricted to positions where `mask` is true. The comparator is the
/// same total order, so the returned set is the top-k of that subset.
pub(crate) fn top_k_masked(scores: &[f64], mask: &[bool], k: usize) -> Vec<u32> {
    let allowed = mask.iter().filter(|&&m| m).count();
    let k = k.min(allowed);
    if k == 0 {
        return Vec::new();
    }
    let mut idx: Vec<u32> = scores
        .iter()
        .enumerate()
        .filter(|(i, &x)| mask[*i] && x > 0.0)
        .map(|(i, _)| i as u32)
        .collect();
    if idx.len() < k {
        idx = mask
            .iter()
            .enumerate()
            .filter(|(_, &m)| m)
            .map(|(i, _)| i as u32)
            .collect();
    }
    let n = idx.len();
    if k < n {
        idx.select_nth_unstable_by(k - 1, |a, b| rank_cmp(scores, *a, *b));
        idx.truncate(k);
    }
    idx.sort_unstable_by(|a, b| rank_cmp(scores, *a, *b));
    idx
}

/// Partial-select top-k over any score vector (score desc, id asc).
#[pyfunction]
pub(crate) fn top_k_of<'py>(
    py: Python<'py>,
    scores: PyReadonlyArray1<'py, f64>,
    k: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let s = scores.as_slice()?;
    let ids = top_k(s, k);
    Ok(ids
        .into_iter()
        .map(|i| i as i64)
        .collect::<Vec<_>>()
        .to_pyarray(py))
}
