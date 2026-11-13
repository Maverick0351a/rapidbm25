//! Incremental BM25 index with stable string ids.
//!
//! Scores use the same kernels as the static index (`score_into`, `okapi_idf`,
//! `compute_norm`). Document order is insertion order: `update` keeps a
//! document's position, `delete` closes the gap. After any sequence of
//! mutations, `get_scores` matches a static index built on `corpus()` with the
//! same variant, parameters and `fixes` flag.
//!
//! Statistics (df, avgdl, idf, length norm) are refreshed lazily on the next
//! query. Posting lists are updated in place; a query does not rebuild them
//! from the token sequences.

use crate::persist::{self, LiveDoc};
use crate::score::{
    absent_uniform, compute_norm, okapi_idf, score_into, top_k, top_k_masked, variant_idf, QueryDedup,
    Variant,
};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::Mutex;

struct Slot {
    alive: bool,
    id: String,
    text: String,
    meta: String,
    /// Original tokens, so a saved index reloads the same sequence the static
    /// builder would have scanned (first-seen order and document length).
    tokens: Vec<String>,
    /// Unique terms in first-occurrence order, with within-document tf.
    terms: Vec<(u32, u32)>,
}

struct Stats {
    idf: Vec<f64>,
    average_idf: f64,
    avgdl: f64,
    norm: Vec<f64>,
    absent: Option<f64>,
    /// slot -> dense position in `order`, or `u32::MAX` when the slot is free.
    slot_pos: Vec<u32>,
    seen: Vec<u32>,
}

impl Stats {
    fn empty() -> Self {
        Self {
            idf: Vec::new(),
            average_idf: f64::NAN,
            avgdl: 0.0,
            norm: Vec::new(),
            absent: None,
            slot_pos: Vec::new(),
            seen: Vec::new(),
        }
    }
}

struct Inner {
    variant: Variant,
    k1: f64,
    b: f64,
    epsilon: f64,
    delta: f64,
    fixes: bool,
    terms: Vec<String>,
    lookup: FxHashMap<String, u32>,
    df: Vec<u32>,
    postings: Vec<Vec<(u32, f64)>>,
    slots: Vec<Slot>,
    free: Vec<u32>,
    /// Live slots, insertion order (score-vector order).
    order: Vec<u32>,
    ids: FxHashMap<String, u32>,
    dirty: bool,
    stats: Stats,
}

impl Inner {
    fn new(variant: Variant, k1: f64, b: f64, epsilon: f64, delta: f64, fixes: bool) -> Self {
        Self {
            variant,
            k1,
            b,
            epsilon,
            delta,
            fixes,
            terms: Vec::new(),
            lookup: FxHashMap::default(),
            df: Vec::new(),
            postings: Vec::new(),
            slots: Vec::new(),
            free: Vec::new(),
            order: Vec::new(),
            ids: FxHashMap::default(),
            dirty: true,
            stats: Stats::empty(),
        }
    }

    fn term_id(&mut self, tok: &str) -> u32 {
        if let Some(&t) = self.lookup.get(tok) {
            return t;
        }
        // shortcut: term ids and per-term document frequencies are u32. A vocabulary
        // or a document frequency past 2^32 wraps. Return an error from term_id
        // when an index can actually hold that many terms.
        let t = self.terms.len() as u32;
        self.lookup.insert(tok.to_owned(), t);
        self.terms.push(tok.to_owned());
        self.df.push(0);
        self.postings.push(Vec::new());
        t
    }

    fn ingest(&mut self, slot: u32, tokens: &[String]) -> Vec<(u32, u32)> {
        let mut local: Vec<(u32, u32)> = Vec::new();
        let mut pos: FxHashMap<u32, usize> = FxHashMap::default();
        for tok in tokens {
            let tid = self.term_id(tok);
            if let Some(&i) = pos.get(&tid) {
                local[i].1 += 1;
            } else {
                pos.insert(tid, local.len());
                local.push((tid, 1));
                self.df[tid as usize] += 1;
            }
        }
        for &(tid, tf) in &local {
            self.postings[tid as usize].push((slot, tf as f64));
        }
        local
    }

    fn strip_slot(&mut self, slot: u32) {
        let terms = std::mem::take(&mut self.slots[slot as usize].terms);
        for (tid, _) in terms {
            let list = &mut self.postings[tid as usize];
            if let Some(i) = list.iter().position(|p| p.0 == slot) {
                list.swap_remove(i);
            }
            self.df[tid as usize] -= 1;
        }
    }

    fn alloc_slot(&mut self) -> u32 {
        if let Some(s) = self.free.pop() {
            s
        } else {
            // shortcut: slot ids are u32. Past 2^32 slots this wraps; check alloc_slot
            // when a live index can hold that many documents.
            let s = self.slots.len() as u32;
            self.slots.push(Slot {
                alive: false,
                id: String::new(),
                text: String::new(),
                meta: String::new(),
                tokens: Vec::new(),
                terms: Vec::new(),
            });
            s
        }
    }

    fn add(&mut self, id: String, tokens: Vec<String>, text: String, meta: String) -> PyResult<()> {
        if self.ids.contains_key(&id) {
            return Err(PyValueError::new_err(format!("duplicate document id {id:?}")));
        }
        let slot = self.alloc_slot();
        let terms = self.ingest(slot, &tokens);
        self.slots[slot as usize] = Slot {
            alive: true,
            id: id.clone(),
            text,
            meta,
            tokens,
            terms,
        };
        self.ids.insert(id, slot);
        self.order.push(slot);
        self.dirty = true;
        Ok(())
    }

    fn delete(&mut self, id: &str) -> bool {
        let Some(slot) = self.ids.remove(id) else {
            return false;
        };
        self.strip_slot(slot);
        let s = &mut self.slots[slot as usize];
        s.alive = false;
        s.id.clear();
        s.text.clear();
        s.meta.clear();
        s.tokens.clear();
        self.order.retain(|&x| x != slot);
        self.free.push(slot);
        self.dirty = true;
        true
    }

    fn update(
        &mut self,
        id: &str,
        tokens: Vec<String>,
        text: String,
        meta: Option<String>,
    ) -> PyResult<()> {
        let slot = *self
            .ids
            .get(id)
            .ok_or_else(|| PyValueError::new_err(format!("unknown document id {id:?}")))?;
        self.strip_slot(slot);
        let terms = self.ingest(slot, &tokens);
        let s = &mut self.slots[slot as usize];
        s.tokens = tokens;
        s.terms = terms;
        s.text = text;
        if let Some(m) = meta {
            s.meta = m;
        }
        self.dirty = true;
        Ok(())
    }

    fn ensure_stats(&mut self) -> PyResult<()> {
        if !self.dirty {
            return Ok(());
        }
        let n_docs = self.order.len();
        let mut slot_pos = vec![u32::MAX; self.slots.len()];
        let mut doc_len = Vec::with_capacity(n_docs);
        let mut seen_flag = vec![false; self.terms.len()];
        let mut seen = Vec::new();
        for (i, &slot) in self.order.iter().enumerate() {
            slot_pos[slot as usize] = i as u32;
            let slot_ref = &self.slots[slot as usize];
            doc_len.push(slot_ref.tokens.len() as u64);
            for &(tid, _) in &slot_ref.terms {
                if !seen_flag[tid as usize] {
                    seen_flag[tid as usize] = true;
                    seen.push(tid);
                }
            }
        }
        if n_docs == 0 {
            self.stats = Stats {
                idf: vec![0.0; self.terms.len()],
                average_idf: if self.variant == Variant::Okapi { 0.0 } else { f64::NAN },
                avgdl: 0.0,
                norm: Vec::new(),
                absent: None,
                slot_pos,
                seen,
            };
            self.dirty = false;
            return Ok(());
        }
        let total: u64 = doc_len.iter().sum();
        let avgdl = total as f64 / n_docs as f64;
        let n = n_docs as f64;
        let dfs: Vec<u64> = seen.iter().map(|&t| self.df[t as usize] as u64).collect();
        let (idf_order, average_idf) = match self.variant {
            Variant::Okapi => okapi_idf(&dfs, n, self.epsilon, self.fixes)?,
            other => (variant_idf(other, &dfs, n), f64::NAN),
        };
        let mut idf = vec![0.0; self.terms.len()];
        for (v, &tid) in idf_order.iter().zip(&seen) {
            idf[tid as usize] = *v;
        }
        let norm = compute_norm(self.variant, &doc_len, avgdl, self.b, self.k1);
        let absent = absent_uniform(self.variant, &norm, self.k1);
        self.stats = Stats {
            idf,
            average_idf,
            avgdl,
            norm,
            absent,
            slot_pos,
            seen,
        };
        self.dirty = false;
        Ok(())
    }

    fn resolve(&self, tokens: &[String]) -> Vec<(Option<u32>, f64)> {
        let mut out = Vec::with_capacity(tokens.len());
        let mut dedup = QueryDedup::default();
        for tok in tokens {
            let (tid, idf) = match self.lookup.get(tok.as_str()) {
                Some(&t) if self.df[t as usize] > 0 => (Some(t), self.stats.idf[t as usize]),
                _ => (None, 0.0),
            };
            if dedup.allow(self.fixes, tid, Some(tok.as_str())) {
                out.push((tid, idf));
            }
        }
        out
    }

    /// Build a CSR in dense insertion-order positions and score with the
    /// static kernel. Posting lists themselves are not rebuilt.
    fn score_resolved(&self, query: &[(Option<u32>, f64)]) -> Vec<f64> {
        let n = self.order.len();
        let vocab = self.terms.len();
        let mut needed = vec![false; vocab];
        for (tid, _) in query {
            if let Some(t) = *tid {
                needed[t as usize] = true;
            }
        }
        let mut lists: Vec<Vec<(u32, f64)>> = Vec::with_capacity(vocab);
        for t in 0..vocab {
            if !needed[t] {
                lists.push(Vec::new());
                continue;
            }
            let mut pairs = Vec::with_capacity(self.postings[t].len());
            for &(slot, tf) in &self.postings[t] {
                let pos = self.stats.slot_pos[slot as usize];
                if pos != u32::MAX {
                    pairs.push((pos, tf));
                }
            }
            pairs.sort_unstable_by_key(|p| p.0);
            lists.push(pairs);
        }
        let mut offsets = vec![0usize; vocab + 1];
        for t in 0..vocab {
            offsets[t + 1] = offsets[t] + lists[t].len();
        }
        let n_posts = offsets[vocab];
        let mut post_doc = vec![0u32; n_posts];
        let mut post_tf = vec![0f64; n_posts];
        for t in 0..vocab {
            for (i, &(pos, tf)) in lists[t].iter().enumerate() {
                post_doc[offsets[t] + i] = pos;
                post_tf[offsets[t] + i] = tf;
            }
        }
        let mut score = vec![0.0f64; n];
        score_into(
            self.variant,
            self.k1,
            self.delta,
            n,
            &self.stats.norm,
            &offsets,
            &post_doc,
            &post_tf,
            self.stats.absent,
            query,
            &mut score,
        );
        score
    }

    fn scores(&mut self, tokens: &[String]) -> PyResult<Vec<f64>> {
        self.ensure_stats()?;
        let q = self.resolve(tokens);
        Ok(self.score_resolved(&q))
    }

    fn docs_in_order(&self) -> Vec<LiveDoc> {
        self.order
            .iter()
            .map(|&slot| {
                let s = &self.slots[slot as usize];
                LiveDoc {
                    id: s.id.clone(),
                    text: s.text.clone(),
                    meta: s.meta.clone(),
                    tokens: s.tokens.clone(),
                }
            })
            .collect()
    }
}

fn lock(inner: &Mutex<Inner>) -> std::sync::MutexGuard<'_, Inner> {
    inner.lock().unwrap_or_else(|e| e.into_inner())
}

#[pyclass(frozen, module = "rapidbm25._native")]
pub struct LiveIndex {
    inner: Mutex<Inner>,
}

#[pymethods]
impl LiveIndex {
    #[new]
    #[pyo3(signature = (variant, k1, b, epsilon, delta, fixes))]
    fn new(variant: &str, k1: f64, b: f64, epsilon: f64, delta: f64, fixes: bool) -> PyResult<Self> {
        let variant = Variant::parse(variant)?;
        Ok(Self {
            inner: Mutex::new(Inner::new(variant, k1, b, epsilon, delta, fixes)),
        })
    }

    #[getter]
    fn corpus_size(&self) -> usize {
        lock(&self.inner).order.len()
    }
    #[getter]
    fn k1(&self) -> f64 {
        lock(&self.inner).k1
    }
    #[getter]
    fn b(&self) -> f64 {
        lock(&self.inner).b
    }
    #[getter]
    fn epsilon(&self) -> f64 {
        lock(&self.inner).epsilon
    }
    #[getter]
    fn delta(&self) -> f64 {
        lock(&self.inner).delta
    }
    #[getter]
    fn fixes(&self) -> bool {
        lock(&self.inner).fixes
    }
    #[getter]
    fn variant(&self) -> String {
        lock(&self.inner).variant.as_str().to_owned()
    }
    #[getter]
    fn compat(&self) -> String {
        lock(&self.inner).variant.compat().to_owned()
    }

    fn avgdl(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| {
            let mut g = lock(&self.inner);
            g.ensure_stats()?;
            Ok(g.stats.avgdl)
        })
    }

    fn average_idf(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| {
            let mut g = lock(&self.inner);
            g.ensure_stats()?;
            Ok(g.stats.average_idf)
        })
    }

    fn add(&self, py: Python<'_>, id: String, tokens: Vec<String>, text: String, meta: String) -> PyResult<()> {
        py.detach(|| lock(&self.inner).add(id, tokens, text, meta))
    }

    fn delete(&self, py: Python<'_>, id: &str) -> bool {
        py.detach(|| lock(&self.inner).delete(id))
    }

    /// `meta` JSON replaces stored metadata when provided; pass None to keep it.
    #[pyo3(signature = (id, tokens, text, meta=None))]
    fn update(
        &self,
        py: Python<'_>,
        id: &str,
        tokens: Vec<String>,
        text: String,
        meta: Option<String>,
    ) -> PyResult<()> {
        py.detach(|| lock(&self.inner).update(id, tokens, text, meta))
    }

    fn ids(&self) -> Vec<String> {
        let g = lock(&self.inner);
        g.order.iter().map(|&s| g.slots[s as usize].id.clone()).collect()
    }

    fn texts(&self) -> Vec<String> {
        let g = lock(&self.inner);
        g.order.iter().map(|&s| g.slots[s as usize].text.clone()).collect()
    }

    fn metas(&self) -> Vec<String> {
        let g = lock(&self.inner);
        g.order.iter().map(|&s| g.slots[s as usize].meta.clone()).collect()
    }

    fn corpus<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let g = lock(&self.inner);
        let out = PyList::empty(py);
        for &slot in &g.order {
            out.append(g.slots[slot as usize].tokens.clone())?;
        }
        Ok(out)
    }

    fn get_scores<'py>(&self, py: Python<'py>, tokens: Vec<String>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let scores = py.detach(|| {
            let mut g = lock(&self.inner);
            g.scores(&tokens)
        })?;
        Ok(PyArray1::from_vec(py, scores))
    }

    fn get_scores_many<'py>(
        &self,
        py: Python<'py>,
        queries: Vec<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n_docs = lock(&self.inner).order.len();
        if queries.is_empty() {
            return Ok(PyArray2::zeros(py, [0, n_docs], false));
        }
        let rows = py.detach(|| {
            let mut g = lock(&self.inner);
            g.ensure_stats()?;
            let resolved: Vec<_> = queries.iter().map(|q| g.resolve(q)).collect();
            let rows: Vec<Vec<f64>> = resolved.par_iter().map(|q| g.score_resolved(q)).collect();
            Ok::<_, PyErr>(rows)
        })?;
        PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Top-k hits as `(id, score, text, meta_json)` in rank order.
    /// `mask`, when set, is aligned with insertion order.
    #[pyo3(signature = (tokens, k, mask=None))]
    fn search(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        k: usize,
        mask: Option<Vec<bool>>,
    ) -> PyResult<Vec<(String, f64, String, String)>> {
        py.detach(|| {
            let mut g = lock(&self.inner);
            g.ensure_stats()?;
            if let Some(ref m) = mask {
                if m.len() != g.order.len() {
                    return Err(PyValueError::new_err(
                        "filter mask length does not match the number of live documents",
                    ));
                }
            }
            let q = g.resolve(&tokens);
            let scores = g.score_resolved(&q);
            let idx = match &mask {
                Some(m) => top_k_masked(&scores, m, k),
                None => top_k(&scores, k),
            };
            let mut out = Vec::with_capacity(idx.len());
            for pos in idx {
                let slot = g.order[pos as usize];
                let s = &g.slots[slot as usize];
                out.push((s.id.clone(), scores[pos as usize], s.text.clone(), s.meta.clone()));
            }
            Ok(out)
        })
    }

    fn idf_items(&self, py: Python<'_>) -> PyResult<Vec<(String, f64)>> {
        py.detach(|| {
            let mut g = lock(&self.inner);
            g.ensure_stats()?;
            let mut out = Vec::with_capacity(g.stats.seen.len());
            for &tid in &g.stats.seen {
                out.push((g.terms[tid as usize].clone(), g.stats.idf[tid as usize]));
            }
            Ok(out)
        })
    }

    fn doc_freq_items(&self) -> Vec<Vec<(String, u32)>> {
        let g = lock(&self.inner);
        g.order
            .iter()
            .map(|&slot| {
                g.slots[slot as usize]
                    .terms
                    .iter()
                    .map(|&(tid, tf)| (g.terms[tid as usize].clone(), tf))
                    .collect()
            })
            .collect()
    }

    fn doc_lens(&self) -> Vec<u64> {
        let g = lock(&self.inner);
        g.order.iter().map(|&s| g.slots[s as usize].tokens.len() as u64).collect()
    }

    #[pyo3(signature = (path, tokenizer_tag=None))]
    fn save(&self, py: Python<'_>, path: &str, tokenizer_tag: Option<String>) -> PyResult<()> {
        let bytes = py.detach(|| {
            let g = lock(&self.inner);
            let docs = g.docs_in_order();
            persist::dump_live(g.variant, g.fixes, g.k1, g.b, g.epsilon, g.delta, tokenizer_tag.as_deref(), &docs)
        })?;
        persist::write_atomic(Path::new(path), &bytes)
    }

    /// Returns `(index, tokenizer_tag_json_or_none)`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<(Self, Option<String>)> {
        let bytes = std::fs::read(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (header, docs) = persist::load_live_docs(&bytes)?;
        let variant = header.get("variant").and_then(|v| v.as_str()).unwrap_or("");
        let fixes = header.get("fixes").and_then(|v| v.as_bool()).unwrap_or(false);
        let num = |key: &str| -> PyResult<f64> {
            let s = header
                .get(key)
                .and_then(|v| v.as_str())
                .ok_or_else(|| PyValueError::new_err(format!("live index missing {key}")))?;
            let bits = u64::from_str_radix(s, 16)
                .map_err(|_| PyValueError::new_err(format!("bad {key} in live index")))?;
            Ok(f64::from_bits(bits))
        };
        let index = Self::new(variant, num("k1")?, num("b")?, num("epsilon")?, num("delta")?, fixes)?;
        {
            let mut g = lock(&index.inner);
            for d in docs {
                g.add(d.id, d.tokens, d.text, d.meta)?;
            }
        }
        let tag = match header.get("tokenizer") {
            None | Some(serde_json::Value::Null) => None,
            Some(v) => Some(v.to_string()),
        };
        Ok((index, tag))
    }
}
