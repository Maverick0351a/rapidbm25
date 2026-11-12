//! Versioned on-disk index (`RBM25` format 1).
//!
//! Layout, little-endian:
//! ```text
//! 0..8     magic b"RBM25\\x01\\n\\x00"
//! 8..16    header length (u64)
//! 16..     UTF-8 JSON header
//! pad to 8
//!          numeric sections described by the header
//! ```
//! Scalar floats are hex IEEE-754 bit patterns, so a round trip does not go
//! through decimal. On 64-bit little-endian hosts a static index can be
//! memory-mapped and scored without copying the posting columns. Other
//! targets (and `mmap=False`) copy the columns into owned vectors.
//!
//! The live index stores documents (id, text, metadata, tokens) and is rebuilt
//! by replaying `add` on load, so it stays mutable. Mapped scoring is for the
//! immutable static index.

use crate::index::{Data, Index, Owned};
use crate::score::Variant;
use memmap2::Mmap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use serde_json::{json, Value};
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub(crate) const MAGIC: &[u8; 8] = b"RBM25\x01\n\x00";

fn hex_u64(v: u64) -> String {
    format!("{v:016x}")
}
fn hex_f64(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}
fn parse_hex_u64(s: &str) -> PyResult<u64> {
    u64::from_str_radix(s, 16).map_err(|_| PyValueError::new_err("bad hex integer in index header"))
}
fn parse_hex_f64(s: &str) -> PyResult<f64> {
    Ok(f64::from_bits(parse_hex_u64(s)?))
}

struct Pay {
    buf: Vec<u8>,
}

impl Pay {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
    fn align(&mut self) {
        while self.buf.len() % 8 != 0 {
            self.buf.push(0);
        }
    }
    fn rel(&self) -> u64 {
        self.buf.len() as u64
    }
    fn write_u64s(&mut self, xs: &[u64]) -> (String, u64) {
        self.align();
        let off = self.rel();
        for x in xs {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
        (hex_u64(off), xs.len() as u64)
    }
    fn write_u32s(&mut self, xs: &[u32]) -> (String, u64) {
        self.align();
        let off = self.rel();
        for x in xs {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
        (hex_u64(off), xs.len() as u64)
    }
    fn write_f64s(&mut self, xs: &[f64]) -> (String, u64) {
        self.align();
        let off = self.rel();
        for x in xs {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
        (hex_u64(off), xs.len() as u64)
    }
    fn write_bytes(&mut self, bytes: &[u8]) -> (String, u64) {
        let off = self.rel();
        self.buf.extend_from_slice(bytes);
        (hex_u64(off), bytes.len() as u64)
    }
}

fn wrap(header_at: impl Fn(u64) -> Value, payload: &[u8]) -> PyResult<Vec<u8>> {
    let probe = serde_json::to_vec(&header_at(0)).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let header_len = probe.len();
    let prefix = 16usize;
    let pad = (8 - ((prefix + header_len) % 8)) % 8;
    let base = (prefix + header_len + pad) as u64;
    let header = serde_json::to_vec(&header_at(base)).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if header.len() != header_len {
        return Err(PyValueError::new_err(
            "internal error: index header size changed while filling section offsets",
        ));
    }
    let mut out = Vec::with_capacity(base as usize + payload.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(header_len as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend(std::iter::repeat(0u8).take(pad));
    out.extend_from_slice(payload);
    Ok(out)
}

pub(crate) struct Columns {
    pub doc_len: Vec<u64>,
    pub terms: Vec<String>,
    pub idf: Vec<f64>,
    pub offsets: Vec<u64>,
    pub post_doc: Vec<u32>,
    pub post_tf: Vec<f64>,
    pub norm: Vec<f64>,
    pub doc_terms_off: Vec<u64>,
    pub doc_terms_tid: Vec<u32>,
    pub doc_terms_tf: Vec<u32>,
}

fn section_pair(v: &Value, name: &str) -> PyResult<(u64, u64)> {
    let arr = v
        .get(name)
        .and_then(|x| x.as_array())
        .ok_or_else(|| PyValueError::new_err(format!("missing section {name}")))?;
    if arr.len() != 2 {
        return Err(PyValueError::new_err(format!("bad section {name}")));
    }
    let off = parse_hex_u64(arr[0].as_str().unwrap_or(""))?;
    let n = arr[1].as_u64().ok_or_else(|| PyValueError::new_err(format!("bad section {name}")))?;
    Ok((off, n))
}

fn need<'a>(file: &'a [u8], off: u64, nbytes: u64) -> PyResult<&'a [u8]> {
    let start = usize::try_from(off).map_err(|_| PyValueError::new_err("offset overflow"))?;
    let n = usize::try_from(nbytes).map_err(|_| PyValueError::new_err("length overflow"))?;
    let end = start.checked_add(n).ok_or_else(|| PyValueError::new_err("section overflow"))?;
    if end > file.len() {
        return Err(PyValueError::new_err("index section extends past end of file"));
    }
    Ok(&file[start..end])
}

fn read_u64s(file: &[u8], off: u64, n: u64) -> PyResult<Vec<u64>> {
    let nusz = usize::try_from(n).map_err(|_| PyValueError::new_err("length overflow"))?;
    let nbytes = n
        .checked_mul(8)
        .ok_or_else(|| PyValueError::new_err("section too large"))?;
    if n != 0 && (off % 8) != 0 {
        return Err(PyValueError::new_err("u64/f64 section is not 8-byte aligned"));
    }
    let bytes = need(file, off, nbytes)?;
    let mut out = Vec::with_capacity(nusz);
    for i in 0..nusz {
        let s = i * 8;
        out.push(u64::from_le_bytes(bytes[s..s + 8].try_into().unwrap()));
    }
    Ok(out)
}

fn read_u32s(file: &[u8], off: u64, n: u64) -> PyResult<Vec<u32>> {
    let nusz = usize::try_from(n).map_err(|_| PyValueError::new_err("length overflow"))?;
    let nbytes = n
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("section too large"))?;
    if n != 0 && (off % 4) != 0 {
        return Err(PyValueError::new_err("u32 section is not 4-byte aligned"));
    }
    let bytes = need(file, off, nbytes)?;
    let mut out = Vec::with_capacity(nusz);
    for i in 0..nusz {
        let s = i * 4;
        out.push(u32::from_le_bytes(bytes[s..s + 4].try_into().unwrap()));
    }
    Ok(out)
}

fn read_f64s(file: &[u8], off: u64, n: u64) -> PyResult<Vec<f64>> {
    Ok(read_u64s(file, off, n)?.into_iter().map(f64::from_bits).collect())
}

fn validate_monotonic(name: &str, offsets: &[u64], expected_last: u64) -> PyResult<()> {
    let mut prev = 0u64;
    for (i, &off) in offsets.iter().enumerate() {
        if i == 0 && off != 0 {
            return Err(PyValueError::new_err(format!("{name} must start at 0")));
        }
        if off < prev {
            return Err(PyValueError::new_err(format!("{name} is not monotonic")));
        }
        prev = off;
    }
    if prev != expected_last {
        return Err(PyValueError::new_err(format!(
            "{name} does not cover the section ({prev} != {expected_last})"
        )));
    }
    Ok(())
}

/// Reject corrupt posting tables before any score loop can index them.
fn validate_static_columns(
    n_docs: usize,
    n_terms: usize,
    offsets: &[u64],
    post_doc: &[u32],
    doc_terms_off: &[u64],
    doc_terms_tid: &[u32],
    doc_terms_tf: &[u32],
) -> PyResult<()> {
    if offsets.len() != n_terms + 1 {
        return Err(PyValueError::new_err("offsets length must be n_terms + 1"));
    }
    if doc_terms_off.len() != n_docs + 1 {
        return Err(PyValueError::new_err("doc_terms_off length must be n_docs + 1"));
    }
    if doc_terms_tid.len() != doc_terms_tf.len() {
        return Err(PyValueError::new_err("doc term columns disagree"));
    }
    validate_monotonic("offsets", offsets, post_doc.len() as u64)?;
    validate_monotonic("doc_terms_off", doc_terms_off, doc_terms_tid.len() as u64)?;
    let mut prev_off = 0u64;
    for &off in offsets {
        let start = prev_off as usize;
        let end = off as usize;
        let mut last: Option<u32> = None;
        for &doc in &post_doc[start..end] {
            if doc as usize >= n_docs {
                return Err(PyValueError::new_err("posting doc id is out of range"));
            }
            if let Some(prev) = last {
                if doc <= prev {
                    return Err(PyValueError::new_err("postings are not strictly increasing"));
                }
            }
            last = Some(doc);
        }
        prev_off = off;
    }
    for &tid in doc_terms_tid {
        if n_terms == 0 || tid as usize >= n_terms {
            return Err(PyValueError::new_err("doc term id is out of range"));
        }
    }
    Ok(())
}

fn read_strings(bytes: &[u8], count: usize) -> PyResult<Vec<String>> {
    if count > bytes.len() {
        return Err(PyValueError::new_err("string table count exceeds the section"));
    }
    let mut out = Vec::new();
    let mut i = 0;
    while out.len() < count {
        if i + 4 > bytes.len() {
            return Err(PyValueError::new_err("truncated string table"));
        }
        let n = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        if i + n > bytes.len() {
            return Err(PyValueError::new_err("truncated string"));
        }
        let s = std::str::from_utf8(&bytes[i..i + n])
            .map_err(|_| PyValueError::new_err("term is not utf-8"))?;
        out.push(s.to_owned());
        i += n;
    }
    Ok(out)
}

fn push_str(buf: &mut Vec<u8>, s: &str) {
    let b = s.as_bytes();
    buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
    buf.extend_from_slice(b);
}

pub(crate) fn dump_static(index: &Index) -> PyResult<Vec<u8>> {
    let cols = index.columns();
    let mut pay = Pay::new();
    let doc_len = pay.write_u64s(&cols.doc_len);
    let idf = pay.write_f64s(&cols.idf);
    let offsets = pay.write_u64s(&cols.offsets);
    let post_doc = pay.write_u32s(&cols.post_doc);
    let post_tf = pay.write_f64s(&cols.post_tf);
    let norm = pay.write_f64s(&cols.norm);
    let doc_terms_off = pay.write_u64s(&cols.doc_terms_off);
    let doc_terms_tid = pay.write_u32s(&cols.doc_terms_tid);
    let doc_terms_tf = pay.write_u32s(&cols.doc_terms_tf);
    let mut term_blob = Vec::new();
    for t in &cols.terms {
        push_str(&mut term_blob, t);
    }
    let terms = pay.write_bytes(&term_blob);

    let variant = index.variant.as_str();
    let fixes = index.fixes;
    let k1 = hex_f64(index.k1);
    let b = hex_f64(index.b);
    let epsilon = hex_f64(index.epsilon);
    let delta = hex_f64(index.delta);
    let avgdl = hex_f64(index.avgdl);
    let average_idf = hex_f64(index.average_idf);
    let n_docs = cols.doc_len.len();
    let n_terms = cols.terms.len();
    let n_posts = cols.post_doc.len();
    let n_doc_terms = cols.doc_terms_tid.len();

    let add = |rel: &str, base: u64| -> PyResult<String> {
        Ok(hex_u64(base + parse_hex_u64(rel)?))
    };

    wrap(
        |base| {
            json!({
                "format": 1,
                "kind": "static",
                "endian": "little",
                "variant": variant,
                "class": index.variant.class_name(),
                "compat": index.variant.compat(),
                "fixes": fixes,
                "k1": k1,
                "b": b,
                "epsilon": epsilon,
                "delta": delta,
                "avgdl": avgdl,
                "average_idf": average_idf,
                "n_docs": n_docs,
                "n_terms": n_terms,
                "n_posts": n_posts,
                "n_doc_terms": n_doc_terms,
                "sections": {
                    "doc_len": [add(&doc_len.0, base).unwrap(), doc_len.1],
                    "idf": [add(&idf.0, base).unwrap(), idf.1],
                    "offsets": [add(&offsets.0, base).unwrap(), offsets.1],
                    "post_doc": [add(&post_doc.0, base).unwrap(), post_doc.1],
                    "post_tf": [add(&post_tf.0, base).unwrap(), post_tf.1],
                    "norm": [add(&norm.0, base).unwrap(), norm.1],
                    "doc_terms_off": [add(&doc_terms_off.0, base).unwrap(), doc_terms_off.1],
                    "doc_terms_tid": [add(&doc_terms_tid.0, base).unwrap(), doc_terms_tid.1],
                    "doc_terms_tf": [add(&doc_terms_tf.0, base).unwrap(), doc_terms_tf.1],
                    "terms": [add(&terms.0, base).unwrap(), terms.1],
                }
            })
        },
        &pay.buf,
    )
}

pub(crate) fn parse_header(file: &[u8]) -> PyResult<Value> {
    if file.len() < 16 || file.get(..8) != Some(MAGIC.as_slice()) {
        return Err(PyValueError::new_err("not a rapidbm25 index (bad magic)"));
    }
    let header_len = u64::from_le_bytes(file[8..16].try_into().unwrap()) as usize;
    if 16 + header_len > file.len() {
        return Err(PyValueError::new_err("truncated index header"));
    }
    let header: Value = serde_json::from_slice(&file[16..16 + header_len])
        .map_err(|e| PyValueError::new_err(format!("index header is not json: {e}")))?;
    let format = header.get("format").and_then(|v| v.as_u64()).unwrap_or(0);
    if format != 1 {
        return Err(PyValueError::new_err(format!(
            "unsupported rapidbm25 index format {format}"
        )));
    }
    match header.get("endian").and_then(|v| v.as_str()) {
        Some("little") => {}
        _ => {
            return Err(PyValueError::new_err(
                "index endian must be \"little\" (this build only reads little-endian files)",
            ))
        }
    }
    Ok(header)
}

fn scalars(header: &Value) -> PyResult<(Variant, bool, f64, f64, f64, f64, f64, f64)> {
    let variant = Variant::parse(header.get("variant").and_then(|v| v.as_str()).unwrap_or(""))?;
    let fixes = header.get("fixes").and_then(|v| v.as_bool()).unwrap_or(false);
    let k1 = parse_hex_f64(header.get("k1").and_then(|v| v.as_str()).unwrap_or(""))?;
    let b = parse_hex_f64(header.get("b").and_then(|v| v.as_str()).unwrap_or(""))?;
    let epsilon = parse_hex_f64(header.get("epsilon").and_then(|v| v.as_str()).unwrap_or(""))?;
    let delta = parse_hex_f64(header.get("delta").and_then(|v| v.as_str()).unwrap_or(""))?;
    let avgdl = parse_hex_f64(header.get("avgdl").and_then(|v| v.as_str()).unwrap_or(""))?;
    let average_idf = parse_hex_f64(header.get("average_idf").and_then(|v| v.as_str()).unwrap_or(""))?;
    Ok((variant, fixes, k1, b, epsilon, delta, avgdl, average_idf))
}

fn can_mmap() -> bool {
    cfg!(all(target_pointer_width = "64", target_endian = "little"))
}

pub(crate) fn load_static_bytes(file: &[u8]) -> PyResult<Index> {
    let header = parse_header(file)?;
    if header.get("kind").and_then(|v| v.as_str()) != Some("static") {
        return Err(PyValueError::new_err("index file is not a static BM25 index"));
    }
    let (variant, fixes, k1, b, epsilon, delta, avgdl, average_idf) = scalars(&header)?;
    let sections = header
        .get("sections")
        .ok_or_else(|| PyValueError::new_err("missing sections"))?;
    let (doc_len_off, n_docs) = section_pair(sections, "doc_len")?;
    let (idf_off, n_terms) = section_pair(sections, "idf")?;
    let (offsets_off, n_offsets) = section_pair(sections, "offsets")?;
    let (post_doc_off, n_posts) = section_pair(sections, "post_doc")?;
    let (post_tf_off, n_posts2) = section_pair(sections, "post_tf")?;
    let (norm_off, n_norm) = section_pair(sections, "norm")?;
    let (dto_off, n_dto) = section_pair(sections, "doc_terms_off")?;
    let (dtt_off, n_doc_terms) = section_pair(sections, "doc_terms_tid")?;
    let (dtf_off, n_doc_terms2) = section_pair(sections, "doc_terms_tf")?;
    let (terms_off, terms_len) = section_pair(sections, "terms")?;
    if n_posts != n_posts2 || n_doc_terms != n_doc_terms2 || n_norm != n_docs {
        return Err(PyValueError::new_err("inconsistent section lengths"));
    }
    if n_offsets != n_terms + 1 || n_dto != n_docs + 1 {
        return Err(PyValueError::new_err("inconsistent offset table lengths"));
    }
    if post_doc_off % 4 != 0 || dtt_off % 4 != 0 || dtf_off % 4 != 0 {
        return Err(PyValueError::new_err("u32 section is not 4-byte aligned"));
    }
    let doc_len = read_u64s(file, doc_len_off, n_docs)?;
    let idf = read_f64s(file, idf_off, n_terms)?;
    let offsets_u = read_u64s(file, offsets_off, n_offsets)?;
    let post_doc = read_u32s(file, post_doc_off, n_posts)?;
    let post_tf = read_f64s(file, post_tf_off, n_posts)?;
    let norm = read_f64s(file, norm_off, n_norm)?;
    let doc_terms_off_u = read_u64s(file, dto_off, n_dto)?;
    let doc_terms_tid = read_u32s(file, dtt_off, n_doc_terms)?;
    let doc_terms_tf = read_u32s(file, dtf_off, n_doc_terms2)?;
    let term_bytes = need(file, terms_off, terms_len)?;
    let n_terms_us = usize::try_from(n_terms).map_err(|_| PyValueError::new_err("overflow"))?;
    let terms = read_strings(term_bytes, n_terms_us)?;
    finish_owned(
        variant,
        fixes,
        k1,
        b,
        epsilon,
        delta,
        avgdl,
        average_idf,
        doc_len,
        terms,
        idf,
        offsets_u,
        post_doc,
        post_tf,
        norm,
        doc_terms_off_u,
        doc_terms_tid,
        doc_terms_tf,
        false,
    )
}

fn finish_owned(
    variant: Variant,
    fixes: bool,
    k1: f64,
    b: f64,
    epsilon: f64,
    delta: f64,
    avgdl: f64,
    average_idf: f64,
    doc_len: Vec<u64>,
    terms: Vec<String>,
    idf: Vec<f64>,
    offsets_u: Vec<u64>,
    post_doc: Vec<u32>,
    post_tf: Vec<f64>,
    norm: Vec<f64>,
    doc_terms_off_u: Vec<u64>,
    doc_terms_tid: Vec<u32>,
    doc_terms_tf: Vec<u32>,
    mapped: bool,
) -> PyResult<Index> {
    if idf.len() != terms.len() {
        return Err(PyValueError::new_err("idf length does not match vocabulary"));
    }
    validate_static_columns(
        doc_len.len(),
        terms.len(),
        &offsets_u,
        &post_doc,
        &doc_terms_off_u,
        &doc_terms_tid,
        &doc_terms_tf,
    )?;
    let mut lookup = FxHashMap::default();
    for (i, t) in terms.iter().enumerate() {
        lookup.insert(t.clone(), i as u32);
    }
    let offsets: Vec<usize> = offsets_u
        .iter()
        .map(|&x| usize::try_from(x).map_err(|_| PyValueError::new_err("offset does not fit usize")))
        .collect::<PyResult<_>>()?;
    let doc_terms_off: Vec<usize> = doc_terms_off_u
        .iter()
        .map(|&x| usize::try_from(x).map_err(|_| PyValueError::new_err("offset does not fit usize")))
        .collect::<PyResult<_>>()?;
    let mut doc_terms = Vec::with_capacity(doc_terms_tid.len());
    for (tid, tf) in doc_terms_tid.into_iter().zip(doc_terms_tf) {
        doc_terms.push((tid, tf));
    }
    let absent_partial = crate::score::absent_uniform(variant, &norm, k1);
    let n_docs = doc_len.len();
    Ok(Index {
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
        mapped,
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

/// Map `path` and, on 64-bit little-endian, score from the mapping.
pub(crate) fn load_static_mmap(path: &Path) -> PyResult<Index> {
    let file = File::open(path).map_err(|e| PyValueError::new_err(format!("cannot open index: {e}")))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| PyValueError::new_err(format!("cannot map index: {e}")))?;
    if !can_mmap() {
        return load_static_bytes(&mmap);
    }
    load_static_mapped(mmap)
}

fn load_static_mapped(mmap: Mmap) -> PyResult<Index> {
    let header = parse_header(&mmap)?;
    if header.get("kind").and_then(|v| v.as_str()) != Some("static") {
        return Err(PyValueError::new_err("index file is not a static BM25 index"));
    }
    let (variant, fixes, k1, b, epsilon, delta, avgdl, average_idf) = scalars(&header)?;
    let sections = header
        .get("sections")
        .ok_or_else(|| PyValueError::new_err("missing sections"))?;
    let (doc_len_off, n_docs) = section_pair(sections, "doc_len")?;
    let (idf_off, n_terms) = section_pair(sections, "idf")?;
    let (offsets_off, n_offsets) = section_pair(sections, "offsets")?;
    let (post_doc_off, n_posts) = section_pair(sections, "post_doc")?;
    let (post_tf_off, n_posts2) = section_pair(sections, "post_tf")?;
    let (norm_off, n_norm) = section_pair(sections, "norm")?;
    let (dto_off, n_dto) = section_pair(sections, "doc_terms_off")?;
    let (dtt_off, n_doc_terms) = section_pair(sections, "doc_terms_tid")?;
    let (dtf_off, n_doc_terms2) = section_pair(sections, "doc_terms_tf")?;
    let (terms_off, terms_len) = section_pair(sections, "terms")?;
    // Validate every numeric span, alignment, and posting id before keeping the map.
    if n_offsets != n_terms + 1 || n_dto != n_docs + 1 {
        return Err(PyValueError::new_err("inconsistent offset table lengths"));
    }
    if n_posts != n_posts2 || n_doc_terms != n_doc_terms2 || n_norm != n_docs {
        return Err(PyValueError::new_err("inconsistent section lengths"));
    }
    if post_doc_off % 4 != 0 || dtt_off % 4 != 0 || dtf_off % 4 != 0 {
        return Err(PyValueError::new_err("u32 section is not 4-byte aligned"));
    }
    let _ = read_u64s(&mmap, doc_len_off, n_docs)?;
    let _ = read_f64s(&mmap, idf_off, n_terms)?;
    let offsets_u = read_u64s(&mmap, offsets_off, n_offsets)?;
    let post_doc = read_u32s(&mmap, post_doc_off, n_posts)?;
    let _ = read_f64s(&mmap, post_tf_off, n_posts)?;
    let _ = read_f64s(&mmap, norm_off, n_docs)?;
    let doc_terms_off_u = read_u64s(&mmap, dto_off, n_docs + 1)?;
    let doc_terms_tid = read_u32s(&mmap, dtt_off, n_doc_terms)?;
    let doc_terms_tf = read_u32s(&mmap, dtf_off, n_doc_terms)?;
    let n_docs_us = usize::try_from(n_docs).map_err(|_| PyValueError::new_err("overflow"))?;
    let n_terms_us = usize::try_from(n_terms).map_err(|_| PyValueError::new_err("overflow"))?;
    validate_static_columns(
        n_docs_us,
        n_terms_us,
        &offsets_u,
        &post_doc,
        &doc_terms_off_u,
        &doc_terms_tid,
        &doc_terms_tf,
    )?;
    let term_bytes = need(&mmap, terms_off, terms_len)?;
    let terms = read_strings(term_bytes, n_terms_us)?;
    let mut lookup = FxHashMap::default();
    for (i, t) in terms.iter().enumerate() {
        lookup.insert(t.clone(), i as u32);
    }
    // Absent-partial is a property of `norm`. Touch the mapped norm once.
    let norm_tmp = read_f64s(&mmap, norm_off, n_docs)?;
    let absent_partial = crate::score::absent_uniform(variant, &norm_tmp, k1);
    Ok(Index {
        variant,
        k1,
        b,
        epsilon,
        delta,
        fixes,
        n_docs: n_docs_us,
        avgdl,
        average_idf,
        absent_partial,
        mapped: true,
        data: Data::Mapped(crate::index::Mapped {
            mmap,
            terms,
            lookup,
            doc_len_off: doc_len_off as usize,
            idf_off: idf_off as usize,
            offsets_off: offsets_off as usize,
            post_doc_off: post_doc_off as usize,
            post_tf_off: post_tf_off as usize,
            norm_off: norm_off as usize,
            doc_terms_off_off: dto_off as usize,
            doc_terms_tid_off: dtt_off as usize,
            doc_terms_tf_off: dtf_off as usize,
            n_terms: n_terms_us,
            n_posts: n_posts as usize,
            n_doc_terms: n_doc_terms as usize,
        }),
    })
}

pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> PyResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
    }
    let tmp = path.with_extension("rbm25.tmp");
    {
        let mut f = File::create(&tmp).map_err(|e| PyValueError::new_err(e.to_string()))?;
        f.write_all(bytes).map_err(|e| PyValueError::new_err(e.to_string()))?;
        f.sync_all().ok();
    }
    if std::fs::rename(&tmp, path).is_err() {
        // Windows rename does not replace an existing destination.
        std::fs::remove_file(path).map_err(|e| PyValueError::new_err(format!("cannot replace {path:?}: {e}")))?;
        std::fs::rename(&tmp, path).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    Ok(())
}

// ----- live documents -------------------------------------------------------

pub(crate) struct LiveDoc {
    pub id: String,
    pub text: String,
    pub meta: String,
    pub tokens: Vec<String>,
}

pub(crate) fn dump_live(
    variant: Variant,
    fixes: bool,
    k1: f64,
    b: f64,
    epsilon: f64,
    delta: f64,
    tokenizer_tag: Option<&str>,
    docs: &[LiveDoc],
) -> PyResult<Vec<u8>> {
    let mut blob = Vec::new();
    blob.extend_from_slice(&(docs.len() as u64).to_le_bytes());
    for d in docs {
        push_str(&mut blob, &d.id);
        push_str(&mut blob, &d.text);
        push_str(&mut blob, &d.meta);
        blob.extend_from_slice(&(d.tokens.len() as u64).to_le_bytes());
        for t in &d.tokens {
            push_str(&mut blob, t);
        }
    }
    let mut pay = Pay::new();
    let docs_sec = pay.write_bytes(&blob);
    let tag = match tokenizer_tag {
        Some(s) => serde_json::from_str::<Value>(s).map_err(|e| PyValueError::new_err(format!("tokenizer tag: {e}")))?,
        None => Value::Null,
    };
    let k1h = hex_f64(k1);
    let bh = hex_f64(b);
    let eh = hex_f64(epsilon);
    let dh = hex_f64(delta);
    let variant_s = variant.as_str();
    let compat_s = variant.compat();
    wrap(
        |base| {
            let off = hex_u64(base + u64::from_str_radix(&docs_sec.0, 16).unwrap());
            json!({
                "format": 1,
                "kind": "live",
                "endian": "little",
                "variant": variant_s,
                "class": "LiveIndex",
                "compat": compat_s,
                "fixes": fixes,
                "k1": k1h,
                "b": bh,
                "epsilon": eh,
                "delta": dh,
                "tokenizer": tag,
                "n_docs": docs.len(),
                "sections": {"docs": [off, docs_sec.1]},
            })
        },
        &pay.buf,
    )
}

fn read_one_str(bytes: &[u8], i: &mut usize) -> PyResult<String> {
    if *i + 4 > bytes.len() {
        return Err(PyValueError::new_err("truncated live index"));
    }
    let n = u32::from_le_bytes(bytes[*i..*i + 4].try_into().unwrap()) as usize;
    *i += 4;
    if *i + n > bytes.len() {
        return Err(PyValueError::new_err("truncated live index string"));
    }
    let s = std::str::from_utf8(&bytes[*i..*i + n])
        .map_err(|_| PyValueError::new_err("live index string is not utf-8"))?
        .to_owned();
    *i += n;
    Ok(s)
}

pub(crate) fn load_live_docs(file: &[u8]) -> PyResult<(Value, Vec<LiveDoc>)> {
    let header = parse_header(file)?;
    if header.get("kind").and_then(|v| v.as_str()) != Some("live") {
        return Err(PyValueError::new_err("index file is not a live index"));
    }
    let sections = header
        .get("sections")
        .ok_or_else(|| PyValueError::new_err("missing sections"))?;
    let (off, len) = section_pair(sections, "docs")?;
    let bytes = need(file, off, len)?;
    if bytes.len() < 8 {
        return Err(PyValueError::new_err("truncated live index"));
    }
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    // Each document is at least one length prefix. A huge `n` must not allocate.
    if n > (bytes.len() - 8) as u64 {
        return Err(PyValueError::new_err("live index document count exceeds the section"));
    }
    let n = n as usize;
    let mut i = 8;
    let mut docs = Vec::new();
    for _ in 0..n {
        let id = read_one_str(bytes, &mut i)?;
        let text = read_one_str(bytes, &mut i)?;
        let meta = read_one_str(bytes, &mut i)?;
        if i + 8 > bytes.len() {
            return Err(PyValueError::new_err("truncated live index"));
        }
        let nt = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        if nt > (bytes.len() - (i + 8)) as u64 {
            return Err(PyValueError::new_err("live index token count exceeds the section"));
        }
        let nt = nt as usize;
        i += 8;
        let mut tokens = Vec::new();
        for _ in 0..nt {
            tokens.push(read_one_str(bytes, &mut i)?);
        }
        docs.push(LiveDoc { id, text, meta, tokens });
    }
    Ok((header, docs))
}
