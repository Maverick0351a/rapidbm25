//! Optional tokenizer. Compat classes never call this; their default behavior
//! is unchanged.
//!
//! Pipeline: Unicode word split (UAX #29), optional CJK overlapping bigrams,
//! optional Unicode lowercase, stopwords, optional Snowball stemmer
//! (`rust-stemmers`, MIT OR BSD-3-Clause). CJK bigrams are produced by this
//! crate; there is no dictionary segmenter.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyString, PyTuple};
use rayon::prelude::*;
use rust_stemmers::{Algorithm, Stemmer};
use rustc_hash::FxHashSet;
use unicode_segmentation::UnicodeSegmentation;

const ENGLISH_STOP: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "should", "now", "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "because", "as", "until", "where", "how", "why",
];

fn stemmer_of(name: &str) -> PyResult<Stemmer> {
    let algo = match name {
        "arabic" => Algorithm::Arabic,
        "danish" => Algorithm::Danish,
        "dutch" => Algorithm::Dutch,
        "english" | "en" => Algorithm::English,
        "finnish" => Algorithm::Finnish,
        "french" => Algorithm::French,
        "german" => Algorithm::German,
        "greek" => Algorithm::Greek,
        "hungarian" => Algorithm::Hungarian,
        "italian" => Algorithm::Italian,
        "norwegian" => Algorithm::Norwegian,
        "portuguese" => Algorithm::Portuguese,
        "romanian" => Algorithm::Romanian,
        "russian" => Algorithm::Russian,
        "spanish" => Algorithm::Spanish,
        "swedish" => Algorithm::Swedish,
        "tamil" => Algorithm::Tamil,
        "turkish" => Algorithm::Turkish,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown Snowball stemmer {name:?}; expected a language such as 'english'"
            )))
        }
    };
    Ok(Stemmer::create(algo))
}

fn is_cjk(ch: char) -> bool {
    let u = ch as u32;
    (0x3040..=0x30FF).contains(&u)
        || (0x3400..=0x4DBF).contains(&u)
        || (0x4E00..=0x9FFF).contains(&u)
        || (0xAC00..=0xD7AF).contains(&u)
        || (0xF900..=0xFAFF).contains(&u)
        || (0x20000..=0x2A6DF).contains(&u)
        || (0x2A700..=0x2B73F).contains(&u)
        // Halfwidth katakana (U+FF65..U+FF9F), including the halfwidth ideographic marks.
        || (0xFF65..=0xFF9F).contains(&u)
}

#[pyclass(frozen, module = "rapidbm25._native")]
pub struct Tokenizer {
    lowercase: bool,
    cjk: bool,
    stemmer_name: Option<String>,
    stemmer: Option<Stemmer>,
    /// `Some("english")` when the built-in list was requested, so pickle and
    /// `save` round-trip the preset instead of an expanded copy.
    stop_preset: Option<String>,
    stop_custom: Vec<String>,
    stop: FxHashSet<String>,
}

impl Tokenizer {
    fn tokenize_str(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        if self.cjk {
            let chars: Vec<char> = text.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                if is_cjk(chars[i]) {
                    let start = i;
                    i += 1;
                    while i < chars.len() && is_cjk(chars[i]) {
                        i += 1;
                    }
                    let run = &chars[start..i];
                    if run.len() == 1 {
                        self.push_token(&run[0].to_string(), &mut out);
                    } else {
                        for w in run.windows(2) {
                            let tok: String = w.iter().collect();
                            self.push_token(&tok, &mut out);
                        }
                    }
                } else {
                    let start = i;
                    i += 1;
                    while i < chars.len() && !is_cjk(chars[i]) {
                        i += 1;
                    }
                    let segment: String = chars[start..i].iter().collect();
                    self.push_words(&segment, &mut out);
                }
            }
        } else {
            self.push_words(text, &mut out);
        }
        out
    }

    fn push_words(&self, text: &str, out: &mut Vec<String>) {
        for w in text.unicode_words() {
            self.push_token(w, out);
        }
    }

    fn push_token(&self, raw: &str, out: &mut Vec<String>) {
        let mut tok = if self.lowercase { raw.to_lowercase() } else { raw.to_owned() };
        if tok.is_empty() || self.stop.contains(&tok) {
            return;
        }
        if let Some(stemmer) = &self.stemmer {
            tok = stemmer.stem(&tok).into_owned();
            if tok.is_empty() || self.stop.contains(&tok) {
                return;
            }
        }
        out.push(tok);
    }

    fn build_stop(lowercase: bool, preset: &Option<String>, custom: &[String]) -> FxHashSet<String> {
        let mut stop = FxHashSet::default();
        let mut add = |w: &str| {
            let folded = if lowercase { w.to_lowercase() } else { w.to_owned() };
            if !folded.is_empty() {
                stop.insert(folded);
            }
        };
        if preset.as_deref() == Some("english") {
            for w in ENGLISH_STOP {
                add(w);
            }
        }
        for w in custom {
            add(w);
        }
        stop
    }
}

#[pymethods]
impl Tokenizer {
    /// `stopwords` is `None`, `"english"`, or an iterable of strings.
    /// `stemmer` is a Snowball language name (`"english"` is Porter2) or `None`.
    /// `cjk=True` emits overlapping character bigrams for CJK / Kana / Hangul runs.
    #[new]
    #[pyo3(signature = (lowercase=true, stemmer=None, stopwords=None, cjk=false))]
    fn new(
        lowercase: bool,
        stemmer: Option<String>,
        stopwords: Option<Bound<'_, PyAny>>,
        cjk: bool,
    ) -> PyResult<Self> {
        let stemmer_name = stemmer;
        let stemmer = match &stemmer_name {
            Some(name) => Some(stemmer_of(name)?),
            None => None,
        };
        let (stop_preset, stop_custom) = match stopwords {
            None => (None, Vec::new()),
            Some(obj) => {
                if let Ok(s) = obj.cast::<PyString>() {
                    let name = s.to_str()?.to_owned();
                    if name != "english" {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "stopwords preset must be 'english' or a list of strings",
                        ));
                    }
                    (Some(name), Vec::new())
                } else {
                    let mut custom = Vec::new();
                    for item in obj.try_iter()? {
                        let item = item?;
                        let s = item
                            .cast::<PyString>()
                            .map_err(|_| PyTypeError::new_err("stopwords must be str"))?;
                        custom.push(s.to_str()?.to_owned());
                    }
                    (None, custom)
                }
            }
        };
        let stop = Self::build_stop(lowercase, &stop_preset, &stop_custom);
        Ok(Self {
            lowercase,
            cjk,
            stemmer_name,
            stemmer,
            stop_preset,
            stop_custom,
            stop,
        })
    }

    #[getter]
    fn lowercase(&self) -> bool {
        self.lowercase
    }
    #[getter]
    fn cjk(&self) -> bool {
        self.cjk
    }
    #[getter]
    fn stemmer(&self) -> Option<String> {
        self.stemmer_name.clone()
    }
    #[getter]
    fn stop_preset(&self) -> Option<String> {
        self.stop_preset.clone()
    }
    #[getter]
    fn stop_custom(&self) -> Vec<String> {
        self.stop_custom.clone()
    }

    fn __call__(&self, text: &str) -> Vec<String> {
        self.tokenize_str(text)
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenize_str(text)
    }

    /// Tokenize many strings. The GIL is released and the work is threaded.
    fn tokenize_many<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Bound<'py, PyList>> {
        let rows: Vec<Vec<String>> = py.detach(|| texts.par_iter().map(|t| self.tokenize_str(t)).collect());
        PyList::new(py, rows)
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let cls = py.get_type::<Self>();
        let stop: Bound<'py, PyAny> = if let Some(preset) = &self.stop_preset {
            PyString::new(py, preset).into_any()
        } else if self.stop_custom.is_empty() {
            py.None().into_bound(py)
        } else {
            PyList::new(py, self.stop_custom.clone())?.into_any()
        };
        let args = (self.lowercase, self.stemmer_name.clone(), stop, self.cjk).into_pyobject(py)?;
        Ok(PyTuple::new(py, [cls.into_any(), args.into_any()])?)
    }
}
