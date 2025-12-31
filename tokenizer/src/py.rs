use crate::tokenizer::Tokenizer;
use pyo3::exceptions::{PyRuntimeError };
use pyo3::prelude::*;
use pyo3::types::PyDict;

// Convert anyhow-ish erros into Python exceptions.
fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[pyclass(name = "RustBPETokenizer")]
pub struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    pub fn new(vocab_path: String, merges_path: String) -> PyResult<Self> {
        let inner = Tokenizer::from_files(&vocab_path, &merges_path).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// encode(text, add_bos=True, add_eos=True) -> List[int]
    ///
    /// We use Option<bool> so Python can omit them and we default to true.
    #[pyo3(signature = (text, add_bos=true, add_eos=true))]
    pub fn encode(
        &self,
        py: Python<'_>,
        text: &str,
        add_bos: Option<bool>,
        add_eos: Option<bool>,
    ) -> PyResult<Vec<u32>> {
        let add_bos = add_bos.unwrap_or(true);
        let add_eos = add_eos.unwrap_or(true);

        // Release GIL during CPU work.
        py.allow_threads(|| self.inner.encode(text, add_bos, add_eos))
            .map_err(to_py_err)
    }

    /// decode(iids, skip_special=True) -> str
    #[pyo3(signature = (ids, skip_special=true))]
    pub fn decode(&self, py: Python<'_>, ids: Vec<u32>, skip_special: Option<bool>) -> PyResult<String> {
        let skip_special = skip_special.unwrap_or(true);

        // decode deosn't currently return Result, so just allow_threads for consistenc
        let s = py.allow_threads(|| self.inner.decode(&ids, skip_special));
        Ok(s)
    }

    //// vocab_size() -> int
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// special_ids() -> dict
    pub fn special_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("pad", self.inner.pad_id)?;
        d.set_item("bos", self.inner.bos_id)?;
        d.set_item("eos", self.inner.eos_id)?;
        d.set_item("unk", self.inner.unk_id)?;
        Ok(d)
    }
}


/// Python module definition
/// THe module name heres matches waht we build/import
#[pymodule]
pub fn tokenizer(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}

