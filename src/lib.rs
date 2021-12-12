mod elasticnet;
mod prox;
mod scaler;

use elasticnet::*;
use pyo3::prelude::*;
use pyo3::wrap_pymodule;

#[pymodule]
fn gmelasticnet(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(elasticnet))?;
    Ok(())
}
