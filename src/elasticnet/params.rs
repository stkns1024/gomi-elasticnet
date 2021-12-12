use crate::scaler::StandardScaler;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct ElasticNetParams {
    scaler: StandardScaler,
    intercept: f64,
    coef: Array1<f64>,
}

impl ElasticNetParams {
    pub fn new(scaler: StandardScaler, intercept: f64, coef: Array1<f64>) -> Self {
        ElasticNetParams {
            scaler,
            intercept,
            coef,
        }
    }

    pub fn coef(&self) -> Array1<f64> {
        self.coef.clone()
    }
}

#[pymethods]
impl ElasticNetParams {
    fn predict(self_: PyRef<Self>, x: PyReadonlyArray2<f64>) -> Py<PyArray1<f64>> {
        let x = x.as_array();

        let x = self_.scaler.standardize(x);
        let mut x = x.dot(&self_.coef);
        x += self_.intercept;

        let py = self_.py();
        x.into_pyarray(py).to_owned()
    }
}
