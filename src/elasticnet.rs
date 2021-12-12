mod params;

pub use crate::elasticnet::params::ElasticNetParams;
use crate::prox::Prox;
use crate::scaler::StandardScaler;
use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::prelude::*;

#[pyclass]
struct ElasticNet {
    scaler: StandardScaler,
    y_mean: f64,
    xy_cov: Array1<f64>,
    x_cov: Array2<f64>,
}

impl ElasticNet {
    fn n_features(&self) -> usize {
        self.xy_cov.len()
    }
}

#[pymethods]
impl ElasticNet {
    #[new]
    fn new(y: PyReadonlyArray1<f64>, x: PyReadonlyArray2<f64>) -> Self {
        let y = y.as_array();
        let x = x.as_array();

        let y_mean = y.mean().unwrap();
        let scaler = StandardScaler::new(x);

        let (xy_cov, x_cov) = {
            let x = scaler.standardize(x);
            let xt = x.t();
            let n = y.len() as f64;
            (xt.dot(&(&y - y_mean)) / n, xt.dot(&x) / n)
        };

        Self {
            scaler,
            y_mean,
            xy_cov,
            x_cov,
        }
    }

    #[args(l1 = "0.1", l2 = "0.1", max_iter = "1000", tolerance = "1e-4")]
    fn fit(
        &self,
        l1: f64,
        l2: f64,
        max_iter: i32,
        tolerance: f64,
        random_state: Option<u64>,
        model: Option<&ElasticNetParams>,
    ) -> ElasticNetParams {
        let mut eps = f64::INFINITY;
        let mut num_iter = 0;
        let mut indices: Vec<usize> = (0..self.n_features()).collect();
        let prox = Prox::new(l1, l2);

        let mut rng = match random_state {
            Some(state) => StdRng::seed_from_u64(state),
            None => StdRng::from_rng(thread_rng()).unwrap(),
        };

        let mut coef = match model {
            Some(model) => model.coef(),
            None => Array1::random_using(self.n_features(), StandardNormal, &mut rng),
        };

        while eps > tolerance {
            if num_iter == max_iter {
                break;
            }
            num_iter += 1;

            indices.shuffle(&mut rng);
            eps = 0.0;

            for &idx in indices.iter() {
                let grad = self.x_cov.row(idx).dot(&coef) - self.xy_cov[idx];
                let coef_idx = coef[idx];
                let new_coef_idx = prox.call(coef_idx - grad);
                eps += (new_coef_idx - coef_idx).abs();
                coef[idx] = new_coef_idx;
            }
        }

        ElasticNetParams::new(self.scaler.clone(), self.y_mean, coef)
    }
}

#[pymodule]
fn elasticnet(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ElasticNet>()?;
    m.add_class::<ElasticNetParams>()?;
    Ok(())
}
