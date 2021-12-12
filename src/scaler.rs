use ndarray::{Array1, Array2, ArrayView2, Zip};

fn mean_var(x: ArrayView2<f64>) -> (Array1<f64>, Array1<f64>) {
    let ncols = x.ncols();
    let mut mean = Array1::zeros(ncols);
    let mut var = Array1::zeros(ncols);
    let mut count: usize = 0;
    for row in x.rows() {
        count += 1;
        let delta = &row - &mean;
        mean += &(&delta / count as f64);
        var += &(delta * (&row - &mean));
    }

    var /= count as f64;

    (mean, var)
}

#[derive(Clone, Debug)]
pub struct StandardScaler {
    offset: Array1<f64>,
    scale: Array1<f64>,
}

impl StandardScaler {
    pub fn new(x: ArrayView2<f64>) -> Self {
        let (offset, mut scale) = mean_var(x);
        scale.mapv_inplace(|x| if x != 0.0 { 1.0 / x.sqrt() } else { 1.0 });
        Self { offset, scale }
    }

    pub fn standardize(&self, x: ArrayView2<f64>) -> Array2<f64> {
        Zip::from(x)
            .and_broadcast(&self.offset)
            .and_broadcast(&self.scale)
            .map_collect(|&x, &offset, &scale| (x - offset) * scale)
    }
}
