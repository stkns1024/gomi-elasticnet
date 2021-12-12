pub struct Prox {
    threshold: f64,
    scale: f64,
}

impl Prox {
    pub fn new(l1: f64, l2: f64) -> Self {
        Self {
            threshold: l1,
            scale: 1.0 / (1.0 + l2),
        }
    }

    pub fn call(&self, x: f64) -> f64 {
        if x > self.threshold {
            (x - self.threshold) * self.scale
        } else if x < -self.threshold {
            (x + self.threshold) * self.scale
        } else {
            0.0
        }
    }
}
