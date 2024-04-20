use rann_traits::Scalar;

#[derive(Clone, Copy, Debug)]
pub struct Random;

impl Into<(fn(usize, usize) -> Scalar, fn(usize) -> Scalar)> for Random {
    fn into(self) -> (fn(usize, usize) -> Scalar, fn(usize) -> Scalar) {
        (random_weights, random_biases)
    }
}

pub fn random_weights(_: usize, _: usize) -> f32 {
    fastrand::f32() * 4.0 - 2.0
}

pub fn random_biases(_: usize) -> f32 {
    fastrand::f32() * 4.0 - 2.0
}
