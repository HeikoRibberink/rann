use rann_traits::{
    deriv::{Deriv, NDeriv},
    Scalar,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::net::GenReq;

/// Leaky Rectified Linear unit activation function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeakyRelu(pub f32);

impl Deriv for LeakyRelu {
    type In = f32;
    type Out = f32;
    fn call(&self, &x: &f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            self.0 * x
        }
    }

    fn deriv(&self, &x: &f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            self.0
        }
    }
}

/// Hypertangent activation function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HTan;

impl Deriv for HTan {
    type In = f32;

    type Out = f32;

    fn call(&self, x: &Self::In) -> Self::Out {
        x.tanh()
    }

    fn deriv(&self, x: &Self::In) -> Self::Out {
        (4.0 * x.cosh().powi(2)) / ((2.0 * x).cosh() + 1.0).powi(2)
    }
}

/// Logistic activation function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Logistic;

impl Deriv for Logistic {
    type In = f32;

    type Out = f32;

    fn call(&self, x: &Self::In) -> Self::Out {
        1.0 / (1.0 + (-x).exp())
    }

    fn deriv(&self, x: &Self::In) -> Self::Out {
        let a = <Self as Deriv>::call(self, x);
        a * (1.0 - a)
    }
}

/// Sums the square of the difference between the expected values and the input values.
#[derive(Clone, Debug)]
pub struct SquareError {
    pub expected: Vec<f32>,
}

impl NDeriv for SquareError {
    type In = Vec<f32>;

    type Out = f32;

    fn call(&self, x: &Self::In) -> Self::Out {
        x.par_iter()
            .zip(self.expected.par_iter())
            .map(|(x, e)| (x - e) * (x - e))
            .sum()
    }

    fn deriv(&self, x: &Self::In, p: usize) -> Self::Out {
        2.0 * (x[p] - self.expected[p])
    }
}

/// Error function that sums the absolute differences between the expected and actual output.
#[derive(Clone, Debug)]
pub struct Sum {
    pub expected: Vec<f32>,
}

impl NDeriv for Sum {
    type In = Vec<f32>;

    type Out = f32;

    fn call(&self, x: &Self::In) -> Self::Out {
        x.par_iter()
            .zip(self.expected.par_iter())
            .map(|(x, e)| (x - e).abs())
            .sum()
    }

    fn deriv(&self, x: &Self::In, p: usize) -> Self::Out {
        x[p] - self.expected[p]
    }
}

/// Generator that generates a random f32 in the range [-2.0, 2.0].
pub fn random_unit(_: GenReq) -> f32 {
    fastrand::f32() * 4.0 - 2.0
}

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
