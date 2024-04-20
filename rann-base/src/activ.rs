use rann_traits::deriv::Deriv;

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
