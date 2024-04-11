use std::ops::Index;

use crate::Scalar;

/// A one-dimensional pure function with its derivative.
pub trait Deriv {
    /// The input type.
    type In;
    /// The output type.
    type Out;
    /// Normal function.
    fn call(&self, x: &Self::In) -> Self::Out;
    /// Derivative of function.
    fn deriv(&self, x: &Self::In) -> Self::Out;
}

// Any tuple of pure functions can also be used as a Derivative.
impl<F, D> Deriv for (F, D)
where
    F: Fn(Scalar) -> Scalar,
    D: Fn(Scalar) -> Scalar,
{
    type In = Scalar;
    type Out = Scalar;

    fn call(&self, &x: &Self::In) -> Self::Out {
        self.0(x)
    }

    fn deriv(&self, &x: &Self::In) -> Self::Out {
        self.1(x)
    }

}

/// A multi-dimensional pure function with its derivatives.
pub trait NDeriv {
    /// The input type.
    type In: Index<usize>;
    /// The output type.
    type Out;
    /// Normal function.
    fn call(&self, x: &Self::In) -> Self::Out;
    /// Partial derivative of function, where p is the index of the input to derive on.
    fn deriv(&self, x: &Self::In, p: usize) -> Self::Out;
}


// Any one-dimensional derivative is also a multi-dimensional derivative.
impl<T> NDeriv for T where T: Deriv {
    type In = [T::In; 1];
    type Out = T::Out;

    fn call(&self, x: &Self::In) -> Self::Out {
        self.call(&x[0])
    }

    fn deriv(&self, x: &Self::In, _: usize) -> Self::Out {
        self.deriv(&x[0])
    }
}
//
// /// Indicates that this function can be used as an error function.
// trait Error: MultiDerivative<In = [Scalar], Out = Scalar> {}
//
// /// Indicates that this function can be used as an activation function.
// trait Activation: Derivative<In = Scalar, Out = Scalar> {}
