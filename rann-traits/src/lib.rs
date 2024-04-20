/*!
# Rann-traits

Rann-traits contains all the different traits for the RANN ecosystem,
enabling you to compose neural networks and build generic, reusable components
for your machine learning applications.

At the center of RANN is the [`Network`] trait,

*/

pub mod compose;
pub mod deriv;

use compose::{Chain, Zip};
use num_traits::One;

/// The default scalar type
pub type Scalar = f32;

/// Trait implemented by neural networks.
pub trait Network {
    /// Type for the network's inputs and derivatives.
    type In;
    /// Type for the network's outputs and derivatives.
    type Out;
    /// Type for storing intermediate calculations.
    type Inter: Intermediate<Out = Self::Out>;

    /// Evaluate the network and return the intermediate calculations.
    fn intermediate(&self, inputs: &Self::In) -> Self::Inter;

    /// Train the network using a previous evaluation, the associated inputs, and gradients from
    /// a following network, and return the gradients over the inputs.
    fn train_deriv(
        &mut self,
        // The previous inputs to the network.
        inputs: &Self::In,
        // The intermediate results of the calculation associated to the inputs.
        intermediate: &Self::Inter,
        // The gradients of the output relative to the error.
        gradients: &Self::Out,
        // The learning rate.
        learning_rate: Scalar,
    ) -> Self::In;

    /// Evaluate the network and return the outputs.
    ///
    /// # Implementation note
    /// The default implementation evaluates the network using [`intermediate`] and discards all
    /// intermediate values but the output of the network. With some networks, it might be more
    /// efficient to override this behaviour.
    fn eval(&self, inputs: &Self::In) -> Self::Out {
        self.intermediate(inputs).into_output()
    }

    /// Trains the network using a previous evaluation and the associated inputs. 
    /// 
    /// 
    fn train<T, const NUM_IN: usize>(
        &mut self,
        inputs: &Self::In,
        intermediate: &Self::Inter,
        learning_rate: Scalar,
    ) where
        for<'a> &'a [T; NUM_IN]: Into<&'a Self::Out>,
        T: One + Copy,
    {
        self.train_deriv(
            inputs,
            intermediate,
            (&[T::one(); NUM_IN]).into(),
            learning_rate,
        );
    }

    /// Chains `self` and `next` together, after eachother.
    /// That is, `next` is connected to the output of this network.
    fn chain<U>(self, next: U) -> Chain<Self, U>
    where
        Self: Sized,
        U: Network<In = Self::Out>,
    {
        Chain {
            first: self,
            second: next,
        }
    }

    /// Zips `self` and `other` together into one network, in parallel, combining their outputs
    /// into one using `zipper`.
    ///`unzipper` must do exactly the reverse of `Z`: take the combined outputs of the networks and pull
    ///them apart.
    fn zip<U, C, Z, UnZ>(self, other: U, zipper: impl Into<(Z, UnZ)>) -> Zip<Self, U, Z, UnZ>
    where
        Self: Sized,
        U: Network,
        Z: Fn(&Self::Out, &U::Out) -> C,
        UnZ: for<'a> Fn(&'a C) -> (&'a Self::Out, &'a U::Out),
    {
        let (zipper, unzipper) = zipper.into();
        Zip {
            top: self,
            bot: other,
            zipper,
            unzipper,
        }
    }
}

/// Trait for types that represent the intermediate values of a network evaluation.
pub trait Intermediate {
    /// Type for the network's outputs and derivatives.
    type Out;

    /// Borrows the output of the network.
    fn output(&self) -> &Self::Out;

    /// Returns the output of the network.
    fn into_output(self) -> Self::Out;
}

impl<const NUM: usize> Intermediate for [Scalar; NUM] {
    type Out = [Scalar; NUM];

    fn output(&self) -> &Self::Out {
        self
    }

    fn into_output(self) -> Self::Out {
        self
    }
}

// Implemented mostly for error functions, as they simply reduce an array of scalars to a single
// scalar.
impl Intermediate for Scalar {
    type Out = Scalar;

    fn output(&self) -> &Self::Out {
        self
    }

    fn into_output(self) -> Self::Out {
        self
    }
}
