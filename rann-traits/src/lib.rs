use compose::Chain;

pub mod compose;
pub mod deriv;

/// The default scalar type
pub type Scalar = f32;

/// Trait implemented by neural networks.
pub trait Network {
    /// Type for the network's inputs and derivatives.
    type In: AsRef<[Scalar]>;
    /// Type for the network's outputs and derivatives.
    type Out: AsRef<[Scalar]>;
    /// Type for storing intermediate calculations.
    type Inter: Intermediate<Out = Self::Out>;

    /// Evaluate the network and return the intermediate calculations.
    fn intermediate(&self, inputs: &Self::In) -> Self::Inter;

    /// Train the network using a previous evaluation and gradients, and return input gradients.
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

    fn eval(&self, input: &Self::In) -> Self::Out {
        self.intermediate(input).into_output()
    }

    fn chain<U>(self, next: U) -> Chain<Self, U>
    where
        Self: Sized,
        U: Network,
        U::In: AsRef<Self::Out>,
    {
        Chain {
            first: self,
            second: next,
        }
    }
}

pub trait Intermediate {
    /// Type for the network's outputs and derivatives.
    type Out: AsRef<[Scalar]>;

    /// Borrows the output of the network.
    fn output(&self) -> &Self::Out;

    /// Returns the output of the network.
    fn into_output(self) -> Self::Out;
}

impl<T, U> Intermediate for (T, U)
where
    T: Intermediate,
    U: Intermediate,
{
    type Out = U::Out;

    fn output(&self) -> &Self::Out {
        self.1.output()
    }

    fn into_output(self) -> Self::Out {
        self.1.into_output()
    }
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
