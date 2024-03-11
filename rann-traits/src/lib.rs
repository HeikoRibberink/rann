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
    type Inter: Intermediate<In = Self::In, Out = Self::Out>;

    /// Evaluate the network and return the intermediate calculations.
    fn intermediate(&self, inputs: Self::In) -> Self::Inter;

    /// Train the network using a previous evaluation and gradients.
    fn train_deriv(
        &mut self,
        intermediate: Self::Inter,
        gradients: Self::Out,
        learning_rate: f32,
    ) -> Self::In;

    
    fn chain<U>(self, next: U) -> Chain<Self, U>
    where
        Self: Sized,
        U: Network,
        U::In: AsRef<Self::Out>,
    {
        Chain {first: self, second: next}
    }
}

pub trait Intermediate {
    /// Type for the network's inputs and derivatives.
    type In: AsRef<[Scalar]>;
    /// Type for the network's outputs and derivatives.
    type Out: AsRef<[Scalar]>;

    /// Returns the inputs to the network.
    fn input(&self) -> Self::In;
    /// Returns the output of the network.
    fn output(&self) -> Self::Out;
}

impl<T, U> Intermediate for (T, U)
where
    T: Intermediate,
    U: Intermediate,
{
    type In = T::In;

    type Out = U::Out;

    fn input(&self) -> Self::In {
        self.0.input()
    }

    fn output(&self) -> Self::Out {
        self.1.output()
    }
}
