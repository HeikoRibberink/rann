use crate::{Intermediate, Network, Scalar};

/// Chains two networks together, after eachother.
pub struct Chain<T, U> {
    /// The first part of the chain.
    pub first: T,
    /// The second part of the chain.
    pub second: U,
}

impl<T, U> Network for Chain<T, U>
where
    T: Network,
    U: Network<In = T::Out>,
{
    type In = T::In;

    type Out = U::Out;

    type Inter = ChainInter<T::Inter, U::Inter>;

    fn intermediate(&self, input: &Self::In) -> Self::Inter {
        // Evaluate the first layer...
        let first = self.first.intermediate(input);
        // ...and use its outputs to evaluate the second.
        let second = self.second.intermediate(first.output());
        ChainInter { first, second }
    }

    fn train_deriv(
        &mut self,
        inputs: &Self::In,
        intermediate: &Self::Inter,
        gradients: &Self::Out,
        learning_rate: Scalar,
    ) -> Self::In {
        // Train the second layer...
        let second = self.second.train_deriv(
            intermediate.first.output(),
            &intermediate.second,
            gradients,
            learning_rate,
        );
        // ...and use the resulting gradients to train the first network.
        let first = self
            .first
            .train_deriv(inputs, &intermediate.first, &second, learning_rate);
        // Output gradients are of first layer.
        first
    }
}

/// The intermediate values of an evaluation of a [`Chain`].
pub struct ChainInter<T, U> {
    /// The intermediate calculation of the first network.
    pub first: T,
    /// The intermediate calculation of the second network.
    pub second: U,
}

impl<T, U> Intermediate for ChainInter<T, U>
where
    T: Intermediate,
    U: Intermediate,
{
    type Out = U::Out;

    fn output(&self) -> &Self::Out {
        self.second.output()
    }

    fn into_output(self) -> Self::Out {
        self.second.into_output()
    }
}

