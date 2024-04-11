use crate::{Intermediate, Network, Scalar};

/// Links to networks together, after eachother.
pub struct Chain<T, U> {
    pub(super) first: T,
    pub(super) second: U,
}

impl<T, U> Network for Chain<T, U>
where
    T: Network,
    U: Network<In = T::Out>,
{
    type In = T::In;

    type Out = U::Out;

    type Inter = (T::Inter, U::Inter);

    fn intermediate(&self, input: &Self::In) -> Self::Inter {
        let a = self.first.intermediate(input);
        let b = self.second.intermediate(a.output());
        (a, b)
    }

    fn train_deriv(
        &mut self,
        inputs: &Self::In,
        intermediate: &Self::Inter,
        gradients: &Self::Out,
        learning_rate: Scalar,
    ) -> Self::In {
        let second = self.second.train_deriv(
            intermediate.0.output(),
            &intermediate.1,
            gradients,
            learning_rate,
        );
        let first = self
            .first
            .train_deriv(inputs, &intermediate.0, &second, learning_rate);
        first
    }
}
