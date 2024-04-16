use arrayvec::ArrayVec;
use rann_traits::{Intermediate, Network, Scalar};

pub struct SquareError<const N: usize> {
    pub expected: [Scalar; N],
}

impl<const N: usize> Network for SquareError<N> {
    type In = [Scalar; N];

    type Out = Scalar;

    type Inter = f32;

    fn intermediate(&self, inputs: &Self::In) -> Self::Inter {
        inputs
            .iter()
            .zip(self.expected)
            .map(|(i, e)| (i - e) * (i - e))
            .sum()
    }

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
    ) -> Self::In {
        inputs
            .iter()
            .zip(self.expected)
            .map(|(i, e)| 2.0 * (i - e))
            .collect::<ArrayVec<Scalar>>()
            .into_inner()
            .expect("Capacity of ArrayVec should equal N.")
    }
}
