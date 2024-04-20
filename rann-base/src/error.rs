use arrayvec::ArrayVec;
use rann_traits::{Network, Scalar};

pub struct SquareError<const N: usize> {
    pub expected: [Scalar; N],
}

impl<const N: usize> Network for SquareError<N> {
    type In = [Scalar; N];

    type Out = [Scalar; 1];

    type Inter = [f32; 1];

    fn intermediate(&self, inputs: &Self::In) -> Self::Inter {
        [inputs
            .iter()
            .zip(self.expected)
            .map(|(i, e)| (i - e) * (i - e))
            .sum()]
    }

    fn train_deriv(
        &mut self,
        // The previous inputs to the network.
        inputs: &Self::In,
        // The intermediate results of the calculation associated to the inputs.
        _intermediate: &Self::Inter,
        // The gradients of the output relative to the error.
        _gradients: &Self::Out,
        // The learning rate.
        _learning_rate: Scalar,
    ) -> Self::In {
        inputs
            .iter()
            .zip(self.expected)
            .map(|(i, e)| 2.0 * (i - e))
            .collect::<ArrayVec<Scalar, N>>()
            .into_inner()
            .expect("Capacity of ArrayVec should equal N.")
    }
}

pub struct SumError<const N: usize> {
    pub expected: [Scalar; N],
}

impl<const N: usize> Network for SumError<N> {
    type In = [Scalar; N];

    type Out = [Scalar; 1];

    type Inter = [f32; 1];

    fn intermediate(&self, inputs: &Self::In) -> Self::Inter {
        [inputs
            .iter()
            .zip(self.expected)
            .map(|(i, e)| (i - e).abs())
            .sum()]
    }

    fn train_deriv(
        &mut self,
        // The previous inputs to the network.
        inputs: &Self::In,
        // The intermediate results of the calculation associated to the inputs.
        _intermediate: &Self::Inter,
        // The gradients of the output relative to the error.
        _gradients: &Self::Out,
        // The learning rate.
        _learning_rate: Scalar,
    ) -> Self::In {
        inputs
            .iter()
            .zip(self.expected)
            .map(|(i, e)| i - e)
            .collect::<ArrayVec<Scalar, N>>()
            .into_inner()
            .expect("Capacity of ArrayVec should equal N.")
    }
}
