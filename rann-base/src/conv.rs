use nalgebra::SMatrix;
use rann_traits::{deriv::Deriv, Network, Scalar};

pub struct Convolutional<const NUM_IN: usize, const NUM_OUT: usize, const ND: usize, A> {
    weights: SMatrix<Scalar, 1, NUM_IN>,
    bias: Scalar,
    act: A,
    dimensions: [usize; ND],
}

impl<const NUM_IN: usize, const NUM_OUT: usize, const ND: usize, A> Network
    for Convolutional<NUM_IN, NUM_OUT, ND, A>
where
    A: Deriv<In = Scalar, Out = Scalar>,
{
    type In = [Scalar; NUM_IN];
    type Out = [Scalar; NUM_OUT];
    type Inter = [Scalar; NUM_OUT];

    fn intermediate(&self, inputs: &Self::In) -> Self::Inter {
        todo!()
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
        todo!()
    }
}
