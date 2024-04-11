use arrayvec::ArrayVec;
use nalgebra::{Const, MatrixView, SMatrix};
use rann_traits::{deriv::Deriv, Intermediate, Network, Scalar};

/// A fully connected network layer.
pub struct Full<const NUM_IN: usize, const NUM_OUT: usize, A> {
    weights: SMatrix<Scalar, NUM_OUT, NUM_IN>,
    biases: [Scalar; NUM_OUT],
    act: A,
}

impl<const NUM_IN: usize, const NUM_OUT: usize, A> Network for Full<NUM_IN, NUM_OUT, A>
where
    A: Deriv<In = Scalar, Out = Scalar>,
{
    type In = [Scalar; NUM_IN];

    type Out = [Scalar; NUM_OUT];

    type Inter = FullInter<NUM_OUT>;

    fn intermediate(&self, input: &Self::In) -> Self::Inter {
        let mat = MatrixView::from_slice_generic(input, Const::<NUM_IN>, Const::<1>);
        // Multiply the matrices to find the weighted sums.
        let mut out = self.weights * mat;
        // Apply bias to the weighted sums.
        for (sum, bias) in out.iter_mut().zip(self.biases) {
            *sum += bias;
        }
        // Clone the weighted sums to store them.
        let sums: [Scalar; NUM_OUT] = out.data.0[0].clone();
        // Apply the activation function to the weighted sums.
        for sum in out.iter_mut() {
            *sum = self.act.call(&sum);
        }
        (sums, out.data.0[0])
    }

    fn train_deriv(
        &mut self,
        input: &Self::In,
        intermediate: &Self::Inter,
        gradients: &Self::Out,
        learning_rate: Scalar,
    ) -> Self::In {
        // Calculate the gradients over the activation
        let grad: ArrayVec<Scalar, NUM_OUT> = gradients
            .iter()
            .zip(intermediate.0.iter())
            .map(|(gr, sum)| gr * self.act.deriv(sum)).collect();
        // Update the biases
        for (bias, grad) in self.biases.iter_mut().zip(grad.iter()) {
            *bias += grad * learning_rate;
        }
        // Calculate the gradients over each weight and update it correspondingly.
        for (mut weights, input) in self.weights.column_iter_mut().zip(input.iter()) {
            for (w, grad) in weights.iter_mut().zip(grad.iter()) {
                *w += input * grad * learning_rate;
            }
        }
        // Amount of columns = NUM_IN, size_grad = NUM_OUT
        let out: ArrayVec<f32, NUM_IN> = self.weights.column_iter().map(|row| {
            let mut sum = 0.0;
            for (w, g) in row.iter().zip(grad.iter()) {
                sum += w * g;
            }
            sum
        }).collect();
        out.into_inner().expect("Capacity of ArrayVec should equal NUM_OUT.")
    }
}

/// The intermediate calculations for an evaluation of [`Full`].
/// Of the form (weighted sums, outputs).
pub type FullInter<const NUM_OUT: usize> = ([Scalar; NUM_OUT], [Scalar; NUM_OUT]);
