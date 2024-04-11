use std::fmt::Display;
use std::iter::repeat;

use nalgebra::{DMatrix, DVector, Dyn, Matrix, U1};
use rann_traits::deriv::{Deriv, NDeriv};
use thiserror::Error;

pub enum GenReq {
    /// Indicates that the generator should return a weight value.
    Weight {
        /// Index of the left layer in the range [0, n-2] where n is the number of layers in the
        /// network.
        l: usize,
        /// Index of the left node.
        i_l: usize,
        /// Index of the right node.
        i_n: usize,
    },
    /// Indicates that the generator should return a bias value.
    Bias {
        /// Index of the layer the node is in, in the range [1, n-1] where n is the number of
        /// layers in the network.
        l: usize,
        /// Index of the node.
        i_l: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct NNetwork {
    pub(crate) size: Vec<usize>,
    /// The weights between each layer. [0, n-1]
    pub(crate) weights: Vec<DMatrix<f32>>,
    /// The biases of each node in each layer, starting from the first hidden layer. [1, n]
    pub(crate) biases: Vec<Vec<f32>>,
}

impl NNetwork {
    // Note: #rows is the size of the next layer, #cols is the size of this layer.
    //       i denotes the element in the output vector, j the element in the input vector.

    /// Creates a new fully-connected neural network with the given size and generator function.
    /// The network should have at least an input and output layer, and no layer may be empty.
    pub fn new(size: &[usize], gen: impl Fn(GenReq) -> f32) -> Result<Self, Error> {
        use crate::net;
        // Do size checks
        if size.len() < 2 {
            Err(net::Error::WrongSize)?
        }
        for l in 0..size.len() {
            if size[l] < 1 {
                Err(net::Error::WrongLayerSize(l))?
            }
        }
        // Generate weights
        let mut weights = Vec::with_capacity(size.len() - 1);
        for l in 0..(size.len() - 1) {
            weights.push(DMatrix::from_fn(size[l + 1], size[l], |i, j| {
                gen(GenReq::Weight { l, i_l: j, i_n: i })
            }));
        }
        // Generate biases.
        let mut biases = Vec::with_capacity(size.len() - 1);
        for l in 1..size.len() {
            let mut layer = Vec::with_capacity(size[l]);
            for i in 0..size[l] {
                layer.push(gen(GenReq::Bias { l, i_l: i }))
            }
            biases.push(layer);
        }
        let size = size.into();
        Ok(Self {
            size,
            weights,
            biases,
        })
    }

    /// Convenience method to get a bias.
    pub fn bias(&self, layer: usize, index: usize) -> Option<f32> {
        if layer < 1 {
            None
        } else {
            self.biases
                .get(layer - 1)
                .and_then(|l| l.get(index).copied())
        }
    }

    /// Convenience method to get a weight.
    pub fn weight(&self, layer: usize, i_cur: usize, i_next: usize) -> Option<f32> {
        self.weights
            .get(layer)
            .and_then(|m| m.get((i_next, i_cur)).copied())
    }

    /// Evaluates the network and stores intermediate values.
    fn eval_inter(
        &self,
        inputs: Vec<f32>,
        activation: &impl Deriv<In = f32, Out = f32>,
        mut inter_sum: impl FnMut(DVector<f32>),
        mut inter_activs: impl FnMut(DVector<f32>),
    ) -> Vec<f32> {
        let mut activ = Matrix::from_vec_generic(Dyn(self.size[0]), U1, inputs);
        // i specifies from which layer the input is taken.
        for i in 0..(self.size.len() - 1) {
            // Calculate with weights
            let weighted_sum = &self.weights[i] * &activ;
            inter_activs(activ);
            activ = weighted_sum.clone();
            inter_sum(weighted_sum);

            // Calculate with biases and activation
            activ
                .iter_mut()
                .enumerate()
                .for_each(|(n_i, m)| *m = activation.call(&(*m + self.biases[i][n_i])));
        }
        activ.data.into()
    }

    /// Evaluates the nnetwork and returns intermediate computations as a tuple (activations,
    /// weighted sums). The inputs of the nnetwork are the first entry in activations, and the
    /// outputs the last.
    pub fn eval_intermediate(
        &self,
        inputs: Vec<f32>,
        activation: &impl Deriv<In = f32, Out = f32>,
    ) -> Intermediate {
        let mut sums: Vec<Vec<f32>> = Vec::with_capacity(self.size.len() - 1);
        let mut activs: Vec<Vec<f32>> = Vec::with_capacity(self.size.len());
        let res = self.eval_inter(
            inputs,
            activation,
            |vec| sums.push(vec.data.into()),
            |vec| activs.push(vec.data.into()),
        );
        activs.push(res);
        Intermediate { activs, sums }
    }

    /// Evaluates the network and outputs the activations of the last layer.
    pub fn eval(
        &self,
        inputs: Vec<f32>,
        activation: &impl Deriv<In = f32, Out = f32>,
    ) -> Vec<f32> {
        self.eval_inter(inputs, activation, |_| {}, |_| {})
    }

    /// Runs a backpropagation cycle on the network in-place, using the given intermediate
    /// calculations, activation function, error function and learning rate.
    ///
    /// Returns the error and the activation derivatives at the input layer.
    pub fn backprop_deriv(
        &mut self,
        intermediate: Intermediate,
        activation: &(impl Deriv<In = f32, Out = f32> + Sync),
        error: &(impl NDeriv<In = Vec<f32>, Out = f32> + Sync),
        learning_rate: f32,
    ) -> (f32, Vec<f32>) {
        let outputs = intermediate
            .activs
            .last()
            .expect("the intermediate values should have a layer with output values");
        let output_size = self
            .size
            .last()
            .copied()
            .expect("the network size should have a last layer");
        let out = error.call(outputs);

        // Calculate error derivatives
        let derivs: Vec<f32> = repeat(outputs)
            .enumerate()
            .take(output_size)
            // Apply error derivatives
            .map(|(i, x)| error.deriv(x, i))
            .collect();
        let mut derivs = Matrix::from_vec_generic(U1, Dyn(output_size), derivs);

        // Loop over the weights and biases over each layers in reverse order.
        for (l_prev, (weights, biases)) in self
            .weights
            .iter_mut()
            .zip(self.biases.iter_mut())
            .enumerate()
            .rev()
        {
            // Apply activation
            for (deriv, sum) in derivs.iter_mut().zip(intermediate.sums[l_prev].iter()) {
                *deriv *= activation.deriv(sum);
            }
            // Update biases
            for (bias, deriv) in biases.iter_mut().zip(derivs.iter()) {
                *bias -= deriv * learning_rate;
            }
            // Update weights
            let new_derivs = &derivs * &*weights;
            for (mut row, deriv) in weights.row_iter_mut().zip(derivs.iter()) {
                for (item, act) in row.iter_mut().zip(intermediate.activs[l_prev].iter()) {
                    *item -= deriv * act * learning_rate;
                }
            }
            // Multiply the current derivatives by the weights to get derivatives for prev layer
            derivs = new_derivs;
        }

        (out, derivs.data.into())
    }

    /// Runs a backpropagation cycle on the network in-place, using the given intermediate
    /// calculations, activation function, error function and learning rate.
    ///
    /// Returns the error.
    pub fn backprop(
        &mut self,
        intermediate: Intermediate,
        activation: &(impl Deriv<In = f32, Out = f32> + Sync),
        error: &(impl NDeriv<In = Vec<f32>, Out = f32> + Sync),
        learning_rate: f32,
    ) -> f32 {
        self.backprop_deriv(intermediate, activation, error, learning_rate).0
    }
}

impl Display for NNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (weights, biases) in self.weights.iter().zip(self.biases.iter()) {
            for row in weights.row_iter() {
                f.write_str("|")?;
                for w in row.iter() {
                    if *w < 0.0 {
                        f.write_fmt(format_args!(" {w:.2}"))?;
                    } else {
                        f.write_fmt(format_args!(" {w:.3}"))?;
                    }
                }
                f.write_str(" |\n")?;
            }
            f.write_str("{")?;
            for b in biases.iter() {
                if *b < 0.0 {
                    f.write_fmt(format_args!(" {b:.2}"))?;
                } else {
                    f.write_fmt(format_args!(" {b:.3}"))?;
                }
            }
            f.write_str(" }\n")?;
        }
        Ok(())
    }
}

/// Intermediate values of an evaluation of the nnetwork.
#[derive(Clone, Debug)]
pub struct Intermediate {
    /// Activation of each node, starting from the input layer and including the output layer.
    pub activs: Vec<Vec<f32>>,
    /// Weighted sums of each node, starting from the first hidden layer.
    pub sums: Vec<Vec<f32>>,
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum Error {
    #[error("the nnetwork should have 2 or more layers")]
    WrongSize,
    #[error("layer {0} should have 1 or more nodes")]
    WrongLayerSize(usize),
    #[error("the number of inputs is incorrect")]
    WrongInputSize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        common::{random_unit, LeakyRelu},
        net,
    };

    // Tests the checks for number of layers.
    #[test]
    fn size() {
        assert_eq!(NNetwork::new(&[], random_unit), Err(net::Error::WrongSize));
        assert_eq!(NNetwork::new(&[1], random_unit), Err(net::Error::WrongSize));
        assert!(NNetwork::new(&[1, 1], random_unit).is_ok());
    }

    // Tests the checks for layer sizes.
    #[test]
    fn layer_size() {
        assert_eq!(
            NNetwork::new(&[1, 0, 1], random_unit),
            Err(net::Error::WrongLayerSize(1))
        );
        assert_eq!(
            NNetwork::new(&[100, 1, 100, 100, 0, 1], random_unit),
            Err(net::Error::WrongLayerSize(4))
        );
        assert!(NNetwork::new(&[100, 1, 100, 100, 1, 1], random_unit).is_ok());
    }

    // Tests the sizes of the weight matrices.
    #[test]
    fn matrix_size() {
        let net = NNetwork::new(&[3, 1, 5], |_| 0.0).unwrap();
        assert_eq!(net.weights[0].shape(), (1, 3));
        assert_eq!(net.weights[1].shape(), (5, 1));
    }

    // Tests a custom generator, and the weight and bias helper functions.
    #[test]
    fn special_generator() {
        fn gen(req: GenReq) -> f32 {
            match req {
                GenReq::Bias { l, i_l } if l == 1 && i_l == 3 => 3.0,
                GenReq::Weight { l, i_l, i_n } if l == 0 && i_l == 0 && i_n == 1 => 1.0,
                _ => 2.0,
            }
        }
        let net = NNetwork::new(&[5, 5, 5], gen).unwrap();
        assert_eq!(net.bias(1, 3), Some(3.0));
        assert_eq!(net.bias(0, 0), None);
        for (l, layer) in net.biases.iter().enumerate() {
            for (i_l, b) in layer.iter().enumerate() {
                if l == (1 - 1) && i_l == 3 {
                    assert_eq!(*b, 3.0)
                } else {
                    assert_eq!(*b, 2.0, "Wrong bias on node {i_l} in layer {l}.")
                }
            }
        }
        assert_eq!(net.weight(0, 0, 1), Some(1.0));
        assert_eq!(net.weight(2, 0, 0), None);
    }

    // Tests if values are passed through the network correctly.
    #[test]
    fn eval_simple() {
        fn gen(req: GenReq) -> f32 {
            match req {
                // GenReq::Bias { l, i_l } if i_l == 0 => 0.0,
                GenReq::Weight {
                    l: _,
                    i_l: 0,
                    i_n: 0,
                } => 1.0,
                _ => 0.0,
            }
        }
        let net = NNetwork::new(&[5, 3, 5], gen).unwrap();
        let inputs = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let out = net.eval(inputs, &LeakyRelu(0.01));
        assert_eq!(&out, &[1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    // Tests a more complex evaluation of the network.
    #[test]
    fn eval_complex() {
        fn gen(req: GenReq) -> f32 {
            match req {
                GenReq::Weight { l: 0, .. } => 2.0,
                GenReq::Weight { l: 1, i_l: 0, .. } => 1.0,
                GenReq::Bias { l: 2, i_l: 0 } => -10.0,
                _ => 0.0,
            }
        }
        let net = NNetwork::new(&[3, 3, 1], gen).unwrap();
        let inputs = vec![1.0, 2.0, 0.5];

        // As visible in the generator, we should get:
        // ( ( 1.0 * 2.0 + 2.0 * 2.0 + 0.5 * 2.0 ) - 10.0 ) * 0.01 = -0.03
        let out = net.eval(inputs, &LeakyRelu(0.01));

        assert_eq!(&out, &[-0.03]);
    }

    // Tests the custom display implementation.
    #[test]
    fn display() {
        let net = NNetwork::new(&[2, 3, 1], |_| -0.9999).unwrap();
        let expected = "\
| -1.00 -1.00 |
| -1.00 -1.00 |
| -1.00 -1.00 |
{ -1.00 -1.00 -1.00 }
| -1.00 -1.00 -1.00 |
{ -1.00 }
";
        assert_eq!(format!("{net}"), expected);
    }
}
