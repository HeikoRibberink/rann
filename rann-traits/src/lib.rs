/// The default scalar type
pub type Scalar = f32;

/// Trait implemented by neural networks that can be evaluated.
pub trait Evaluable {
    const NUM_IN: usize;
    const NUM_OUT: usize;
    fn eval(&self, inputs: &[f32; Self::NUM_IN]) -> Self::NUM_OUT;
}

/// Trait implemented by neural networks that can be trained.
pub trait Trainable {

}
