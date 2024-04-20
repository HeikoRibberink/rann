
pub mod common;
pub mod error;
pub mod full;
pub mod net;

pub use common::{random_biases, random_weights, Logistic};
pub use error::SquareError;
pub use full::{Full, FullInter};
