//! Network composition
//!
//! If you want to build a network, such as by connecting different layers or networks, then you
//! have come to the right place! This module provides methods to compose networks in different
//! ways, such as chaining and zipping.

pub mod zip;
pub mod chain;

pub use chain::*;
pub use zip::{Zip, ZipInter};
