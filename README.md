# RANN
Rust Artificial Neural Network crate

---
A set of experimental libraries for neural networks in Rust, focusing on allocation-free and reusable components.

[`rann-traits`](./rann-traits/README.md) contains all the different traits necessary to compose neural networks and build generic, reusable components.

[`rann-base`](./rann-base/README.md) contains *allocation-free* implementations of network layers, such as:
- [X] Fully connected layer: [`Full`],
- [ ] Convolution layer,
- [ ] [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) layer,
and utilities such as:
- activation functions,
- error functions,
- network generators,
and others.
