use rann_base::{
    common::{random_biases, random_unit, random_weights, Logistic},
    full::Full,
};
use rann_traits::{compose::zip, Network};

#[test]
fn simple() {
    let net = Full::<1, 5, _>::new(Logistic, random_weights, random_biases)
        .chain(Full::new(Logistic, random_weights, random_biases))
        .chain(Full::<10, 5, _>::new(
            Logistic,
            random_weights,
            random_biases,
        ))
        .chain(Full::<5, 1, _>::new(
            Logistic,
            random_weights,
            random_biases,
        ));
    let other = Full::<5, 100, _>::new(Logistic, random_weights, random_biases);
    let net = net.zip(other, zip::stacked, zip::)
}   
