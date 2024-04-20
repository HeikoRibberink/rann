use float_cmp::{ApproxEq, F32Margin};
use rann_base::{
    common::{random_biases, random_weights, Logistic, Random},
    error::SquareError,
    full::Full,
};
use rann_traits::{compose::zip, Intermediate, Network};

#[test]
/// A simple example showcasing how to compose a neural network.
fn simple_compose() {
    // The learning rate.
    const RATE: f32 = 0.5;
    // Prints statistics every `REP` iterations.
    const STAT: usize = 1000;
    // The amount of iterations to train the network for.
    const ITER: usize = 100000;
    // The expected values.
    const EXPECTED: [f32; 6] = [0.99, 0.1, 0.5, 0.3, 0.789, 0.6];
    // Network inputs.
    const INPUT: ([f32; 1], [f32; 5]) = ([5.0], [2.0; 5]);
    // Builds a chain of layers, forming a network.
    let net = Full::<1, 5, _>::new(Logistic, Random)
        // You can even remove some intermediate const parameters.
        .chain(Full::new(Logistic, Random))
        .chain(Full::<10, 5, _>::new(Logistic, Random))
        .chain(Full::<5, 1, _>::new(Logistic, Random));

    // Here we declare an independent layer...
    let other = Full::<5, 5, _>::new(Logistic, Random);
    // ... which we zip together with the previously defined network.
    let net = net.zip(other, zip::Stacker::<1, 5, { 1 + 5 }>);

    // Finally, we declare an error function, after which we have a fully functional and trainable
    // network.
    let mut net = net.chain(SquareError { expected: EXPECTED });

    // Then, we train the network periodically.
    for i in 0..ITER {
        // To train the network, we have to evaluate it at least once.
        let inter = net.intermediate(&INPUT);

        // We periodically print some statistics to monitor the network.
        if i % STAT == 0 {
            println!("Error: {:?}", inter.output());
            println!("Outputs: {:?}", inter.first.output())
        }

        // And then we train the network on the inputs and calculations we already did.
        net.train(&INPUT, &inter, RATE);
    }

    // After fully training the network, we test its performance.

    let inter = net.intermediate(&INPUT);
    // We can break the intermediate calculation up into an error...
    let err = inter.output();
    // ...and the actual output of the network.
    let out = inter.first.output();

    assert!(err[0] < 0.1f32, "Error {err:?} is too large.");
    assert!(
        out.approx_eq(
            &EXPECTED,
            F32Margin {
                epsilon: 0.01,
                ulps: 10
            }
        ),
        "{out:?} is too different from {EXPECTED:?}."
    );
}
