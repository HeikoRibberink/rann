use fastrand::Rng;
use rann_base::{activ::LeakyRelu, error::SumError, Full};
use rann_traits::{Intermediate, Network};

// Trains a neural network that approximates the XOR function, and tests if it doesn't diverge.
#[test]
fn xor_divergence() {
    /// The learning rate.
    const RATE: f32 = 0.10;
    /// The number of iterations to calculate an average loss.
    const AVG_NUM: usize = 10000;
    // The array of errors to average.
    let mut avg = vec![0.0; AVG_NUM];

    // The number of iterations to print the average loss and the current ntetwork.
    const PRINT_NUM: usize = 10000;

    // The activation function to use.
    let activation = LeakyRelu(0.1);

    // The generator for the network values. For the test to be deterministic, we have to seed the
    // generator.
    let mut rng = Rng::with_seed(0x2);
    let gen = (
        {
            let mut rng = rng.clone();
            move |_, _| rng.f32() * 4.0 - 2.0
        },
        {
            let mut rng = rng.clone();
            move |_| rng.f32() * 4.0 - 2.0
        },
    );

    // Initializes the neural network with 2 input, 3 hidden and 1 output layer.
    let mut net = Full::<2, 3, _>::new(activation, gen.clone())
        .chain(Full::<3, 1, _>::new(activation, gen))
        .chain(SumError { expected: [0.0] });

    // Train network
    for i in 1..100000 {
        // Prepare input
        let a = rng.bool();
        let b = rng.bool();
        // Prepare output
        let e_out = (a ^ b).into();
        net.second.expected[0] = e_out;

        let a_f: f32 = a.into();
        let b_f: f32 = b.into();
        let inputs = [a_f, b_f];
        // Evaluate
        let inter = net.intermediate(&inputs);
        // Backpropagate
        net.train(&inputs, &inter, RATE);
        let err = inter.output()[0];
        // Assert that the network doesn't diverge.
        assert!(
            !err.is_nan(),
            "{err} is not normal. Prev: {:?}",
            avg[(i - 1) % AVG_NUM]
        );
        avg[i % AVG_NUM] = err;
        if i % PRINT_NUM == 0 {
            let avg: f32 = avg.iter().sum::<f32>() / AVG_NUM as f32;
            println!("{avg}");
        }
    }
    let tests = [(false, false), (false, true), (true, false), (true, true)];
    // Test network
    for (a, b) in tests {
        // Prepare output
        let e_out = (a ^ b).into();
        net.second.expected[0] = e_out;

        let a_f: f32 = a.into();
        let b_f: f32 = b.into();

        let eval = net.intermediate(&[a_f, b_f]);
        let err = eval.output()[0];
        let out = eval.first.output()[0];

        println!(
            "Evaluated:
          Inputs           {a_f}, {b_f}
          Expected output  {e_out}
          Output           {out:?}
          Error            {err}"
        );

        assert!(
            (out - e_out).abs() < 0.1,
            "{out} should be close to {e_out}."
        );
    }
}
