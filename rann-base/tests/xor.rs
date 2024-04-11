use fastrand as rnd;
use rann_base::{
    common::{random_unit, LeakyRelu, Sum},
    net::NNetwork,
};
use rann_traits::deriv::NDeriv;

// Trains a neural network that approximates the XOR function, and tests if it doesn't diverge.
#[test]
fn xor_divergence() {
    // The learning rate.
    const RATE: f32 = 0.10;
    // The number of iterations to calculate an average loss.
    const AVG_NUM: usize = 10000;
    let mut avg = vec![0.0; AVG_NUM];
    // The number of iterations to print the average loss and the current ntetwork.
    const PRINT_NUM: usize = 10000;
    // The activation function to use.
    let activation = LeakyRelu(0.1);
    // Initializes the neural network with 2 input, 3 hidden and 1 output layer.
    let mut net = NNetwork::new(&[2, 3, 1], random_unit).unwrap();
    // The error function to use.
    let mut error_fn = Sum {
        expected: vec![0.0],
    };
    // Train network
    for i in 1..100000 {
        // Prepare input
        let a = rnd::bool();
        let b = rnd::bool();
        // Prepare output
        let e_out = (a ^ b).into();
        error_fn.expected[0] = e_out;

        let a_f: f32 = a.into();
        let b_f: f32 = b.into();
        // Evaluate
        let inter = net.eval_intermediate(vec![a_f, b_f], &activation);
        // Backpropagate
        let err = net.backprop(inter, &activation, &error_fn, RATE);
        // Assert that the network doesn't diverge.
        assert!(!err.is_nan());
        avg[i % AVG_NUM] = err;
        if i % PRINT_NUM == 0 {
            let sum: f32 = avg.iter().sum();
            println!("{}", sum / AVG_NUM as f32);
            println!("{}", &net);
        }
    }
    // Test network
    for _ in 0..100 {
        let a = rnd::bool();
        let b = rnd::bool();
        // Prepare output
        let e_out = (a ^ b).into();
        error_fn.expected[0] = e_out;

        let a_f: f32 = a.into();
        let b_f: f32 = b.into();

        let out = net.eval(vec![a_f, b_f], &activation);
        let err = error_fn.call(&out);
        println!(
        "Evaluated:
          Inputs           {a_f}, {b_f}
          Expected output  {e_out}
          Output           {out:?}
          Error            {err}");
    }
}
