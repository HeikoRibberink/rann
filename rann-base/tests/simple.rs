use rann_base::{
    common::{random_unit, LeakyRelu, Sum},
    net::NNetwork,
};

// Tests if a network can approximate a function with constant inputs and outputs.
#[test]
fn simple() {
    const RATE: f32 = 0.10;
    let activation = LeakyRelu(0.1);

    let mut net = NNetwork::new(&[3, 5, 8], random_unit).unwrap();
    let expected = vec![1.0, 0.1, 0.4, 0.3, 0.5, 0.2, 0.7, 0.8];
    let error_fn = Sum {
        expected: expected.clone(),
    };
    let input = vec![1.0, 0.0, 0.5];
    for i in 0..10000 {
        let inter = net.eval_intermediate(input.clone(), &activation);
        if i % 100 == 0 {
            println!();
            dbg!(&expected);
            dbg!(&inter.activs.last().unwrap());
        }
        let err = net.backprop(inter, &activation, &error_fn, RATE);
        if i % 100 == 0 {
            dbg!(&err);
        }
    }
    let mut out = net.eval(input.clone(), &activation);
    out.iter_mut()
        .for_each(|x| *x = (*x * 1000.0).round() / 1000.0);
    assert_eq!(expected, out);
}
