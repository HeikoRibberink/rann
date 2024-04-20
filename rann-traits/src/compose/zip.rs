use arrayvec::ArrayVec;

use crate::{Intermediate, Network, Scalar};
/// Zip two parallel networks into the same output.
///
/// # Type parameters
/// - `T` and `U` represent the zipped networks.
/// - `Z` is a function that combines the outputs of both networks into one.
/// - `UnZ` must do exactly the reverse of `Z`: take the combined outputs of the networks and pull
/// them apart.
#[derive(Debug, Clone)]
pub struct Zip<T, U, Z, UnZ> {
    pub top: T,
    pub bot: U,
    pub zipper: Z,
    pub unzipper: UnZ,
}

impl<T, U, Z, UnZ, C> Network for Zip<T, U, Z, UnZ>
where
    T: Network,
    U: Network,
    Z: Fn(&T::Out, &U::Out) -> C,
    UnZ: for<'a> Fn(&'a C) -> (&'a T::Out, &'a U::Out),
{
    type In = (T::In, U::In);

    type Out = C;

    type Inter = ZipInter<T::Inter, U::Inter, C>;

    fn intermediate(&self, input: &Self::In) -> Self::Inter {
        // Evaluate both networks.
        let top = self.top.intermediate(&input.0);
        let bot = self.bot.intermediate(&input.1);
        ZipInter {
            // Combine both outputs.
            zipped: (self.zipper)(top.output(), bot.output()),
            top,
            bot,
        }
    }

    fn train_deriv(
        &mut self,
        inputs: &Self::In,
        intermediate: &Self::Inter,
        gradients: &Self::Out,
        learning_rate: Scalar,
    ) -> Self::In {
        // Unzip the gradients.
        let (top_gr, bot_gr) = (self.unzipper)(gradients);
        // Train the top network.
        let top = self
            .top
            .train_deriv(&inputs.0, &intermediate.top, top_gr, learning_rate);
        // Train the bottom network.
        let bot = self
            .bot
            .train_deriv(&inputs.1, &intermediate.bot, bot_gr, learning_rate);
        // Combine gradients.
        (top, bot)
    }
}

/// The intermediate values of an evaluation of a [`Zip`].
#[derive(Debug)]
pub struct ZipInter<T, U, Z> {
    /// The intermediate values of the top network.
    pub top: T,
    /// The intermediate values of the bottom network.
    pub bot: U,
    /// The combined output of the networks.
    pub zipped: Z,
}

impl<T, U, Z> Intermediate for ZipInter<T, U, Z>
where
    T: Intermediate,
    U: Intermediate,
{
    type Out = Z;

    fn output(&self) -> &Self::Out {
        &self.zipped
    }

    fn into_output(self) -> Self::Out {
        self.zipped
    }
}

// Zippers

/// Stacks and unstacks constant arrays.
#[derive(Clone, Copy, Debug)]
pub struct Stacker<const A: usize, const B: usize, const SUM: usize>;

impl<const A: usize, const B: usize, const SUM: usize>
    Into<(
        fn(&[Scalar; A], &[Scalar; B]) -> [Scalar; SUM],
        fn(&[Scalar; SUM]) -> (&[Scalar; A], &[Scalar; B]),
    )> for Stacker<A, B, SUM>
{
    fn into(
        self,
    ) -> (
        for<'a, 'b> fn(&'a [f32; A], &'b [f32; B]) -> [f32; SUM],
        for<'a> fn(&'a [f32; SUM]) -> (&'a [f32; A], &'a [f32; B]),
    ) {
        (stacked, unstacked)
    }
}

/// Stacks the vectors.
pub fn stacked<const A: usize, const B: usize, const SUM: usize>(
    top: &[Scalar; A],
    bot: &[Scalar; B],
) -> [Scalar; SUM] {
    top.iter()
        .chain(bot)
        .map(|x| *x)
        .collect::<ArrayVec<Scalar, SUM>>()
        .into_inner()
        .expect("SUM should be A + B.")
}

/// Unstacks the vectors.
pub fn unstacked<const A: usize, const B: usize, const SUM: usize>(
    x: &[Scalar; SUM],
) -> (&[Scalar; A], &[Scalar; B]) {
    let (a, b) = x.split_at(A);
    (a.try_into().unwrap(), b.try_into().unwrap())
}
