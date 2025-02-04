use crate::dft::DFT;
use mod_exp::mod_exp;
use rand::{self, Rng};

pub enum Prime {
    /// Any prime <= 2^64
    Generic,
    /// Any prime <= 2^51
    /// The implementation uses floating point arithmetic
    AvxCompatible,
    /// Goldilocks prime, i.e. p = phi^2 - phi + 1
    Goldilocks,
}

pub struct Table<O> {
    /// NTT friendly prime modulus
    q: O,
    /// n-th root of unity
    psi: O,
}

impl Table<u64> {
    pub fn new() -> Self {
        Self {
            q: 0x1fffffffffe00001u64,
            psi: 0x15eb043c7aa2b01fu64, //2^17th root of unity
        }
    }

    pub fn new_simple() -> Self {
        // Values taken from https://eprint.iacr.org/2024/585.pdf
        Self {
            q: 7681,
            // 2^3th root of unity
            psi: 4121,
        }
    }
}

impl DFT<u64> for Table<u64> {
    /// NTT forward routine
    ///
    /// - `a`: vector with each element in range `[0, q)`
    fn forward_inplace(&self, a: &mut [u64]) {
        self.forward_inplace_core::<false>(a)
    }

    /// NTT forward lazy routine
    ///
    /// - `a`: vector with each element in range `[0, 2q)`
    fn forward_inplace_lazy(&self, a: &mut [u64]) {
        self.forward_inplace_core::<true>(a)
    }

    /// NTT backward routine
    ///
    /// - `a`: vector with each element in range `[0, q)`
    fn backward_inplace(&self, a: &mut [u64]) {
        self.backward_inplace_core::<false>(a)
    }

    /// NTT backward lazy routine
    ///
    /// - `a`: vector with each element in range `[0, 2q)`
    fn backward_inplace_lazy(&self, a: &mut [u64]) {
        self.backward_inplace_core::<true>(a)
    }
}

impl Table<u64> {
    fn mod_exp(&self, base: u64, mut exp: u64) -> u64 {
        let mut out = 1;

        let mut acc = base;

        while exp > 0 {
            if exp % 2 == 1 {
                out = ((out as u128 * acc as u128) % self.q as u128) as u64;
            }

            acc = ((acc as u128 * acc as u128) % self.q as u128) as u64;

            exp >>= 1;
        }

        out
    }

    pub fn forward_inplace_core<const LAZY: bool>(&self, a: &mut [u64]) {
        let N = 2 ^ 16;

        let mut t: u64 = N;
        let mut m: u64 = 1;

        loop {
            if m >= N {
                break;
            }

            t = t / 2;

            for i in 0..m {
                let j_1 = 2 * i * t;
                let j_2 = j_1 + t - 1;

                let cap_s = self.mod_exp(self.psi, ((m + i) as u16).reverse_bits() as u64);

                for j in j_1..=j_2 {
                    let cap_u = a[j as usize];
                    let cap_v = a[j as usize + t as usize] * cap_s;

                    a[j as usize] = (cap_u + cap_v) % self.q;
                    a[j as usize + t as usize] = (cap_u - cap_v) % self.q;
                }
            }

            m *= 2;
        }
    }

    pub fn backward_inplace_core<const LAZY: bool>(&self, a: &mut [u64]) {}
}

/// Assuming that m is prime and m-1 = 2^n * ...
fn find_two_nth_unity_root(m: u64) -> u64 {
    let mut rand = rand::rng();

    let mut tmp;
    loop {
        println!("iteration");
        tmp = rand.random_range(2..m);

        match mod_exp(tmp as u128, ((m - 1) / 2) as u128, m as u128) {
            1 => continue,
            _ => break tmp,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::u128;

    use crate::dft::DFT;

    use super::{find_two_nth_unity_root, Table};

    trait NaiveImplementation {
        fn forward_naive(&self, a: &mut [u64], omega: u64);

        fn backward_naive(&self, a: &mut [u64], omega: u64);
    }

    impl NaiveImplementation for Table<u64> {
        fn forward_naive(&self, a: &mut [u64], omega: u64) {
            let a_len = a.len();

            let mut out = Vec::<u64>::with_capacity(a_len);
            out.resize(a_len, 0);

            for j in 0..a_len {
                for i in 0..a_len {
                    let i_j = (i * j) as u64 % self.q;

                    let omega_i_j = self.mod_exp(omega, i_j);

                    out[j] = out[j]
                        + ((a[i] as u128 * omega_i_j as u128) % self.q as u128) as u64 % self.q;
                }
            }

            for i in 0..a_len {
                a[i] = out[i];
            }
        }

        fn backward_naive(&self, a: &mut [u64], omega: u64) {
            let a_len = a.len();

            let mut out = Vec::<u64>::with_capacity(a_len);
            out.resize(a_len, 0);

            // Invert by exponentiation
            let omega_inv = self.mod_exp(omega, self.q - 2);

            for j in 0..a_len {
                for i in 0..a_len {
                    let i_j = (i * j) as u64 % self.q;

                    let psi_i_j = self.mod_exp(omega_inv, i_j);

                    out[j] = out[j]
                        + ((a[i] as u128 * psi_i_j as u128) % self.q as u128) as u64 % self.q;
                }
            }

            // Invert by exponentiation
            let n_inv = self.mod_exp(a_len as u64, self.q - 2);

            for i in 0..a_len {
                a[i] = ((n_inv as u128 * out[i] as u128) % self.q as u128) as u64;
            }
        }
    }

    #[test]
    fn test_mod_exp() {
        let table = Table::<u64>::new_simple();

        let res = table.mod_exp(4, table.q - 2);

        assert_eq!(res, 5761);
        println!("res {}", res);
    }

    #[test]
    fn test_simple_case() {
        let table = Table::<u64>::new_simple();

        let mut a = [1, 2, 3, 4];
        table.forward_inplace(&mut a);

        println!("ntt {a:?}");

        // table.backward_inplace(&mut a);

        println!("intt {a:?}");
    }

    // #[test]
    // fn test_mod_mul() {
    //     println!("{}", mod_mul_u64(2, 3, 23));
    // }

    // #[test]
    // fn test_mod_inv() {
    //     println!("{}", mod_inv_u64(3383, 7681));
    // }

    #[test]
    fn test_find_nth_unity_root() {
        println!("{}", find_two_nth_unity_root(7681));
    }
}
