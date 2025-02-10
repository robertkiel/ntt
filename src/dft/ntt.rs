use crate::dft::DFT;
use crate::utils::bit_reverse;
use rand::{self, Rng};

pub struct Table<O> {
    /// NTT friendly prime modulus
    q: O,
    /// n-th root of unity
    psi: O,
    n: O,
    powers_psi_bo: Vec<u64>,
    powers_psi_inv_bo: Vec<u64>,
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
    pub fn new() -> Self {
        let mut res = Self {
            q: 0x1fffffffffe00001u64,
            psi: 0x15eb043c7aa2b01fu64, //2^17th root of unity
            n: 2u64.pow(16),
            powers_psi_bo: Vec::with_capacity(2usize.pow(16)),
            powers_psi_inv_bo: Vec::with_capacity(2usize.pow(16)),
        };

        res.with_precomputes();

        res
    }

    pub fn new_float_compatible() -> Self {
        let mut res = Self {
            q: 4503599626321921,
            psi: 4183818951195512,
            n: 2u64.pow(16),
            powers_psi_bo: Vec::with_capacity(2usize.pow(16)),
            powers_psi_inv_bo: Vec::with_capacity(2usize.pow(16)),
        };

        res.with_precomputes();

        res
    }

    pub fn new_u32_compatible() -> Self {
        let mut res = Self {
            q: 4293918721,
            psi: 2004365341,
            n: 2u64.pow(16),
            powers_psi_bo: Vec::with_capacity(2usize.pow(16)),
            powers_psi_inv_bo: Vec::with_capacity(2usize.pow(16)),
        };

        res.with_precomputes();

        res
    }

    pub fn new_simple() -> Self {
        // Values taken from https://eprint.iacr.org/2024/585.pdf
        let mut res = Self {
            q: 7681,
            // 2^3th root of unity
            psi: 1925,
            n: 2u64.pow(2),
            powers_psi_bo: Vec::with_capacity(2usize.pow(3)),
            powers_psi_inv_bo: Vec::with_capacity(2usize.pow(3)),
        };

        res.with_precomputes();

        res
    }

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
        let a_len = a.len();

        let mut t = a_len;
        let mut m = 1;

        loop {
            if m >= a_len {
                break;
            }

            t = t / 2;

            for i in 0..m {
                let j_1 = 2 * i * t;
                let j_2 = j_1 + t - 1;

                let cap_s = self.powers_psi_bo[(m + i) as usize];

                for j in j_1..=j_2 {
                    let cap_u = a[j as usize];
                    let cap_v = self.mul_reduce(a[j as usize + t as usize], cap_s);

                    let mut cap_u_add_cap_v = cap_u + cap_v;
                    if cap_u_add_cap_v > self.q {
                        cap_u_add_cap_v -= self.q;
                    }
                    a[j as usize] = cap_u_add_cap_v;

                    let cap_u_sub_cap_v = match cap_u.overflowing_sub(cap_v) {
                        (res, true) => {
                            let (inner_res, overflow) = res.overflowing_add(self.q);

                            assert!(overflow);

                            inner_res
                        }
                        (res, false) => res,
                    };

                    a[j as usize + t as usize] = cap_u_sub_cap_v % self.q;
                }
            }

            m *= 2;
        }
    }

    pub fn backward_inplace_core<const LAZY: bool>(&self, a: &mut [u64]) {
        let a_len = a.len();

        let mut t = 1;
        let mut m: u64 = a_len as u64;

        loop {
            if m == 1 {
                break;
            }

            let mut j_1 = 0;
            let h = m / 2;

            for i in 0..h {
                let j_2 = j_1 + t - 1;
                let cap_s = self.powers_psi_inv_bo[(h + i) as usize];

                for j in j_1..=j_2 {
                    let cap_u = a[j];
                    let cap_v = a[j + t];

                    let mut cap_u_add_cap_v = cap_u + cap_v;

                    if cap_u_add_cap_v > self.q {
                        cap_u_add_cap_v -= self.q;
                    }
                    a[j] = cap_u_add_cap_v;

                    let cap_u_sub_cap_v = match cap_u.overflowing_sub(cap_v) {
                        (res, true) => {
                            let (inner_res, overflow) = res.overflowing_add(self.q);

                            assert!(overflow);

                            inner_res
                        }
                        (res, false) => res,
                    };

                    a[j as usize + t as usize] = self.mul_reduce(cap_u_sub_cap_v, cap_s);
                }
                j_1 += 2 * t;
            }

            t *= 2;

            m /= 2;
        }

        let n_inv = self.mod_exp(a_len as u64, self.q - 2);
        for j in 0..a_len {
            a[j as usize] = self.mul_reduce(a[j as usize], n_inv);
        }
    }

    #[inline]
    fn mul_reduce(&self, a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) % self.q as u128) as u64
    }

    fn with_precomputes(&mut self) {
        let psi_inv = self.mod_exp(self.psi, self.q - 2);

        let mut tmp_psi = 1u64;
        let mut tmp_psi_inv = 1u64;

        self.powers_psi_bo.resize(self.n as usize, 0);
        self.powers_psi_inv_bo.resize(self.n as usize, 0);

        let log_n = self.n.ilog2();

        for i in 0..self.n {
            let reversed_index = bit_reverse(i, log_n) as usize;
            self.powers_psi_bo[reversed_index] = tmp_psi;
            self.powers_psi_inv_bo[reversed_index] = tmp_psi_inv;

            tmp_psi = self.mul_reduce(tmp_psi, self.psi);
            tmp_psi_inv = self.mul_reduce(tmp_psi_inv, psi_inv);
        }
    }

    /// Assuming that m is prime and m-1 = 2^n * ...
    fn find_nth_unity_root(&self, n: u64, m: u64) -> u64 {
        let mut rand = rand::rng();

        let mut tmp;
        loop {
            tmp = rand.random_range(2..m);

            let g = self.mod_exp(tmp, (m - 1) / n);

            match self.mod_exp(g, n / 2) {
                1 => continue,
                _ => break g as u64,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::u128;

    use rand::Rng;

    use crate::dft::DFT;

    use super::Table;

    trait NaiveImplementation {
        fn forward_naive(&self, a: &mut [u64]);

        fn backward_naive(&self, a: &mut [u64]);
    }

    impl NaiveImplementation for Table<u64> {
        fn forward_naive(&self, a: &mut [u64]) {
            let a_len = a.len();

            let omega = self.mod_exp(self.psi, 2);

            let mut out = Vec::<u64>::with_capacity(a_len);
            out.resize(a_len, 0);

            for j in 0..a_len {
                for i in 0..a_len {
                    let i_j = (i * j) as u64 % self.q;

                    let omega_i_j = self.mod_exp(omega, i_j);

                    out[j] = (out[j]
                        + ((a[i] as u128 * omega_i_j as u128) % self.q as u128) as u64)
                        % self.q;
                }
            }

            for i in 0..a_len {
                a[i] = out[i];
            }
        }

        fn backward_naive(&self, a: &mut [u64]) {
            let a_len = a.len();

            let omega = self.mod_exp(self.psi, 2);

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

        let mut a = [1001, 2002, 3003, 4004];
        table.forward_inplace(&mut a);

        println!("ntt {a:?}");

        let mut a_naive = [1, 2, 3, 4];

        table.forward_naive(&mut a_naive);

        println!("ntt naive {a_naive:?}");

        table.backward_inplace(&mut a);

        println!("intt {a:?}");
    }

    #[test]
    fn test_normal_case() {
        let mut rng = rand::rng();
        let table = Table::<u64>::new();

        let mut a = [0u64; 2u64.pow(16) as usize];

        for i in 0..a.len() {
            a[i] = rng.random_range(0..2 ^ 61);
        }

        let a_clone = a.clone();
        table.forward_inplace(&mut a);

        table.backward_inplace(&mut a);

        assert_eq!(a_clone, a);
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
        let foo = Table::new();
        println!("{}", foo.find_nth_unity_root(8, 7681));
    }

    #[test]
    fn test_arithmetics() {
        let res = 3u64.overflowing_sub(5u64);

        println!("{:?}", res.0 + 5);
    }
}
