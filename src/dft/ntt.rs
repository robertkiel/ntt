//! NTT implementation

use crate::dft::DFT;
use crate::utils::{bit_reverse, egcd_i128, MontgomeryTransformer};

pub struct Table<O> {
    /// NTT friendly prime modulus
    pub q: O,
    /// n-th root of unity
    pub psi: O,
    montgomery: MontgomeryTransformer<O>,
    /// Multiplicative inverse of n in Montgomery form
    n_inv_montgomery: O,
    n: O,
    /// Multiplicative inverse of psi in Montgomery form
    psi_inv_montgomery: O,
    pub powers_psi_bo: Vec<O>,
    /// Precomputed powers of the multiplicative inverse of psi in Montgomery form
    /// and stored in bit-reversed order
    pub powers_psi_inv_bo: Vec<O>,
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
    /// Instantiates NTT with a base prime
    pub fn new() -> Self {
        Self::default()
    }

    /// Instantiates NTT with NTT-friendly prime that allows very fast reduction
    pub fn new_goldilock() -> Self {
        Self::from_prime_and_root(0xffffffff00000001, 0xabd0a6e8aa3d8a0e, 2u64.pow(16))
    }

    /// Instantiates NTT with prime that can be used with 32-bit
    /// registers, i.e. on AVX2
    pub fn new_u32_compatible() -> Self {
        Self::from_prime_and_root(4293918721, 2004365341, 2u64.pow(16))
    }

    /// Instantiates NTT with small prime to check correctness of
    /// the implementation
    pub fn new_simple() -> Self {
        // Values taken from https://eprint.iacr.org/2024/585.pdf
        Self::from_prime_and_root(7681, 1925, 2u64.pow(3))
    }

    /// Takes a modulus and a n-th root of unity and instatiates a struct that allows fast NTT and INTT operations
    ///
    /// ```rust
    /// use ntt::dft::{ntt::Table, DFT};
    ///
    /// const PRIME: u64 = 0x1fffffffffe00001u64;
    /// /// 2^17th root of unity
    /// const ROOT: u64 = 0x15eb043c7aa2b01fu64;
    /// const N: u64 = 2u64.pow(16);
    ///
    /// Table::from_prime_and_root(PRIME, ROOT, N);
    /// ```
    ///
    pub fn from_prime_and_root(modulus: u64, root: u64, n: u64) -> Self {
        let montgomery = MontgomeryTransformer::<u64>::from_modulus(modulus);

        let n_inv_raw = egcd_i128(n as i128, modulus as i128).1;

        let n_inv = if n_inv_raw < 0 {
            (n_inv_raw + modulus as i128) as u64
        } else {
            n_inv_raw as u64
        };

        let psi_inv_raw = egcd_i128(root as i128, modulus as i128).1;

        let psi_inv = if psi_inv_raw < 0 {
            (psi_inv_raw + modulus as i128) as u64
        } else {
            psi_inv_raw as u64
        };

        Self {
            q: modulus,
            psi: root,
            n_inv_montgomery: montgomery.transform(n_inv),
            psi_inv_montgomery: montgomery.transform(psi_inv),
            montgomery,
            n,
            powers_psi_bo: Vec::with_capacity(n as usize),
            powers_psi_inv_bo: Vec::with_capacity(n as usize),
        }
        .with_precomputes()
    }

    pub fn forward_inplace_core<const LAZY: bool>(&self, a: &mut [u64]) {
        let a_len = a.len();

        for a_j in a.iter_mut() {
            *a_j = self.montgomery.transform(*a_j);
        }

        let mut t = a_len;
        let mut m = 1;

        loop {
            if m >= a_len {
                break;
            }

            t /= 2;

            for i in 0..m {
                let j_1 = 2 * i * t;
                let j_2 = j_1 + t - 1;

                let cap_s = self.powers_psi_bo[m + i];

                for j in j_1..=j_2 {
                    let cap_u = a[j];
                    let cap_v = self.montgomery.mul(a[j + t], cap_s);

                    let cap_u_add_cap_v = match cap_u.overflowing_add(cap_v) {
                        (res, true) => res.overflowing_sub(self.q).0,
                        (mut res, false) => {
                            if res > self.q {
                                res -= self.q
                            }
                            res
                        }
                    };

                    a[j] = cap_u_add_cap_v;

                    let cap_u_sub_cap_v = match cap_u.overflowing_sub(cap_v) {
                        (res, true) => res.overflowing_add(self.q).0,
                        (res, false) => res,
                    };

                    a[j + t] = cap_u_sub_cap_v;
                }
            }

            m *= 2;
        }

        for a_j in a.iter_mut() {
            *a_j = self.montgomery.reduce(*a_j as u128)
        }
    }

    pub fn backward_inplace_core<const LAZY: bool>(&self, a: &mut [u64]) {
        let a_len = a.len();

        for a_j in a.iter_mut() {
            *a_j = self.montgomery.transform(*a_j);
        }

        let mut t = 1;
        let mut m = a_len;

        loop {
            if m == 1 {
                break;
            }

            let mut j_1 = 0;
            let h = m / 2;

            for i in 0..h {
                let j_2 = j_1 + t - 1;
                let cap_s = self.powers_psi_inv_bo[h + i];

                for j in j_1..=j_2 {
                    let cap_u = a[j];
                    let cap_v = a[j + t];

                    let cap_u_add_cap_v = match cap_u.overflowing_add(cap_v) {
                        (res, true) => res.overflowing_sub(self.q).0,
                        (mut res, false) => {
                            if res > self.q {
                                res -= self.q
                            }
                            res
                        }
                    };

                    a[j] = cap_u_add_cap_v;

                    let cap_u_sub_cap_v = match cap_u.overflowing_sub(cap_v) {
                        (res, true) => res.overflowing_add(self.q).0,
                        (res, false) => res,
                    };

                    a[j + t] = self.montgomery.mul(cap_u_sub_cap_v, cap_s);
                }
                j_1 += 2 * t;
            }

            t *= 2;

            m /= 2;
        }

        for a_j in a.iter_mut() {
            *a_j = self
                .montgomery
                .reduce(self.montgomery.mul(*a_j, self.n_inv_montgomery) as u128);
        }
    }

    /// Computes and stores precomputable values, especially powers of `psi`.
    fn with_precomputes(mut self) -> Self {
        let psi_montgomery = self.montgomery.transform(self.psi);

        let mut tmp_psi = self.montgomery.transform(1);
        let mut tmp_psi_inv = self.montgomery.transform(1);

        let n = self.powers_psi_bo.capacity();
        self.powers_psi_bo.resize(self.n as usize, 0);
        self.powers_psi_inv_bo.resize(self.n as usize, 0);

        let log_n = n.ilog2();

        for i in 0..self.n {
            let reversed_index = bit_reverse(i, log_n) as usize;
            self.powers_psi_bo[reversed_index] = tmp_psi;
            self.powers_psi_inv_bo[reversed_index] = tmp_psi_inv;

            tmp_psi = self.montgomery.mul(tmp_psi, psi_montgomery);
            tmp_psi_inv = self.montgomery.mul(tmp_psi_inv, self.psi_inv_montgomery);
        }

        self
    }
}

impl Default for Table<u64> {
    fn default() -> Self {
        Self::from_prime_and_root(0x1fffffffffe00001u64, 0x15eb043c7aa2b01fu64, 2u64.pow(16))
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::dft::DFT;

    use super::Table;

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
}
