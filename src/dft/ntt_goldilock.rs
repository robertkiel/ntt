//! NTT implementation using specialized prime that allows quick reductions

use crate::utils::bit_reverse;
use crate::{dft::DFT, utils::mod_exp_u64};

pub struct TableGoldilock<O> {
    /// NTT friendly prime modulus
    pub q: O,
    /// n-th root of unity
    pub psi: O,
    n: O,
    n_inv: O,
    powers_psi_bo: Vec<O>,
    powers_psi_inv_bo: Vec<O>,
}

impl DFT<u64> for TableGoldilock<u64> {
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

impl TableGoldilock<u64> {
    pub fn new() -> Self {
        Self {
            q: 0xffffffff00000001,
            psi: 0xabd0a6e8aa3d8a0e, //2^17th root of unity
            n: 2u64.pow(16),
            n_inv: 0xfffeffff00010001,
            powers_psi_bo: Vec::with_capacity(2usize.pow(16)),
            powers_psi_inv_bo: Vec::with_capacity(2usize.pow(16)),
        }
        .with_precomputes()
    }

    /// Reduce a 159-bit number modulo 18446744069414584321
    ///
    /// see [original post](https://cp4space.hatsya.com/2021/09/01/an-efficient-prime-for-number-theoretic-transforms/)
    pub fn reduce_159(&self, low: u64, middle: u64, high: u64) -> u64 {
        let low_2 = match low.overflowing_sub(high) {
            (res, true) => res.overflowing_add(self.q).0,
            (res, false) => res,
        };

        let mut product = middle << 32;
        product -= product >> 32;

        let mut res = low_2.overflowing_add(product).0;

        if (res < product) || (res >= self.q) {
            res = res.overflowing_sub(self.q).0;
        }

        res
    }

    pub fn forward_inplace_core<const LAZY: bool>(&self, a: &mut [u64]) {
        let a_len = a.len();

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
                    let cap_v = self.mul_reduce(a[j + t], cap_s);

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

                    a[j + t] = cap_u_sub_cap_v % self.q;
                }
            }

            m *= 2;
        }
    }

    pub fn backward_inplace_core<const LAZY: bool>(&self, a: &mut [u64]) {
        let a_len = a.len();

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

                    a[j + t] = self.mul_reduce(cap_u_sub_cap_v, cap_s);
                }
                j_1 += 2 * t;
            }

            t *= 2;

            m /= 2;
        }

        for a_j in a.iter_mut() {
            *a_j = self.mul_reduce(*a_j, self.n_inv);
        }
    }

    #[inline]
    fn mul_reduce(&self, a: u64, b: u64) -> u64 {
        let tmp = u128_to_u64(a as u128 * b as u128);

        let high = tmp.0 >> 32;
        let middle = tmp.0 & 0xffffffff;

        self.reduce_159(tmp.1, middle, high)
    }

    fn with_precomputes(mut self) -> Self {
        let psi_inv = mod_exp_u64(self.psi, self.q - 2, self.q);

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

        self
    }
}

/// Converts a u128 bit values into two u64 values, given as `(low, high)`
#[inline]
pub fn u128_to_u64(a: u128) -> (u64, u64) {
    let raw = a.to_be_bytes();

    let mut low = [0u8; 8];
    low.copy_from_slice(&raw[0..8]);
    let mut high = [0u8; 8];
    high.copy_from_slice(&raw[8..16]);

    (u64::from_be_bytes(low), u64::from_be_bytes(high))
}

/// Converts two u64 values (low, high) into one u128 values
#[inline]
pub fn u64_to_u128(low: u64, high: u64) -> u128 {
    let mut raw = [0u8; 16];

    raw[0..8].copy_from_slice(&low.to_be_bytes());
    raw[8..16].copy_from_slice(&high.to_be_bytes());

    u128::from_be_bytes(raw)
}

#[cfg(test)]
mod tests {
    use crate::dft::ntt_goldilock::u64_to_u128;

    use super::{u128_to_u64, TableGoldilock};

    #[test]
    fn u128_u64_conversion() {
        let originals = [1u128, 2u128.pow(64), 2u128.pow(64) + 2];

        for test_data in originals {
            let tmp = u128_to_u64(test_data);
            assert_eq!(u64_to_u128(tmp.0, tmp.1), test_data);
        }
    }

    #[test]
    fn mul_reduce_159() {
        let table = TableGoldilock::new();

        // naive multiplication and division
        let naive = |a: u64, b: u64| ((a as u128 * b as u128) % table.q as u128) as u64;

        let test_data = [1u64, 0x57ef77bbd0411810];

        for i in 0..test_data.len() {
            for j in 0..test_data.len() {
                let fast_res = table.mul_reduce(test_data[i], test_data[j]);

                let naively_computed = naive(test_data[i], test_data[j]);

                assert_eq!(fast_res, naively_computed);
            }
        }
    }
}
