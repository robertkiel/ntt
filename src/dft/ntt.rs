use crate::utils::bit_reverse;
use crate::{dft::DFT, utils::mod_exp_u64};

pub struct Table<O> {
    /// NTT friendly prime modulus
    pub q: O,
    /// n-th root of unity
    pub psi: O,
    /// Which root of unity `psi` is
    n: O,
    /// Multiplicative inverse of n in Montgomery form
    n_inv_montgomery: O,
    /// Value such that `q * k + 2^bit_len(O) * r' = 1` for some integer r.
    /// Needed for Montgomery multiplication
    k: O,
    /// Precomputed value `( 2^bit_len(O) ^ 2 ) mod q`.
    /// Needed for quick Montgomery transformation
    r_square: O,
    /// Precomputed powers of psi in Montgomery form and stored in bit-reversed order
    powers_psi_bo: Vec<u64>,
    /// Precomputed powers of the multiplicative inverse of psi in Montgomery form
    /// and stored in bit-reversed order
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
    /// Instantiates NTT with base prime
    pub fn new() -> Self {
        let mut res = Self {
            q: 0x1fffffffffe00001u64,
            psi: 0x15eb043c7aa2b01fu64, //2^17th root of unity
            n: 2u64.pow(16),
            n_inv_montgomery: 281474976710656,
            k: 6917533425689690113,
            r_square: 0xfffff0000040,
            powers_psi_bo: Vec::with_capacity(2usize.pow(16)),
            powers_psi_inv_bo: Vec::with_capacity(2usize.pow(16)),
        };

        res.with_precomputes();

        res
    }

    /// Instantiates NTT with NTT-friendly prime that allows very
    /// fast reduction
    pub fn new_goldilock() -> Self {
        let mut res = Self {
            q: 0xffffffff00000001,
            psi: 0xabd0a6e8aa3d8a0e, //2^17th root of unity
            n: 2u64.pow(16),
            n_inv_montgomery: 281474976710656,
            k: 4294967297,
            r_square: 18446744065119617025,
            powers_psi_bo: Vec::with_capacity(2usize.pow(16)),
            powers_psi_inv_bo: Vec::with_capacity(2usize.pow(16)),
        };

        res.with_precomputes();

        res
    }

    /// Instantiates NTT with prime that can be used with 32-bit
    /// registers, i.e. on AVX2
    pub fn new_u32_compatible() -> Self {
        let mut res = Self {
            q: 4293918721,
            psi: 2004365341,
            n: 2u64.pow(16),
            n_inv_montgomery: 16711664,
            k: 1143915400569815041,
            r_square: 2564090464,
            powers_psi_bo: Vec::with_capacity(2usize.pow(16)),
            powers_psi_inv_bo: Vec::with_capacity(2usize.pow(16)),
        };

        res.with_precomputes();

        res
    }

    /// Instantiates NTT with small prime to check correctness of
    /// the implementation
    pub fn new_simple() -> Self {
        // Values taken from https://eprint.iacr.org/2024/585.pdf
        let mut res = Self {
            q: 7681,
            // 2^3th root of unity
            psi: 1925,
            n: 2u64.pow(2),
            n_inv_montgomery: 1391,
            k: 13079152223072805377,
            r_square: 3666,
            powers_psi_bo: Vec::with_capacity(2usize.pow(3)),
            powers_psi_inv_bo: Vec::with_capacity(2usize.pow(3)),
        };

        res.with_precomputes();

        res
    }

    pub fn forward_inplace_core<const LAZY: bool>(&self, a: &mut [u64]) {
        let a_len = a.len();

        for a_j in a.iter_mut() {
            *a_j = self.to_montgomery(*a_j);
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
                    let cap_v = self.montgomery_mul(a[j + t], cap_s);

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
            *a_j = self.montgomery_reduce(*a_j as u128)
        }
    }

    pub fn backward_inplace_core<const LAZY: bool>(&self, a: &mut [u64]) {
        let a_len = a.len();

        for a_j in a.iter_mut() {
            *a_j = self.to_montgomery(*a_j);
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

                    a[j + t] = self.montgomery_mul(cap_u_sub_cap_v, cap_s);
                }
                j_1 += 2 * t;
            }

            t *= 2;

            m /= 2;
        }

        for a_j in a.iter_mut() {
            *a_j = self.montgomery_reduce(self.montgomery_mul(*a_j, self.n_inv_montgomery) as u128);
        }
    }

    /// Montgomery reduction using R=2^64-1
    pub fn montgomery_reduce(&self, a: u128) -> u64 {
        // m = ((a mod 2^64) * k) mod 2^64
        let m = ((a & 0xffffffffffffffff) * self.k as u128) & 0xffffffffffffffff;

        let m_n = m * self.q as u128;

        let y = (a.overflowing_sub(m_n).0 >> 64) as u64;

        if a < m_n {
            y.overflowing_add(self.q).0
        } else {
            y
        }
    }

    /// Transforms a into Montgomery form using R=2^64-1
    pub fn to_montgomery(&self, a: u64) -> u64 {
        self.montgomery_reduce(a as u128 * self.r_square as u128)
    }

    /// Performs a Montgomery multiplication, assuming that both `a` and `b` and in Montgomery form
    #[inline]
    fn montgomery_mul(&self, a: u64, b: u64) -> u64 {
        self.montgomery_reduce(a as u128 * b as u128)
    }

    /// Computes and stores precomputable values, especially powers of `psi`.
    fn with_precomputes(&mut self) {
        let psi_inv_montgomery = self.to_montgomery(mod_exp_u64(self.psi, self.q - 2, self.q));
        let psi_montgomery = self.to_montgomery(self.psi);

        let mut tmp_psi = self.to_montgomery(1);
        let mut tmp_psi_inv = self.to_montgomery(1);

        self.powers_psi_bo.resize(self.n as usize, 0);
        self.powers_psi_inv_bo.resize(self.n as usize, 0);

        let log_n = self.n.ilog2();

        for i in 0..self.n {
            let reversed_index = bit_reverse(i, log_n) as usize;
            self.powers_psi_bo[reversed_index] = tmp_psi;
            self.powers_psi_inv_bo[reversed_index] = tmp_psi_inv;

            tmp_psi = self.montgomery_mul(tmp_psi, psi_montgomery);
            tmp_psi_inv = self.montgomery_mul(tmp_psi_inv, psi_inv_montgomery);
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::{dft::DFT, utils::mod_exp_u64};

    use super::Table;

    #[test]
    fn test_mod_exp() {
        let table = Table::<u64>::new_simple();

        let res = mod_exp_u64(4, table.q - 2, table.q);

        assert_eq!(res, 5761);
    }

    #[test]
    fn montgomery_transformation() {
        let table = Table::<u64>::new();

        let values = [1u64, 16777208u64];

        for val in values {
            assert_eq!(
                val,
                table.montgomery_reduce(table.to_montgomery(val) as u128)
            );
        }
    }

    #[test]
    fn montgomery_transformation_goldilock() {
        let table = Table::<u64>::new_goldilock();

        let values = [1u64, 16777208u64];

        for val in values {
            assert_eq!(
                val,
                table.montgomery_reduce(table.to_montgomery(val) as u128)
            );
        }
    }

    #[test]
    fn montgomery_transformation_u32_compatible() {
        let table = Table::<u64>::new_u32_compatible();

        let values = [1u64, 16777208u64];

        for val in values {
            assert_eq!(
                val,
                table.montgomery_reduce(table.to_montgomery(val) as u128)
            );
        }
    }

    #[test]
    fn montgomery_transformation_simple() {
        let table = Table::<u64>::new_simple();

        let values = [1u64, 1904u64];

        for val in values {
            assert_eq!(
                val,
                table.montgomery_reduce(table.to_montgomery(val) as u128)
            );
        }
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
}
