use rand::{self, Rng};

use crate::dft::DFT;
use crate::utils::bit_reverse;
use std::arch::x86_64::*;

// prime to be used in the 51-bit setting
// TODO: make mulmod compatible with values > 32 bits, e.g. use Karatsuba multiplication
// pub const DOUBLE_PRIME: f64 = 4503599626321921.0;

/// NTT-friendly prime with 2^20 roots of unity
pub const U32_PRIME: i64 = 4293918721;

pub struct TableAVX2 {
    /// The NTT modulus. This implementation support primes with <=32 bits
    pub q: i64,
    /// N-th root of unity
    pub psi: i64,
    /// Which root of unity psi is
    n: usize,
    k: i64,
    k_vec: __m256i,
    r_square: __m256i,
    // ------ internals, mostly caching --------
    /// inverse of n, in AVX2 struct
    n_inv: __m256i,
    /// 2^32 in AVX2 struct, used for `a * b mod q` computation
    power: __m256d,
    /// q as AVX2 integer vector
    q_vec: __m256i,
    /// q as AVX2 float vector
    q_vec_float: __m256d,
    /// powers of psi in byte order
    powers_psi_bo: Vec<i64>,
    /// powers of inv_psi in byte order
    powers_psi_inv_bo: Vec<i64>,
}

impl DFT<i64> for TableAVX2 {
    /// NTT forward routine
    ///
    /// - `a`: vector with each element in range `[0, q)`
    fn forward_inplace(&self, a: &mut [i64]) {
        self.forward_inplace_core::<false>(a)
    }

    /// NTT forward lazy routine
    ///
    /// - `a`: vector with each element in range `[0, 2q)`
    fn forward_inplace_lazy(&self, a: &mut [i64]) {
        self.forward_inplace_core::<true>(a)
    }

    /// NTT backward routine
    ///
    /// - `a`: vector with each element in range `[0, q)`
    fn backward_inplace(&self, a: &mut [i64]) {
        self.backward_inplace_core::<false>(a)
    }

    /// NTT backward lazy routine
    ///
    /// - `a`: vector with each element in range `[0, 2q)`
    fn backward_inplace_lazy(&self, a: &mut [i64]) {
        self.backward_inplace_core::<true>(a)
    }
}

impl TableAVX2 {
    pub fn new() -> Self {
        let n = 2usize.pow(16);
        let mut res = Self {
            q: U32_PRIME,
            power: unsafe { _mm256_set1_pd(2.0f64.powi(32)) },
            q_vec: unsafe { _mm256_set1_epi64x(U32_PRIME as i64) },
            q_vec_float: unsafe { _mm256_set1_pd(U32_PRIME as f64) },
            psi: 2004365341,
            n,
            k: 1048577,
            r_square: unsafe { _mm256_set1_epi64x((2u128.pow(64) % U32_PRIME as u128) as i64) },
            k_vec: unsafe { _mm256_set1_epi64x(1048577) },
            n_inv: unsafe { _mm256_set1_epi64x(65536) },
            // -- precomputed powers of psi
            powers_psi_bo: Vec::with_capacity(n),
            // -- precomputed powers of inv_psi
            powers_psi_inv_bo: Vec::with_capacity(n),
        };

        println!("n_inv {}", res.to_montgomery(4293853201));

        res.with_precomputes();

        res
    }

    /// Finds a nth root of unity. Can be computed once and then cached.
    pub fn find_nth_unity_root(&self, n: i64, m: i64) -> i64 {
        let mut rand = rand::rng();

        let mut tmp;
        loop {
            tmp = rand.random_range(2..m);

            let g = self.mod_exp(tmp, (m - 1) / n);

            match self.mod_exp(g, n / 2) {
                1 => continue,
                _ => break g,
            }
        }
    }

    /// Modular exponentiation, i.e. `base^exp mod q`
    fn mod_exp(&self, base: i64, mut exp: i64) -> i64 {
        let mut out = 1;

        let mut acc = base;

        while exp > 0 {
            if exp % 2 == 1 {
                out = ((out as i128 * acc as i128) % self.q as i128) as i64;
            }

            acc = ((acc as i128 * acc as i128) % self.q as i128) as i64;

            exp >>= 1;
        }

        out
    }

    /// Computes the forward NTT using AVX2
    pub fn forward_inplace_core<const LAZY: bool>(&self, a: &mut [i64]) {
        let a_len = a.len();

        let mut t = a_len;
        let mut m = 1;

        unsafe { self.to_montgomery_vec(a) };

        loop {
            if m >= a_len {
                break;
            }

            t /= 2;

            if t == 1 && m >= 4 {
                let mut i = 0;

                loop {
                    if i >= m {
                        break;
                    }

                    unsafe {
                        let mut a_j = _mm256_setr_epi64x(
                            a[2 * i],
                            a[2 * (i + 1)],
                            a[2 * (i + 2)],
                            a[2 * (i + 3)],
                        );

                        let mut a_j_t = _mm256_setr_epi64x(
                            a[2 * i + 1],
                            a[2 * (i + 1) + 1],
                            a[2 * (i + 2) + 1],
                            a[2 * (i + 3) + 1],
                        );

                        self.ntt_kernel_4(
                            &_mm256_setr_epi64x(
                                self.to_montgomery(self.powers_psi_bo[m + i]),
                                self.to_montgomery(self.powers_psi_bo[m + i + 1]),
                                self.to_montgomery(self.powers_psi_bo[m + i + 2]),
                                self.to_montgomery(self.powers_psi_bo[m + i + 3]),
                            ),
                            &mut a_j,
                            &mut a_j_t,
                        );

                        a[2 * i] = _mm256_extract_epi64::<0>(a_j);
                        a[2 * (i + 1)] = _mm256_extract_epi64::<1>(a_j);
                        a[2 * (i + 2)] = _mm256_extract_epi64::<2>(a_j);
                        a[2 * (i + 3)] = _mm256_extract_epi64::<3>(a_j);

                        a[2 * i + 1] = _mm256_extract_epi64::<0>(a_j_t);
                        a[2 * (i + 1) + 1] = _mm256_extract_epi64::<1>(a_j_t);
                        a[2 * (i + 2) + 1] = _mm256_extract_epi64::<2>(a_j_t);
                        a[2 * (i + 3) + 1] = _mm256_extract_epi64::<3>(a_j_t)
                    }

                    i += 4;
                }
            } else if t == 2 && m >= 4 {
                let mut i = 0;

                loop {
                    if i >= m {
                        break;
                    }

                    let j_i1 = 2 * i * 2;
                    let j_i1_t = j_i1 + 2;

                    let j_i2 = 2 * (i + 1) * 2;
                    let j_i2_t = j_i2 + 2;

                    unsafe {
                        let mut a_j =
                            _mm256_setr_epi64x(a[j_i1], a[j_i1 + 1], a[j_i2], a[j_i2 + 1]);

                        let mut a_j_t =
                            _mm256_setr_epi64x(a[j_i1_t], a[j_i1_t + 1], a[j_i2_t], a[j_i2_t + 1]);

                        self.ntt_kernel_4(
                            &_mm256_setr_epi64x(
                                self.to_montgomery(self.powers_psi_bo[m + i]),
                                self.to_montgomery(self.powers_psi_bo[m + i]),
                                self.to_montgomery(self.powers_psi_bo[m + i + 1]),
                                self.to_montgomery(self.powers_psi_bo[m + i + 1]),
                            ),
                            &mut a_j,
                            &mut a_j_t,
                        );

                        a[j_i1] = _mm256_extract_epi64::<0>(a_j);
                        a[j_i1 + 1] = _mm256_extract_epi64::<1>(a_j);
                        a[j_i2] = _mm256_extract_epi64::<2>(a_j);
                        a[j_i2 + 1] = _mm256_extract_epi64::<3>(a_j);

                        a[j_i1_t] = _mm256_extract_epi64::<0>(a_j_t);
                        a[j_i1_t + 1] = _mm256_extract_epi64::<1>(a_j_t);
                        a[j_i2_t] = _mm256_extract_epi64::<2>(a_j_t);
                        a[j_i2_t + 1] = _mm256_extract_epi64::<3>(a_j_t);
                    }

                    i += 2;
                }
            } else {
                for i in 0..m {
                    let j_1 = 2 * i * t;
                    let j_2 = j_1 + t - 1;

                    let cap_s = self.powers_psi_bo[m + i];

                    for mut j in j_1..=j_2 {
                        loop {
                            if t < 4 || j >= j_2 {
                                break;
                            }

                            unsafe {
                                let mut a_j =
                                    _mm256_setr_epi64x(a[j], a[j + 1], a[j + 2], a[j + 3]);

                                let mut a_j_t = _mm256_setr_epi64x(
                                    a[j + t],
                                    a[j + t + 1],
                                    a[j + t + 2],
                                    a[j + t + 3],
                                );

                                self.ntt_kernel_4(
                                    &_mm256_set1_epi64x(self.to_montgomery(cap_s)),
                                    &mut a_j,
                                    &mut a_j_t,
                                );

                                a[j] = _mm256_extract_epi64::<0>(a_j);
                                a[j + 1] = _mm256_extract_epi64::<1>(a_j);
                                a[j + 2] = _mm256_extract_epi64::<2>(a_j);
                                a[j + 3] = _mm256_extract_epi64::<3>(a_j);

                                a[j + t] = _mm256_extract_epi64::<0>(a_j_t);
                                a[j + t + 1] = _mm256_extract_epi64::<1>(a_j_t);
                                a[j + t + 2] = _mm256_extract_epi64::<2>(a_j_t);
                                a[j + t + 3] = _mm256_extract_epi64::<3>(a_j_t);
                            };

                            j += 4;
                            continue;
                        }

                        if j > j_2 {
                            break;
                        }

                        // only used for uncommon values of n
                        let (a_j, a_j_t) = self.ntt_kernel_1(cap_s, a[j], a[j + t]);

                        a[j] = a_j;
                        a[j + t] = a_j_t;
                    }
                }
            }

            m *= 2;
        }

        unsafe {
            self.montgomery_reduce_vec(a);
        }
    }

    pub fn backward_inplace_core<const LAZY: bool>(&self, a: &mut [i64]) {
        let a_len = a.len();

        let mut t = 1;
        let mut m = a_len;

        unsafe { self.to_montgomery_vec(a) };

        loop {
            if m == 1 {
                break;
            }

            let mut j_1 = 0;
            let h = m / 2;

            if t == 1 && h >= 4 {
                let mut i = 0;

                loop {
                    if i >= h {
                        break;
                    }

                    unsafe {
                        let mut a_j = _mm256_setr_epi64x(
                            a[2 * i],
                            a[2 * (i + 1)],
                            a[2 * (i + 2)],
                            a[2 * (i + 3)],
                        );

                        let mut a_j_t = _mm256_setr_epi64x(
                            a[2 * i + 1],
                            a[2 * (i + 1) + 1],
                            a[2 * (i + 2) + 1],
                            a[2 * (i + 3) + 1],
                        );

                        self.intt_kernel_4(
                            &_mm256_setr_epi64x(
                                self.to_montgomery(self.powers_psi_inv_bo[h + i]),
                                self.to_montgomery(self.powers_psi_inv_bo[h + i + 1]),
                                self.to_montgomery(self.powers_psi_inv_bo[h + i + 2]),
                                self.to_montgomery(self.powers_psi_inv_bo[h + i + 3]),
                            ),
                            &mut a_j,
                            &mut a_j_t,
                        );

                        a[2 * i] = _mm256_extract_epi64::<0>(a_j);
                        a[2 * (i + 1)] = _mm256_extract_epi64::<1>(a_j);
                        a[2 * (i + 2)] = _mm256_extract_epi64::<2>(a_j);
                        a[2 * (i + 3)] = _mm256_extract_epi64::<3>(a_j);

                        a[2 * i + 1] = _mm256_extract_epi64::<0>(a_j_t);
                        a[2 * (i + 1) + 1] = _mm256_extract_epi64::<1>(a_j_t);
                        a[2 * (i + 2) + 1] = _mm256_extract_epi64::<2>(a_j_t);
                        a[2 * (i + 3) + 1] = _mm256_extract_epi64::<3>(a_j_t);
                    }

                    i += 4;
                }
            } else if t == 2 && h >= 4 {
                let mut i = 0;

                loop {
                    if i >= h {
                        break;
                    }

                    let j_i1 = 2 * i * 2;
                    let j_i1_t = j_i1 + 2;

                    let j_i2 = 2 * (i + 1) * 2;
                    let j_i2_t = j_i2 + 2;

                    unsafe {
                        let mut a_j =
                            _mm256_setr_epi64x(a[j_i1], a[j_i1 + 1], a[j_i2], a[j_i2 + 1]);

                        let mut a_j_t =
                            _mm256_setr_epi64x(a[j_i1_t], a[j_i1_t + 1], a[j_i2_t], a[j_i2_t + 1]);

                        self.intt_kernel_4(
                            &_mm256_setr_epi64x(
                                self.to_montgomery(self.powers_psi_inv_bo[h + i]),
                                self.to_montgomery(self.powers_psi_inv_bo[h + i]),
                                self.to_montgomery(self.powers_psi_inv_bo[h + i + 1]),
                                self.to_montgomery(self.powers_psi_inv_bo[h + i + 1]),
                            ),
                            &mut a_j,
                            &mut a_j_t,
                        );

                        a[j_i1] = _mm256_extract_epi64::<0>(a_j);
                        a[j_i1 + 1] = _mm256_extract_epi64::<1>(a_j);
                        a[j_i2] = _mm256_extract_epi64::<2>(a_j);
                        a[j_i2 + 1] = _mm256_extract_epi64::<3>(a_j);

                        a[j_i1_t] = _mm256_extract_epi64::<0>(a_j_t);
                        a[j_i1_t + 1] = _mm256_extract_epi64::<1>(a_j_t);
                        a[j_i2_t] = _mm256_extract_epi64::<2>(a_j_t);
                        a[j_i2_t + 1] = _mm256_extract_epi64::<3>(a_j_t);
                    }

                    i += 2;
                }
            } else {
                for i in 0..h {
                    let j_2 = j_1 + t - 1;
                    let cap_s = self.powers_psi_inv_bo[h + i];

                    for mut j in j_1..=j_2 {
                        loop {
                            if t < 4 || j >= j_2 {
                                break;
                            }

                            unsafe {
                                let mut a_j =
                                    _mm256_setr_epi64x(a[j], a[j + 1], a[j + 2], a[j + 3]);

                                let mut a_j_t = _mm256_setr_epi64x(
                                    a[j + t],
                                    a[j + t + 1],
                                    a[j + t + 2],
                                    a[j + t + 3],
                                );

                                self.intt_kernel_4(
                                    &_mm256_set1_epi64x(self.to_montgomery(cap_s)),
                                    &mut a_j,
                                    &mut a_j_t,
                                );

                                a[j] = _mm256_extract_epi64::<0>(a_j);
                                a[j + 1] = _mm256_extract_epi64::<1>(a_j);
                                a[j + 2] = _mm256_extract_epi64::<2>(a_j);
                                a[j + 3] = _mm256_extract_epi64::<3>(a_j);

                                a[j + t] = _mm256_extract_epi64::<0>(a_j_t);
                                a[j + t + 1] = _mm256_extract_epi64::<1>(a_j_t);
                                a[j + t + 2] = _mm256_extract_epi64::<2>(a_j_t);
                                a[j + t + 3] = _mm256_extract_epi64::<3>(a_j_t);
                            };

                            j += 4;
                            continue;
                        }

                        if j > j_2 {
                            break;
                        }

                        let (a_j, a_j_t) = self.intt_kernel_1(cap_s, a[j], a[j + t]);

                        a[j] = a_j;
                        a[j + t] = a_j_t;
                    }
                    j_1 += 2 * t;
                }
            }

            t *= 2;

            m /= 2;
        }

        let mut j = 0;
        unsafe {
            loop {
                if j >= a_len {
                    break;
                }

                let a_j = _mm256_setr_epi64x(a[j], a[j + 1], a[j + 2], a[j + 3]);
                let res = self.montgomery_reduce_32(self.montgomery_mul_vec(&a_j, &self.n_inv));
                a[j] = _mm256_extract_epi64::<0>(res);
                a[j + 1] = _mm256_extract_epi64::<1>(res);
                a[j + 2] = _mm256_extract_epi64::<2>(res);
                a[j + 3] = _mm256_extract_epi64::<3>(res);

                j += 4;
            }
        }
    }

    /// Computes the CT butterfly
    fn ntt_kernel_1(&self, cap_s: i64, mut a_j: i64, mut a_j_t: i64) -> (i64, i64) {
        // classic
        let cap_u = a_j;
        let cap_v = self.mul_reduce(a_j_t, cap_s);

        // println!("correct cap_v {cap_v}");
        let mut cap_u_add_cap_v = cap_u + cap_v;
        if cap_u_add_cap_v >= self.q {
            cap_u_add_cap_v -= self.q;
        }
        a_j = cap_u_add_cap_v;

        let mut cap_u_sub_cap_v = cap_u - cap_v;
        if cap_u_sub_cap_v < 0 {
            cap_u_sub_cap_v += self.q;
        }

        a_j_t = cap_u_sub_cap_v;

        (a_j, a_j_t)
    }

    fn intt_kernel_1(&self, cap_s: i64, mut a_j: i64, mut a_j_t: i64) -> (i64, i64) {
        let cap_u = a_j;
        let cap_v = a_j_t;

        let mut cap_u_add_cap_v = cap_u + cap_v;
        if cap_u_add_cap_v >= self.q {
            cap_u_add_cap_v -= self.q;
        }

        a_j = cap_u_add_cap_v;

        let mut cap_u_sub_cap_v = cap_u - cap_v;
        if cap_u_sub_cap_v < 0 {
            cap_u_sub_cap_v += self.q;
        }

        a_j_t = self.mul_reduce(cap_u_sub_cap_v, cap_s);

        (a_j, a_j_t)
    }

    /// Computes the CT butterfly using AVX2.
    #[inline]
    unsafe fn ntt_kernel_4(&self, cap_s: &__m256i, a_j: &mut __m256i, a_j_t: &mut __m256i) {
        let cap_u = *a_j;

        let cap_v = self.montgomery_mul_vec(a_j_t, cap_s);

        let mut cap_u_add_cap_v = _mm256_add_epi64(cap_u, cap_v);

        self.reduce_if_greater_equal_q(&mut cap_u_add_cap_v);

        *a_j = cap_u_add_cap_v;

        let mut cap_u_sub_cap_v = _mm256_sub_epi64(cap_u, cap_v);
        self.reduce_if_negative(&mut cap_u_sub_cap_v);

        *a_j_t = cap_u_sub_cap_v;
    }

    /// Computes the GS butterly using AVX2
    #[inline]
    unsafe fn intt_kernel_4(&self, cap_s: &__m256i, a_j: &mut __m256i, a_j_t: &mut __m256i) {
        let cap_u = *a_j;
        let cap_v = *a_j_t;

        let mut cap_u_add_cap_v = _mm256_add_epi64(cap_u, cap_v);
        self.reduce_if_greater_equal_q(&mut cap_u_add_cap_v);

        *a_j = cap_u_add_cap_v;

        let mut cap_u_sub_cap_v = _mm256_sub_epi64(cap_u, cap_v);
        self.reduce_if_negative(&mut cap_u_sub_cap_v);

        *a_j_t = self.montgomery_mul_vec(&cap_u_sub_cap_v, cap_s);
    }

    /// Computes `a * b mod q` assuming `a` and `b` are in Montgomery form
    #[inline]
    unsafe fn montgomery_mul_vec(&self, a: &__m256i, b: &__m256i) -> __m256i {
        // product = (u64) a * b
        let product = _mm256_mul_epu32(*a, *b);

        self.montgomery_reduce_32(product)
    }

    /// Computes `a mod q` for the case that `a >= q`, i.e. subtracting `q` from it using AVX2
    #[inline]
    unsafe fn reduce_if_greater_equal_q(&self, a: &mut __m256i) {
        // a > q ? a-q : a
        let masked = _mm256_and_si256(_mm256_cmpgt_epi64(*a, self.q_vec), self.q_vec);
        *a = _mm256_sub_epi64(*a, masked);

        // a == q ? 0 : a
        let masked = _mm256_and_si256(_mm256_cmpeq_epi64(*a, self.q_vec), self.q_vec);
        *a = _mm256_sub_epi64(*a, masked);
    }

    /// Computes `a mod q` for the case that `a < 0`, i.e. adding `q` to it using AVX2
    #[inline]
    unsafe fn reduce_if_negative(&self, a: &mut __m256i) {
        // a < 0 ? q : 0
        let masked = _mm256_and_si256(self.q_vec, _mm256_cmpgt_epi64(_mm256_setzero_si256(), *a));
        // println!("reduce if negative {masked:?}");
        *a = _mm256_add_epi64(*a, masked);
    }

    /// Adds all kind of precomputes, mainly storing precomputed values batched in AVX2-compatible structs
    fn with_precomputes(&mut self) {
        let psi_inv = self.mod_exp(self.psi, self.q - 2);

        let mut tmp_psi = 1;
        let mut tmp_psi_inv = 1;

        let mut tmp_powers_of_psi_bo = Vec::<i64>::with_capacity(self.n);
        let mut tmp_powers_of_psi_inv_bo = Vec::<i64>::with_capacity(self.n);

        tmp_powers_of_psi_bo.resize(self.n, 0);
        tmp_powers_of_psi_inv_bo.resize(self.n, 0);

        let log_n = self.n.ilog2();

        for i in 0..self.n {
            let reversed_index = bit_reverse(i as u64, log_n) as usize;
            tmp_powers_of_psi_bo[reversed_index] = tmp_psi;
            tmp_powers_of_psi_inv_bo[reversed_index] = tmp_psi_inv;

            tmp_psi = self.mul_reduce(tmp_psi, self.psi);
            tmp_psi_inv = self.mul_reduce(tmp_psi_inv, psi_inv);
        }

        for i in 0..self.n {
            self.powers_psi_bo.push(tmp_powers_of_psi_bo[i]);
            self.powers_psi_inv_bo.push(tmp_powers_of_psi_inv_bo[i]);
        }
    }

    /// Computes a * b mod self.q
    #[inline]
    fn mul_reduce(&self, a: i64, b: i64) -> i64 {
        ((a as u64 * b as u64) % self.q as u64) as i64
    }

    #[inline]
    unsafe fn montgomery_reduce_32(&self, mut t: __m256i) -> __m256i {
        // m = (t mod 2^32) * k mod 2^32
        let m = _mm256_and_si256(
            _mm256_set1_epi64x(0xffffffff),
            _mm256_mul_epu32(t, self.k_vec),
        );

        // m_n = (u64) m * self.q
        let m_n = _mm256_mul_epu32(m, self.q_vec);

        // y = (t - m_n) / 2^32
        let y = _mm256_srli_epi64::<32>(_mm256_sub_epi64(t, m_n));

        // mask = (u64) x < (u64) m
        let mask = _mm256_and_si256(a_le_b(t, m_n), self.q_vec);

        // t = t < m_n ? t + self.q : t
        t = _mm256_add_epi64(mask, y);

        // t = t mod 2^32
        _mm256_and_si256(_mm256_set1_epi64x(0xffffffff), t)
    }

    unsafe fn montgomery_reduce_vec(&self, a: &mut [i64]) {
        let a_len = a.len();

        let mut j = 0;
        unsafe {
            loop {
                if j >= a_len {
                    break;
                }

                let res = self.montgomery_reduce_32(_mm256_setr_epi64x(
                    a[j],
                    a[j + 1],
                    a[j + 2],
                    a[j + 3],
                ));

                a[j] = _mm256_extract_epi64::<0>(res);
                a[j + 1] = _mm256_extract_epi64::<1>(res);
                a[j + 2] = _mm256_extract_epi64::<2>(res);
                a[j + 3] = _mm256_extract_epi64::<3>(res);

                j += 4;
            }
        }
    }

    fn montgomery_reduce(&self, t: i64) -> i64 {
        let m = ((t & 0xffffffff) as u64 * self.k as u64) & 0xffffffff;
        let m_n = self.q as u64 * m;
        println!("m_n {m_n}");
        let mut res = (match (m_n as u64).overflowing_add(t as u64) {
            (res, true) => {
                println!("overflow added {}", res >> 32 | 1 << 33);

                res >> 32 | 1 << 33
            }
            (res, false) => res >> 32,
        }) as i64;

        if res >= self.q {
            res -= self.q
        }

        res as i64
    }

    fn to_montgomery(&self, a: i64) -> i64 {
        ((a as u64 * 2u64.pow(32)) % self.q as u64) as i64
    }

    unsafe fn to_montgomery_vec(&self, a: &mut [i64]) {
        let a_len = a.len();

        let mut j = 0;
        unsafe {
            loop {
                if j >= a_len {
                    break;
                }

                let res = self.montgomery_reduce_32(_mm256_mul_epu32(
                    self.r_square,
                    _mm256_setr_epi64x(a[j], a[j + 1], a[j + 2], a[j + 3]),
                ));

                a[j] = _mm256_extract_epi64::<0>(res);
                a[j + 1] = _mm256_extract_epi64::<1>(res);
                a[j + 2] = _mm256_extract_epi64::<2>(res);
                a[j + 3] = _mm256_extract_epi64::<3>(res);

                j += 4;
            }
        }
    }
}

/// Takes a AVX2 64-bit register and extracts its values to a unsigned integer array (`[u64; 4]`)
#[inline]
pub fn extract_m256i_to_i64(a: &__m256i) -> [i64; 4] {
    let mut res = [0i64; 4];
    unsafe {
        res[0] = _mm256_extract_epi64::<0>(*a);
        res[1] = _mm256_extract_epi64::<1>(*a);
        res[2] = _mm256_extract_epi64::<2>(*a);
        res[3] = _mm256_extract_epi64::<3>(*a);
    }

    res
}

/// Takes a AVX2 float (double precision) and extracts its values to a double array (`[f64; 4]`).
#[inline]
pub fn extract_m256d_to_f64(a: &__m256d) -> [f64; 4] {
    let mut res = [0.0f64; 4];
    unsafe { _mm256_storeu_pd(res.as_mut_ptr(), *a) };

    res
}

/// Converts AVX2 floats (double precision) to AVX2 32-bit integers stored in AVX2 64-bit register, very handy for testing.
///
/// The 32-bit integers must be arranged as `(0, a, 0, b, 0, c, 0, d)`.
#[inline]
pub fn m256d_to_m256i(a: &__m256d) -> __m256i {
    unsafe {
        let packed = _mm256_cvtpd_epi32(*a);

        let unpacked_raw = _mm256_set_m128i(
            _mm_shuffle_epi32::<0b11_00_10_00>(packed),
            _mm_shuffle_epi32::<0b01_00_00_00>(packed),
        );

        _mm256_srli_epi64(unpacked_raw, 32)
    }
}

/// Converts AVX2 32-bit integers to AVX2 floats (double precision), very handy for testing
#[inline]
pub fn m256i_to_m256d(a: &__m256i) -> __m256d {
    unsafe {
        let shuffled = _mm_or_si128(
            _mm256_extracti128_si256(_mm256_shuffle_epi32::<0b10_00_11_01>(*a), 1),
            _mm256_castsi256_si128(_mm256_shuffle_epi32::<0b11_01_10_00>(*a)),
        );

        _mm256_cvtepi32_pd(shuffled)
    }
}

pub unsafe fn a_add_b_overflow(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    let sum = _mm256_add_epi64(a, b);

    // a < 0 && sum >= 0
    // b < 0 && sum >= 0
    //
    // a < 0 && b < 0 && sum < a
    // a < 0 && b < 0 && sum < b
    //
    let a_lt_0 = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a);
    let b_lt_0 = _mm256_cmpgt_epi64(_mm256_setzero_si256(), b);

    let sum_ge_0 = _mm256_or_si256(
        _mm256_cmpeq_epi64(sum, _mm256_setzero_si256()),
        _mm256_cmpgt_epi64(sum, _mm256_setzero_si256()),
    );

    let a_lt_0_and_sum_ge_0 = _mm256_and_si256(a_lt_0, sum_ge_0);
    let b_lt_0_and_sum_ge_0 = _mm256_and_si256(b_lt_0, sum_ge_0);

    let a_lt_0_and_b_lt_0 = _mm256_and_si256(a_lt_0, a_lt_0);

    let sum_lt_a = _mm256_cmpgt_epi64(a, sum);
    let sum_lt_b = _mm256_cmpgt_epi64(b, sum);

    let a_lt_0_and_b_lt_0_and_sum_lt_a = _mm256_and_si256(a_lt_0_and_b_lt_0, sum_lt_a);
    let a_lt_0_and_b_lt_0_and_sum_lt_b = _mm256_and_si256(a_lt_0_and_b_lt_0, sum_lt_b);

    let overflow_mask = _mm256_or_si256(
        _mm256_or_si256(a_lt_0_and_sum_ge_0, b_lt_0_and_sum_ge_0),
        _mm256_or_si256(
            a_lt_0_and_b_lt_0_and_sum_lt_a,
            a_lt_0_and_b_lt_0_and_sum_lt_b,
        ),
    );

    let overflow = _mm256_and_si256(_mm256_set1_epi64x(1), overflow_mask);

    (overflow, sum)
}

/// Compute `a < b ? 0xFF..FF : 0`, treating _signed_ integers as _unsigned_ integers
pub unsafe fn a_le_b(a: __m256i, b: __m256i) -> __m256i {
    let a_lt_0 = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a);
    let b_lt_0 = _mm256_cmpgt_epi64(_mm256_setzero_si256(), b);

    let a_lt_b = _mm256_cmpgt_epi64(b, a);
    let b_lt_a = _mm256_cmpgt_epi64(b, a);

    let first = _mm256_andnot_si256(_mm256_or_si256(a_lt_0, b_lt_0), a_lt_b);
    let second = _mm256_and_si256(_mm256_andnot_si256(a_lt_0, b_lt_0), _mm256_set1_epi64x(-1));
    let third = _mm256_and_si256(_mm256_andnot_si256(b_lt_0, a_lt_0), _mm256_setzero_si256());
    let fourth = _mm256_and_si256(_mm256_and_si256(a_lt_0, b_lt_0), b_lt_a);

    _mm256_or_si256(
        _mm256_or_si256(first, second),
        _mm256_or_si256(third, fourth),
    )
}

#[cfg(test)]
mod tests {
    use std::{arch::x86_64::*, i64};

    use crate::dft::ntt_avx2::a_le_b;

    use super::{
        a_add_b_overflow, extract_m256d_to_f64, extract_m256i_to_i64, m256d_to_m256i,
        m256i_to_m256d, TableAVX2, U32_PRIME,
    };

    #[test]
    fn extract_u64() {
        let data = unsafe { _mm256_setr_epi64x(1, 2, 3, 4) };

        let res = extract_m256i_to_i64(&data);

        assert_eq!(res, [1, 2, 3, 4]);
    }

    #[test]
    fn extract_f64() {
        let __m256d = unsafe { _mm256_setr_pd(1.0, 2.0, 3.0, 4.0) };
        let data = __m256d;

        let res = extract_m256d_to_f64(&data);

        assert_eq!(res, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn m256d_to_m256i_conversion() {
        let data = unsafe { _mm256_setr_pd(1.0, 2.0, 3.0, 4.0) };

        let res = m256d_to_m256i(&data);

        assert_eq!(extract_m256i_to_i64(&res), [1, 2, 3, 4]);
    }

    #[test]
    fn m256i_to_m256d_conversion() {
        let data = unsafe { _mm256_setr_epi64x(1, 2, 3, 4) };

        let res = m256i_to_m256d(&data);

        assert_eq!(extract_m256d_to_f64(&res), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn montgomery_step() {
        let table = TableAVX2::new();
        let r = 2u64.pow(32);

        let values: [i64; 2] = [
            1,
            // 23,
            // 1073741824,
            // 1073741824 * 1073741824,
            U32_PRIME - 1048321,
        ];

        println!("values {values:?}");
        // 3810847490
        for val in values {
            // let t = (val * r) % table.q as u64;

            println!("1 montgomery {}", table.to_montgomery(1));
            println!("{:?}", unsafe {
                table.montgomery_reduce_32(_mm256_set1_epi64x(table.to_montgomery(val) as i64))
            });
        }
    }

    #[test]
    fn reduce_if_negative() {
        let table = TableAVX2::new();
        let control = unsafe {
            let mut res = _mm256_setr_epi64x(-1, -2, 3, 0);

            table.reduce_if_negative(&mut res);

            let mut control = _mm256_setzero_si256();

            _mm256_store_si256(&mut control, res);

            control
        };

        assert_eq!(
            [(table.q - 1), (table.q - 2), 3, 0],
            extract_m256i_to_i64(&control)
        );
    }

    #[test]
    fn reduce_if_greate_equal_q() {
        let table = TableAVX2::new();
        let control = unsafe {
            let mut res = _mm256_setr_epi64x((table.q + 1) as i64, table.q as i64, -1, 0);

            table.reduce_if_greater_equal_q(&mut res);

            let mut control = _mm256_setzero_si256();

            _mm256_store_si256(&mut control, res);

            control
        };

        assert_eq!([1, 0, -1, 0], extract_m256i_to_i64(&control))
    }

    #[test]
    fn overflow_addition() {
        let (high, low) = unsafe {
            let a = _mm256_setr_epi64x(-1, -1, 1, i64::MIN);
            let b = _mm256_setr_epi64x(1, -1, -1, i64::MIN);

            a_add_b_overflow(a, b)
        };

        println!("high {high:?}, low {low:?}");
    }

    #[test]
    fn less_than() {
        let mask = unsafe {
            let a = _mm256_setr_epi64x(1, 1, -1, -2);
            let b = _mm256_setr_epi64x(2, -1, 1, -1);

            let c = _mm256_setr_epi64x(2, -1, 1, -1);
            let d = _mm256_setr_epi64x(1, 1, -1, -2);

            (a_le_b(a, b), a_le_b(c, d))
        };

        assert_eq!(extract_m256i_to_i64(&mask.0), [-1, -1, 0, -1]);
        assert_eq!(extract_m256i_to_i64(&mask.1), [0, 0, -1, 0]);
    }

    #[test]
    fn mul_reduce_vec() {
        let table = TableAVX2::new();

        let r = 2u64.pow(32);

        let to_montgomery = |a: i64| ((a as u64 * r) % table.q as u64) as i64;

        println!(
            "montgomery reduced {}",
            table.montgomery_reduce(-1383570113403652876)
        );
        let product: i64 = -1383570113403652876;
        println!("product {product}");
        let m = ((product as u64 % 2u64.pow(32)) * 4293918719) % 2u64.pow(32);
        println!("m {m}");
        let m_n = 4293918721 * m;
        println!("m_n {m_n}");
        let product_add_m_n = (m_n as i128 + product as i128) / 2i128.pow(32);
        println!("product_add_m_n {product_add_m_n}");
        let control = unsafe {
            let a = _mm256_setr_epi64x(
                to_montgomery(4119828181),
                to_montgomery(1439360284),
                to_montgomery(1073741824),
                -5376484460904719641,
            );
            let b = _mm256_setr_epi64x(
                to_montgomery(3770581993),
                to_montgomery(2870057844),
                to_montgomery(1073741824),
                1,
            );

            table.montgomery_reduce_32(table.montgomery_mul_vec(&a, &b))
        };

        assert_eq!(
            [
                1197697452,
                2614870421,
                table.mul_reduce(1073741824, 1073741824),
                table.mul_reduce(1073741824, 1073741824),
            ],
            extract_m256i_to_i64(&control)
        );

        println!("res {:?}", control);
    }

    #[test]
    fn ntt_kernel_4() {
        let table = TableAVX2::new();

        let (a_j, a_j_t) = unsafe {
            let mut a_j = _mm256_setr_epi64x(
                table.to_montgomery(2825332433),
                table.to_montgomery(3362019074),
                table.to_montgomery(2509869327),
                table.to_montgomery(1527549775),
            );
            let mut a_j_t = _mm256_setr_epi64x(
                table.to_montgomery(4119828181),
                table.to_montgomery(1439360284),
                table.to_montgomery(2754719283),
                table.to_montgomery(357587952),
            );
            let cap_s = _mm256_setr_epi64x(
                table.to_montgomery(3770581993),
                table.to_montgomery(2870057844),
                table.to_montgomery(3898594835),
                table.to_montgomery(2318024136),
            );

            table.ntt_kernel_4(&cap_s, &mut a_j, &mut a_j_t);

            (
                table.montgomery_reduce_32(a_j),
                table.montgomery_reduce_32(a_j_t),
            )
        };

        assert_eq!(
            [4023029885, 1682970774, 2185318912, 1901790419],
            extract_m256i_to_i64(&a_j)
        );
        assert_eq!(
            [1627634981, 747148653, 2834419742, 1153309131],
            extract_m256i_to_i64(&a_j_t)
        );
    }
}
