//! NTT implementation using AVX2 instruction for enhanced performance

use crate::dft::DFT;
use crate::utils::{bit_reverse, egcd_i128, MontgomeryTransformer};
use std::arch::x86_64::*;

pub struct TableAVX2 {
    /// The NTT modulus. This implementation support primes with <= 61 bits
    pub q: i64,
    /// N-th root of unity
    pub psi: i64,
    /// Multiplicative inverse of psi in Montgomery form
    psi_inv_montgomery: i64,
    /// Which root of unity psi is
    n: usize,
    // ------ internals, mostly caching --------
    /// multiplicative inverse of n in Montgomery form
    n_inv_montgomery: i64,
    montgomery: MontgomeryTransformer<i64>,
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
        Self::from_prime_and_root(0x1fffffffffe00001, 0x15eb043c7aa2b01f, 2i64.pow(16))
    }

    /// Takes a modulus and a n-th root of unity and instatiates a struct that allows fast NTT and INTT operations
    ///
    /// ```rust
    /// use ntt::dft::{ntt_avx2::TableAVX2, DFT};
    ///
    /// const PRIME: i64 = 0x1fffffffffe00001;
    /// /// 2^17th root of unity
    /// const ROOT: i64 = 0x15eb043c7aa2b01f;
    /// const N: i64 = 2i64.pow(16);
    ///
    /// TableAVX2::from_prime_and_root(PRIME, ROOT, N);
    /// ```
    ///
    pub fn from_prime_and_root(modulus: i64, root: i64, n: i64) -> Self {
        let montgomery = MontgomeryTransformer::<i64>::from_modulus(modulus);

        let n_inv_raw = egcd_i128(n as i128, modulus as i128).1;

        let n_inv = if n_inv_raw < 0 {
            n_inv_raw as i64 + modulus
        } else {
            n_inv_raw as i64
        };

        let psi_inv_raw = egcd_i128(root as i128, modulus as i128).1;

        let psi_inv = if psi_inv_raw < 0 {
            (psi_inv_raw + modulus as i128) as i64
        } else {
            psi_inv_raw as i64
        };

        Self {
            q: modulus,
            psi: root,
            psi_inv_montgomery: montgomery.transform(psi_inv),
            n_inv_montgomery: montgomery.transform(n_inv),
            n: n as usize,
            montgomery,
            // montgomery,
            powers_psi_bo: Vec::with_capacity(n as usize),
            powers_psi_inv_bo: Vec::with_capacity(n as usize),
        }
        .with_precomputes()
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

                        self.cs_butterfly_simd_4(
                            &_mm256_setr_epi64x(
                                self.powers_psi_bo[m + i],
                                self.powers_psi_bo[m + i + 1],
                                self.powers_psi_bo[m + i + 2],
                                self.powers_psi_bo[m + i + 3],
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

                        self.cs_butterfly_simd_4(
                            &_mm256_setr_epi64x(
                                self.powers_psi_bo[m + i],
                                self.powers_psi_bo[m + i],
                                self.powers_psi_bo[m + i + 1],
                                self.powers_psi_bo[m + i + 1],
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

                                self.cs_butterfly_simd_4(
                                    &_mm256_set1_epi64x(cap_s),
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
                        let (a_j, a_j_t) = self.cs_butterfly(cap_s, a[j], a[j + t]);

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

                        self.gs_butterfly_simd_4(
                            &_mm256_setr_epi64x(
                                self.powers_psi_inv_bo[h + i],
                                self.powers_psi_inv_bo[h + i + 1],
                                self.powers_psi_inv_bo[h + i + 2],
                                self.powers_psi_inv_bo[h + i + 3],
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

                        self.gs_butterfly_simd_4(
                            &_mm256_setr_epi64x(
                                self.powers_psi_inv_bo[h + i],
                                self.powers_psi_inv_bo[h + i],
                                self.powers_psi_inv_bo[h + i + 1],
                                self.powers_psi_inv_bo[h + i + 1],
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

                                self.gs_butterfly_simd_4(
                                    &_mm256_set1_epi64x(cap_s),
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

                        let (a_j, a_j_t) = self.gs_butterfly(cap_s, a[j], a[j + t]);

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
                let res = self.montgomery_reduce_64(
                    _mm256_setzero_si256(),
                    self.montgomery_mul_vec(&a_j, &_mm256_set1_epi64x(self.n_inv_montgomery)),
                );
                a[j] = _mm256_extract_epi64::<0>(res);
                a[j + 1] = _mm256_extract_epi64::<1>(res);
                a[j + 2] = _mm256_extract_epi64::<2>(res);
                a[j + 3] = _mm256_extract_epi64::<3>(res);

                j += 4;
            }
        }
    }

    /// Computes the CT butterfly
    fn cs_butterfly(&self, cap_s: i64, mut a_j: i64, mut a_j_t: i64) -> (i64, i64) {
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

    fn gs_butterfly(&self, cap_s: i64, mut a_j: i64, mut a_j_t: i64) -> (i64, i64) {
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
    unsafe fn cs_butterfly_simd_4(&self, cap_s: &__m256i, a_j: &mut __m256i, a_j_t: &mut __m256i) {
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
    unsafe fn gs_butterfly_simd_4(&self, cap_s: &__m256i, a_j: &mut __m256i, a_j_t: &mut __m256i) {
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
        let (high, low) = a_mul_b_karatsuba(*a, *b);

        self.montgomery_reduce_64(high, low)
    }

    /// Computes `a mod q` for the case that `a >= q`, i.e. subtracting `q` from it using AVX2
    #[inline]
    unsafe fn reduce_if_greater_equal_q(&self, a: &mut __m256i) {
        // a > q ? a-q : a
        let masked = _mm256_and_si256(
            _mm256_cmpgt_epi64(*a, _mm256_set1_epi64x(self.q)),
            _mm256_set1_epi64x(self.q),
        );
        *a = _mm256_sub_epi64(*a, masked);

        // a == q ? 0 : a
        let masked = _mm256_and_si256(
            _mm256_cmpeq_epi64(*a, _mm256_set1_epi64x(self.q)),
            _mm256_set1_epi64x(self.q),
        );
        *a = _mm256_sub_epi64(*a, masked);
    }

    /// Computes `a mod q` for the case that `a < 0`, i.e. adding `q` to it using AVX2
    #[inline]
    unsafe fn reduce_if_negative(&self, a: &mut __m256i) {
        // a < 0 ? q : 0
        let masked = _mm256_and_si256(
            _mm256_set1_epi64x(self.q),
            _mm256_cmpgt_epi64(_mm256_setzero_si256(), *a),
        );
        // println!("reduce if negative {masked:?}");
        *a = _mm256_add_epi64(*a, masked);
    }

    /// Adds all kind of precomputes, mainly storing precomputed values batched in AVX2-compatible structs
    fn with_precomputes(mut self) -> Self {
        let psi_montgomery = self.montgomery.transform(self.psi);

        let mut tmp_psi = self.montgomery.transform(1);
        let mut tmp_psi_inv = self.montgomery.transform(1);

        self.powers_psi_bo.resize(self.n, 0);
        self.powers_psi_inv_bo.resize(self.n, 0);

        let log_n = self.n.ilog2();

        for i in 0..self.n {
            let reversed_index = bit_reverse(i as u64, log_n) as usize;
            self.powers_psi_bo[reversed_index] = tmp_psi;
            self.powers_psi_inv_bo[reversed_index] = tmp_psi_inv;

            tmp_psi = self.montgomery.mul(tmp_psi, psi_montgomery);
            tmp_psi_inv = self.montgomery.mul(tmp_psi_inv, self.psi_inv_montgomery);
        }

        self
    }

    /// Computes a * b mod self.q
    #[inline]
    fn mul_reduce(&self, a: i64, b: i64) -> i64 {
        ((a as u128 * b as u128) % self.q as u128) as i64
    }

    /// Computes Montgomery reduction using R=2^32-1
    #[inline]
    unsafe fn montgomery_reduce_64(&self, t_high: __m256i, mut t_low: __m256i) -> __m256i {
        // m = (t mod 2^32) * k mod 2^32
        let (_, m_low) = a_mul_b_karatsuba(t_low, _mm256_set1_epi64x(self.montgomery.k));

        // m_n = (u64) m * self.q
        let (m_n_high, m_n_low) = a_mul_b_karatsuba(m_low, _mm256_set1_epi64x(self.q));

        // y = (t - m_n) / 2^32
        let y = a_sub_b(t_high, t_low, m_n_high, m_n_low).0;

        // mask = (u64) x < (u64) m
        let mask = _mm256_and_si256(
            a_lt_b_wide(t_high, t_low, m_n_high, m_n_low),
            _mm256_set1_epi64x(self.q),
        );

        // t = t < m_n ? t + self.q : t
        t_low = _mm256_add_epi64(mask, y);

        // t = t mod 2^32
        t_low
    }

    /// Performs a Montgomery reduction on all values of the given vector.
    #[inline]
    unsafe fn montgomery_reduce_vec(&self, a: &mut [i64]) {
        let a_len = a.len();

        let mut j = 0;

        loop {
            if j >= a_len {
                break;
            }

            let res = self.montgomery_reduce_64(
                _mm256_setzero_si256(),
                _mm256_setr_epi64x(a[j], a[j + 1], a[j + 2], a[j + 3]),
            );

            a[j] = _mm256_extract_epi64::<0>(res);
            a[j + 1] = _mm256_extract_epi64::<1>(res);
            a[j + 2] = _mm256_extract_epi64::<2>(res);
            a[j + 3] = _mm256_extract_epi64::<3>(res);

            j += 4;
        }
    }

    /// Transforms the given value to Montgomery form
    #[inline]
    unsafe fn to_montgomery(&self, a: __m256i) -> __m256i {
        let (high, low) = a_mul_b_karatsuba(a, _mm256_set1_epi64x(self.montgomery.r_square));

        self.montgomery_reduce_64(high, low)
    }

    /// Transforms a vector into Montgomery form
    #[inline]
    unsafe fn to_montgomery_vec(&self, a: &mut [i64]) {
        let a_len = a.len();

        let mut j = 0;

        loop {
            if j >= a_len {
                break;
            }

            let res = self.to_montgomery(_mm256_setr_epi64x(a[j], a[j + 1], a[j + 2], a[j + 3]));

            a[j] = _mm256_extract_epi64::<0>(res);
            a[j + 1] = _mm256_extract_epi64::<1>(res);
            a[j + 2] = _mm256_extract_epi64::<2>(res);
            a[j + 3] = _mm256_extract_epi64::<3>(res);

            j += 4;
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

/// Computes `a < b ? 0xFF..FF : 0`, treating _signed_ integers as _unsigned_ integers
pub unsafe fn a_lt_b(a: __m256i, b: __m256i) -> __m256i {
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

/// Computes `a * b` using 32bit multiplication and the Karatsuba algorithm
pub unsafe fn a_mul_b_karatsuba(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    let a_low = _mm256_and_si256(_mm256_set1_epi64x(0xffffffff), a);
    let a_high = _mm256_srli_epi64::<32>(a);
    let b_low = _mm256_and_si256(_mm256_set1_epi64x(0xffffffff), b);
    let b_high = _mm256_srli_epi64::<32>(b);

    let mut high = _mm256_mul_epu32(a_high, b_high);
    let low = _mm256_mul_epu32(a_low, b_low);
    let low_low = _mm256_and_si256(_mm256_set1_epi64x(0xffffffff), low);
    let mut low_high = _mm256_srli_epi64::<32>(low);

    let middle_0 = _mm256_mul_epu32(a_high, b_low);
    let middle_1 = _mm256_mul_epu32(a_low, b_high);

    let middle_0_low = _mm256_and_si256(_mm256_set1_epi64x(0xffffffff), middle_0);
    let middle_0_high = _mm256_srli_epi64::<32>(middle_0);

    let middle_1_low = _mm256_and_si256(_mm256_set1_epi64x(0xffffffff), middle_1);
    let middle_1_high = _mm256_srli_epi64::<32>(middle_1);

    high = _mm256_add_epi64(high, middle_0_high);
    high = _mm256_add_epi64(high, middle_1_high);

    low_high = _mm256_add_epi64(low_high, middle_0_low);
    low_high = _mm256_add_epi64(low_high, middle_1_low);

    high = _mm256_add_epi64(high, _mm256_srli_epi64::<32>(low_high));

    (
        high,
        _mm256_or_si256(low_low, _mm256_slli_epi64::<32>(low_high)),
    )
}

/// Computes `a - b` for 128bit values
pub unsafe fn a_sub_b(
    a_high: __m256i,
    a_low: __m256i,
    b_high: __m256i,
    b_low: __m256i,
) -> (__m256i, __m256i) {
    let low = _mm256_sub_epi64(a_low, b_low);

    // b_low > a_low ? 0x01 : 0
    let low_carry = _mm256_and_si256(_mm256_set1_epi64x(1), a_lt_b(a_low, b_low));

    let high = _mm256_sub_epi64(_mm256_sub_epi64(a_high, low_carry), b_high);

    (high, low)
}

/// Computes `a < b ? 0xFF..FF : 0` for 128bit values
pub unsafe fn a_lt_b_wide(
    a_high: __m256i,
    a_low: __m256i,
    b_high: __m256i,
    b_low: __m256i,
) -> __m256i {
    let a_low_lt_b_low = a_lt_b(a_low, b_low);
    let a_high_lt_b_high = a_lt_b(a_high, b_high);
    let a_high_eq_b_high = _mm256_cmpeq_epi64(a_high, b_high);

    _mm256_or_si256(
        a_high_lt_b_high,
        _mm256_and_si256(a_high_eq_b_high, a_low_lt_b_low),
    )
}

#[cfg(test)]
mod tests {
    use std::{arch::x86_64::*, i64};

    use crate::dft::ntt_avx2::a_lt_b;

    use super::{
        a_mul_b_karatsuba, a_sub_b, extract_m256d_to_f64, extract_m256i_to_i64, TableAVX2,
    };

    #[test]
    fn a_sub_b_wide() {
        let (high, low) = unsafe {
            let a_high = _mm256_setr_epi64x(0, 0, 2, 2);
            let a_low = _mm256_setr_epi64x(0, 1, 2, 1);
            let b_high = _mm256_setr_epi64x(0, 0, 2, 0);
            let b_low = _mm256_setr_epi64x(0, 2, 1, 2);

            a_sub_b(a_high, a_low, b_high, b_low)
        };

        assert_eq!(extract_m256i_to_i64(&high), [0, -1, 0, 1]);
        assert_eq!(extract_m256i_to_i64(&low), [0, -1, 1, -1]);
    }

    #[test]
    fn karatsuba() {
        let (high, low) = unsafe {
            let a = _mm256_setr_epi64x(-4294967296, 4294967295, -1, -1);
            let b = _mm256_setr_epi64x(-4294967296, 4294967295, 4294967295, -1);

            a_mul_b_karatsuba(a, b)
        };

        assert_eq!(
            extract_m256i_to_i64(&high),
            [-8589934591, 0, 4294967294, -2]
        );
        assert_eq!(extract_m256i_to_i64(&low), [0, -8589934591, -4294967295, 1]);
    }
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
    fn montgomery_transformation() {
        let table = TableAVX2::new();

        let values: [i64; 2] = [1, 0x1fffffffffe00001 - 1048321];

        for val in values {
            let res = unsafe {
                let low = table.to_montgomery(_mm256_set1_epi64x(val));
                table.montgomery_reduce_64(_mm256_setzero_si256(), low)
            };

            assert_eq!(extract_m256i_to_i64(&res), [val, val, val, val]);
        }
    }

    #[test]
    fn reduce_if_negative() {
        let table = TableAVX2::new();
        let control = unsafe {
            let mut res = _mm256_setr_epi64x(-1, -2, 3, 0);

            table.reduce_if_negative(&mut res);

            res
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
    fn less_than() {
        let mask = unsafe {
            let a = _mm256_setr_epi64x(1, 1, -1, -2);
            let b = _mm256_setr_epi64x(2, -1, 1, -1);

            let c = _mm256_setr_epi64x(2, -1, 1, -1);
            let d = _mm256_setr_epi64x(1, 1, -1, -2);

            (a_lt_b(a, b), a_lt_b(c, d))
        };

        assert_eq!(extract_m256i_to_i64(&mask.0), [-1, -1, 0, -1]);
        assert_eq!(extract_m256i_to_i64(&mask.1), [0, 0, -1, 0]);
    }

    #[test]
    fn mul_reduce_vec() {
        let table = TableAVX2::new();

        let r = 2u64.pow(32);
        let to_montgomery = |a: i64| ((a as u64 * r) % table.q as u64) as i64;

        let control = unsafe {
            let a = _mm256_setr_epi64x(
                to_montgomery(4119828181),
                to_montgomery(1439360284),
                to_montgomery(1073741824),
                to_montgomery(1527549775),
            );
            let b = _mm256_setr_epi64x(
                to_montgomery(3770581993),
                to_montgomery(2870057844),
                to_montgomery(1073741824),
                to_montgomery(2318024136),
            );

            table.montgomery_reduce_64(_mm256_setzero_si256(), table.montgomery_mul_vec(&a, &b))
        };

        assert_eq!(
            [
                151105094258736720,
                37308892573532894,
                2161727821135872001,
                1644454774947165347
            ],
            extract_m256i_to_i64(&control)
        );
    }

    #[test]
    fn ntt_kernel_4() {
        let table = TableAVX2::new();

        let r = 2u64.pow(32);

        let to_montgomery = |a: i64| ((a as u64 * r) % table.q as u64) as i64;

        let (a_j, a_j_t) = unsafe {
            let mut a_j = _mm256_setr_epi64x(
                to_montgomery(2825332433),
                to_montgomery(3362019074),
                to_montgomery(2509869327),
                to_montgomery(1527549775),
            );
            let mut a_j_t = _mm256_setr_epi64x(
                to_montgomery(4119828181),
                to_montgomery(1439360284),
                to_montgomery(2754719283),
                to_montgomery(357587952),
            );
            let cap_s = _mm256_setr_epi64x(
                to_montgomery(3770581993),
                to_montgomery(2870057844),
                to_montgomery(3898594835),
                to_montgomery(2318024136),
            );

            table.cs_butterfly_simd_4(&cap_s, &mut a_j, &mut a_j_t);

            (
                table.montgomery_reduce_64(_mm256_setzero_si256(), a_j),
                table.montgomery_reduce_64(_mm256_setzero_si256(), a_j_t),
            )
        };

        assert_eq!(
            [
                1281256975236828982,
                824160231704762218,
                1601391116647552192,
                1613286033865546758
            ],
            extract_m256i_to_i64(&a_j)
        );
        assert_eq!(
            [
                979046786719355542,
                749542446557696430,
                2010948492306909607,
                1756774469921775392
            ],
            extract_m256i_to_i64(&a_j_t)
        );
    }
}
