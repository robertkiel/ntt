use rand::{self, Rng};

use crate::dft::DFT;
use crate::utils::bit_reverse;
use std::arch::x86_64::*;

const DOUBLE_PRIME: f64 = 4503599626321921.0;
const U32_PRIME: u32 = 4293918721;

pub struct TableAVX {
    /// The NTT modulus. This implementation support primes with <=32 bits
    pub q: u32,
    /// N-th root of unity
    pub psi: u32,
    /// Which root of unity psi is
    n: usize,
    // ------ internals, mostly caching --------
    power: __m256d,
    /// q as AVX2 integer vector
    q_vec: __m256i,
    /// q as AVX2 float vector
    q_vec_float: __m256d,
    /// powers of psi in byte order
    powers_psi_bo: Vec<u32>,
    /// powers of psi in byte order, batches in AVX2 structs
    /// i, i + 1, i + 2, i + 3
    powers_psi_bo_chunks_4: Vec<__m256i>,
    /// powers of psi in byte order, batches in AVX2 structs
    /// i, i, i + 1, i + 1
    powers_psi_bo_chunks_2: Vec<__m256i>,
    /// powers of psi in byte order, batches in AVX2 structs
    /// i, i, i, i (4 times some value)
    powers_psi_bo_chunks_1: Vec<__m256i>,

    powers_psi_inv_bo_chunks_1: Vec<__m256i>,
}

impl DFT<u32> for TableAVX {
    /// NTT forward routine
    ///
    /// - `a`: vector with each element in range `[0, q)`
    fn forward_inplace(&self, a: &mut [u32]) {
        self.forward_inplace_core::<false>(a)
    }

    /// NTT forward lazy routine
    ///
    /// - `a`: vector with each element in range `[0, 2q)`
    fn forward_inplace_lazy(&self, a: &mut [u32]) {
        self.forward_inplace_core::<true>(a)
    }

    /// NTT backward routine
    ///
    /// - `a`: vector with each element in range `[0, q)`
    fn backward_inplace(&self, a: &mut [u32]) {
        self.backward_inplace_core::<false>(a)
    }

    /// NTT backward lazy routine
    ///
    /// - `a`: vector with each element in range `[0, 2q)`
    fn backward_inplace_lazy(&self, a: &mut [u32]) {
        self.backward_inplace_core::<true>(a)
    }
}

impl TableAVX {
    pub fn new() -> Self {
        let n = 2usize.pow(16);
        let mut res = Self {
            q: U32_PRIME,
            power: unsafe {
                _mm256_set_pd(
                    2.0f64.powi(32),
                    2.0f64.powi(32),
                    2.0f64.powi(32),
                    2.0f64.powi(32),
                )
            },
            q_vec: unsafe {
                _mm256_set_epi64x(
                    U32_PRIME as i64,
                    U32_PRIME as i64,
                    U32_PRIME as i64,
                    U32_PRIME as i64,
                )
            },
            q_vec_float: unsafe {
                _mm256_set_pd(
                    U32_PRIME as f64,
                    U32_PRIME as f64,
                    U32_PRIME as f64,
                    U32_PRIME as f64,
                )
            },
            psi: 2004365341,
            n,
            powers_psi_bo: Vec::with_capacity(n),
            powers_psi_bo_chunks_4: Vec::with_capacity(n / 4),
            powers_psi_bo_chunks_2: Vec::with_capacity(n / 2),
            powers_psi_bo_chunks_1: Vec::with_capacity(n),
            powers_psi_inv_bo_chunks_1: Vec::with_capacity(n),
        };

        res.with_precomputes();

        res
    }

    /// Finds a nth root of unity. Can be computed once and then cached.
    fn find_nth_unity_root(&self, n: u64, m: u64) -> u64 {
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

    fn mod_exp(&self, base: u64, mut exp: u64) -> u64 {
        let mut out = 1;

        let mut acc = base;

        while exp > 0 {
            if exp % 2 == 1 {
                out = (out * acc) % self.q as u64;
            }

            acc = (acc as u64 * acc as u64) % self.q as u64;

            exp >>= 1;
        }

        out
    }

    /// Computes the forward NTT using AVX2
    pub fn forward_inplace_core<const LAZY: bool>(&self, a: &mut [u32]) {
        let a_len = a.len();

        let mut t = a_len;
        let mut m = 1;

        loop {
            if m >= a_len {
                break;
            }

            t = t / 2;

            if t == 1 && m >= 4 {
                let mut i = 0;

                loop {
                    if i >= m {
                        break;
                    }

                    unsafe {
                        let mut a_j = _mm256_setr_epi64x(
                            a[2 * i] as i64,
                            a[2 * (i + 1)] as i64,
                            a[2 * (i + 2)] as i64,
                            a[2 * (i + 3)] as i64,
                        );

                        let mut a_j_t = _mm256_setr_epi64x(
                            a[2 * i + 1] as i64,
                            a[2 * (i + 1) + 1] as i64,
                            a[2 * (i + 2) + 1] as i64,
                            a[2 * (i + 3) + 1] as i64,
                        );

                        self.ntt_kernel_4(
                            &self.powers_psi_bo_chunks_4[(m + i) / 4],
                            &mut a_j,
                            &mut a_j_t,
                        );

                        let a_j = extract_m256i_to_u64(&a_j);
                        a[2 * i] = a_j[0] as u32;
                        a[2 * (i + 1)] = a_j[1] as u32;
                        a[2 * (i + 2)] = a_j[2] as u32;
                        a[2 * (i + 3)] = a_j[3] as u32;

                        let a_j_t = extract_m256i_to_u64(&a_j_t);
                        a[2 * i + 1] = a_j_t[0] as u32;
                        a[2 * (i + 1) + 1] = a_j_t[1] as u32;
                        a[2 * (i + 2) + 1] = a_j_t[2] as u32;
                        a[2 * (i + 3) + 1] = a_j_t[3] as u32;
                    }

                    i += 4;
                }
            } else if t == 2 && m >= 4 {
                let mut i = 0;

                loop {
                    if i >= m {
                        break;
                    }

                    unsafe {
                        let mut a_j = _mm256_setr_epi64x(
                            a[2 * i * 2] as i64,
                            a[2 * i * 2 + 1] as i64,
                            a[2 * (i + 1) * 2] as i64,
                            a[2 * (i + 1) * 2 + 1] as i64,
                        );

                        let mut a_j_t = _mm256_setr_epi64x(
                            a[2 * i * 2 + 2] as i64,
                            a[2 * i * 2 + 1 + 2] as i64,
                            a[2 * (i + 1) * 2 + 2] as i64,
                            a[2 * (i + 1) * 2 + 1 + 2] as i64,
                        );

                        self.ntt_kernel_4(
                            &self.powers_psi_bo_chunks_2[(m + i) / 2],
                            &mut a_j,
                            &mut a_j_t,
                        );

                        let a_j = extract_m256i_to_u64(&a_j);
                        a[2 * i * 2] = a_j[0] as u32;
                        a[2 * i * 2 + 1] = a_j[1] as u32;
                        a[2 * (i + 1) * 2] = a_j[2] as u32;
                        a[2 * (i + 1) * 2 + 1] = a_j[3] as u32;

                        let a_j_t = extract_m256i_to_u64(&a_j_t);
                        a[2 * i * 2 + 2] = a_j_t[0] as u32;
                        a[2 * i * 2 + 1 + 2] = a_j_t[1] as u32;
                        a[2 * (i + 1) * 2 + 2] = a_j_t[2] as u32;
                        a[2 * (i + 1) * 2 + 1 + 2] = a_j_t[3] as u32;
                    }

                    i += 2;
                }
            } else {
                for i in 0..m {
                    let j_1 = 2 * i * t;
                    let j_2 = j_1 + t - 1;

                    let cap_s_vec = self.powers_psi_bo_chunks_1[(m + i) as usize];
                    let cap_s = self.powers_psi_bo[(m + i) as usize];

                    for mut j in j_1..=j_2 {
                        loop {
                            if t < 4 || j >= j_2 {
                                break;
                            }

                            unsafe {
                                let mut a_j = _mm256_setr_epi64x(
                                    a[j] as i64,
                                    a[j + 1] as i64,
                                    a[j + 2] as i64,
                                    a[j + 3] as i64,
                                );

                                let mut a_j_t = _mm256_setr_epi64x(
                                    a[j + t] as i64,
                                    a[j + t + 1] as i64,
                                    a[j + t + 2] as i64,
                                    a[j + t + 3] as i64,
                                );

                                self.ntt_kernel_4(&cap_s_vec, &mut a_j, &mut a_j_t);

                                let a_j = extract_m256i_to_u64(&a_j);
                                a[j] = a_j[0] as u32;
                                a[j + 1] = a_j[1] as u32;
                                a[j + 2] = a_j[2] as u32;
                                a[j + 3] = a_j[3] as u32;

                                let a_j_t = extract_m256i_to_u64(&a_j_t);
                                a[j + t] = a_j_t[0] as u32;
                                a[j + t + 1] = a_j_t[1] as u32;
                                a[j + t + 2] = a_j_t[2] as u32;
                                a[j + t + 3] = a_j_t[3] as u32;
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
    }

    pub fn backward_inplace_core<const LAZY: bool>(&self, a: &mut [u32]) {}

    /// Computes the CT butterfly
    fn ntt_kernel_1(&self, cap_s: u32, mut a_j: u32, mut a_j_t: u32) -> (u32, u32) {
        // println!("cap_s {cap_s} a_j {a_j} a_j_t {a_j_t}");
        // classic
        let cap_u = a_j as i64;
        let cap_v = self.mul_reduce(a_j_t, cap_s) as i64;

        let mut cap_u_add_cap_v = cap_u + cap_v;
        if cap_u_add_cap_v >= self.q as i64 {
            cap_u_add_cap_v -= self.q as i64;
        }
        a_j = cap_u_add_cap_v as u32;

        let mut cap_u_sub_cap_v = cap_u - cap_v;
        if cap_u_sub_cap_v < 0 {
            cap_u_sub_cap_v += self.q as i64;
        }

        a_j_t = cap_u_sub_cap_v as u32;

        // println!("res a_j {a_j} a_j_t {a_j_t}");

        (a_j, a_j_t)
    }

    /// Computes the CT butterfly using AVX2.
    #[inline]
    unsafe fn ntt_kernel_4(&self, cap_s: &__m256i, a_j: &mut __m256i, a_j_t: &mut __m256i) {
        let cap_u = *a_j;

        let cap_v = self.mul_reduce_vec(a_j_t, &cap_s);

        let mut cap_u_add_cap_v = _mm256_add_epi64(cap_u, cap_v);

        self.reduce_if_greater_equal_q(&mut cap_u_add_cap_v);

        *a_j = cap_u_add_cap_v;

        let mut cap_u_sub_cap_v = _mm256_sub_epi64(cap_u, cap_v);
        self.reduce_if_negative(&mut cap_u_sub_cap_v);

        *a_j_t = cap_u_sub_cap_v;
    }

    /// Computes `a * b mod q` using AVX2 instructions
    #[inline]
    unsafe fn mul_reduce_vec(&self, a: &__m256i, b: &__m256i) -> __m256i {
        // product = a * b
        let product = _mm256_mul_epu32(*a, *b);

        // shifted = product / 2^32
        let shifted = _mm256_srli_epi64(product, 32);

        // products_above_threshold = shifted > 2^32 - 1 ? 0xFF..FF : 0
        let products_above_threshold = _mm256_cmpgt_epi64(
            shifted,
            _mm256_setr_epi64x(0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff),
        );

        // shifted_corrected = shifted > 2^32 - 1 ? shifted - 2^32 : shifted
        let shifted_corrected = _mm256_andnot_si256(
            _mm256_setr_epi64x(0x80000000, 0x80000000, 0x80000000, 0x80000000),
            shifted,
        );

        // shifted_float = float(shifted_corrected)
        let shifted_float = m256i_to_m256d(&shifted_corrected);

        // shifted_float = shifted > 2^32-1 ? shifted_float + 2^32 : shifted_float
        let shifted_float = _mm256_add_pd(
            shifted_float,
            _mm256_and_pd(
                _mm256_castsi256_pd(products_above_threshold),
                _mm256_setr_pd(2147483648.0, 2147483648.0, 2147483648.0, 2147483648.0),
            ),
        );

        // shifted_float_shifted = shifted_float * 2^32
        let shifted_float_shifted = _mm256_mul_pd(shifted_float, self.power);

        // divied = round_to_next_int(shifted_float_shifted / self.q)
        let divided =
            _mm256_round_pd::<0x00>(_mm256_div_pd(shifted_float_shifted, self.q_vec_float));

        // masked = divided > 2^32 - 1 ? 0xFF..FF : 0
        let masked = _mm256_cmp_pd::<_CMP_GT_OQ>(
            divided,
            _mm256_setr_pd(2147483647.0, 2147483647.0, 2147483647.0, 2147483647.0),
        );

        // divided_correct = divided > 2^32 -1 ? divided - 2^32 : divided
        let divided_corrected = _mm256_sub_pd(
            divided,
            _mm256_and_pd(
                masked,
                _mm256_setr_pd(2147483648.0, 2147483648.0, 2147483648.0, 2147483648.0),
            ),
        );

        // quotient = int(divided_corrected)
        let quotient = m256d_to_m256i(&divided_corrected);

        // quotient_corrected = divided > 2^32 -1 ? quotient + 2^32 : quotient
        let quotient_corrected = _mm256_add_epi64(
            _mm256_and_si256(
                _mm256_castpd_si256(masked),
                _mm256_setr_epi64x(0x80000000, 0x80000000, 0x80000000, 0x80000000),
            ),
            quotient,
        );

        // quotient_mul_q = quotient_corrected * self.q
        let quotient_mul_q = _mm256_mul_epu32(quotient_corrected, self.q_vec);

        // subtracted = product - quotient_mul_q;
        let mut subtracted = _mm256_sub_epi64(product, quotient_mul_q);

        // subtracted = subtracted % self.q
        self.reduce_if_negative(&mut subtracted);
        self.reduce_if_greater_equal_q(&mut subtracted);

        subtracted
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
        *a = _mm256_add_epi64(*a, masked);
    }

    /// Adds all kind of precomputes, mainly storing precomputed values batched in AVX2-compatible structs
    fn with_precomputes(&mut self) {
        let psi_inv = self.mod_exp(self.psi as u64, (self.q - 2) as u64) as u32;

        let mut tmp_psi = 1;
        let mut tmp_psi_inv = 1;

        let mut tmp_powers_of_psi_bo = Vec::<u32>::with_capacity(self.n as usize);
        let mut tmp_powers_of_psi_inv_bo = Vec::<u32>::with_capacity(self.n as usize);

        tmp_powers_of_psi_bo.resize(self.n as usize, 0);
        tmp_powers_of_psi_inv_bo.resize(self.n as usize, 0);

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
            self.powers_psi_bo_chunks_1.push(unsafe {
                _mm256_setr_epi64x(
                    tmp_powers_of_psi_bo[i] as i64,
                    tmp_powers_of_psi_bo[i] as i64,
                    tmp_powers_of_psi_bo[i] as i64,
                    tmp_powers_of_psi_bo[i] as i64,
                )
            });
            self.powers_psi_inv_bo_chunks_1.push(unsafe {
                _mm256_setr_epi64x(
                    tmp_powers_of_psi_inv_bo[i] as i64,
                    tmp_powers_of_psi_inv_bo[i] as i64,
                    tmp_powers_of_psi_inv_bo[i] as i64,
                    tmp_powers_of_psi_inv_bo[i] as i64,
                )
            })
        }

        for i in (0..self.n).step_by(4) {
            self.powers_psi_bo_chunks_4.push(unsafe {
                _mm256_setr_epi64x(
                    tmp_powers_of_psi_bo[i] as i64,
                    tmp_powers_of_psi_bo[i + 1] as i64,
                    tmp_powers_of_psi_bo[i + 2] as i64,
                    tmp_powers_of_psi_bo[i + 3] as i64,
                )
            })
        }

        for i in (0..self.n).step_by(2) {
            self.powers_psi_bo_chunks_2.push(unsafe {
                _mm256_setr_epi64x(
                    tmp_powers_of_psi_bo[i] as i64,
                    tmp_powers_of_psi_bo[i] as i64,
                    tmp_powers_of_psi_bo[i + 1] as i64,
                    tmp_powers_of_psi_bo[i + 1] as i64,
                )
            })
        }
    }

    /// Computes a * b mod self.q
    #[inline]
    fn mul_reduce(&self, a: u32, b: u32) -> u32 {
        ((a as u64 * b as u64) % self.q as u64) as u32
    }
}

#[inline]
fn extract_m256i_to_u64(a: &__m256i) -> [u64; 4] {
    let mut res = [0u64; 4];
    unsafe {
        res[0] = _mm256_extract_epi64::<0>(*a) as u64;
        res[1] = _mm256_extract_epi64::<1>(*a) as u64;
        res[2] = _mm256_extract_epi64::<2>(*a) as u64;
        res[3] = _mm256_extract_epi64::<3>(*a) as u64;
    }

    res
}

#[inline]
fn extract_m256d_to_f64(a: &__m256d) -> [f64; 4] {
    let mut res = [0.0f64; 4];
    unsafe { _mm256_storeu_pd(res.as_mut_ptr(), *a) };

    res
}

#[inline]
fn m256d_to_m256i(a: &__m256d) -> __m256i {
    unsafe {
        let packed = _mm256_cvtpd_epi32(*a);

        let unpacked_raw = _mm256_set_m128i(
            _mm_shuffle_epi32::<0b11_00_10_00>(packed),
            _mm_shuffle_epi32::<0b01_00_00_00>(packed),
        );

        _mm256_srli_epi64(unpacked_raw, 32)
    }
}

#[inline]
fn m256i_to_m256d(a: &__m256i) -> __m256d {
    unsafe {
        let shuffled = _mm_or_si128(
            // _mm_set_epi32(0, 0, 0, 0),
            _mm256_extracti128_si256(_mm256_shuffle_epi32::<0b10_00_11_01>(*a), 1),
            //
            // _mm256_extracti128_si256
            _mm256_castsi256_si128(_mm256_shuffle_epi32::<0b11_01_10_00>(*a)),
        );

        _mm256_cvtepi32_pd(shuffled)
    }
}

#[cfg(test)]
mod tests {
    use std::arch::x86_64::*;

    use super::{
        extract_m256d_to_f64, extract_m256i_to_u64, m256d_to_m256i, m256i_to_m256d, TableAVX,
    };

    #[test]
    fn extract_u64() {
        let data = unsafe { _mm256_setr_epi64x(1, 2, 3, 4) };

        let res = extract_m256i_to_u64(&data);

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

        assert_eq!(extract_m256i_to_u64(&res), [1, 2, 3, 4]);
    }

    #[test]
    fn m256i_to_m256d_conversion() {
        let data = unsafe { _mm256_setr_epi64x(1, 2, 3, 4) };

        let res = m256i_to_m256d(&data);

        assert_eq!(extract_m256d_to_f64(&res), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn reduce_if_negative() {
        let table = TableAVX::new();
        let control = unsafe {
            let mut res = _mm256_setr_epi64x(-1, -2, -3, -4);

            table.reduce_if_negative(&mut res);

            let mut control = _mm256_setzero_si256();

            _mm256_store_si256(&mut control, res);

            control
        };

        assert_eq!(
            [
                (table.q - 1) as u64,
                (table.q - 2) as u64,
                (table.q - 3) as u64,
                (table.q - 4) as u64
            ],
            extract_m256i_to_u64(&control)
        );
    }

    #[test]
    fn reduce_if_greate_equal_q() {
        let table = TableAVX::new();
        let control = unsafe {
            let mut res = _mm256_setr_epi64x(
                (table.q + 1) as i64,
                table.q as i64,
                (table.q + 3) as i64,
                (table.q + 4) as i64,
            );

            table.reduce_if_greater_equal_q(&mut res);

            let mut control = _mm256_setzero_si256();

            _mm256_store_si256(&mut control, res);

            control
        };

        assert_eq!([1, 0, 3, 4], extract_m256i_to_u64(&control))
    }

    // 3616826132
    // 678141164
    #[test]
    fn mul_reduce_vec() {
        let table = TableAVX::new();
        let control = unsafe {
            let a = _mm256_setr_epi64x(4119828181, 1439360284, 1073741824, 1073741824);
            let b = _mm256_setr_epi64x(3770581993, 2870057844, 1073741824, 1073741824);

            let res = table.mul_reduce_vec(&a, &b);

            let mut control = _mm256_setzero_si256();

            _mm256_store_si256(&mut control, res);
            control
        };

        assert_eq!(
            [
                1197697452,
                2614870421,
                table.mul_reduce(1073741824, 1073741824) as u64,
                table.mul_reduce(1073741824, 1073741824) as u64,
            ],
            extract_m256i_to_u64(&control)
        );

        println!("res {:?}", control);
    }

    #[test]
    fn ntt_kernel_4() {
        let table = TableAVX::new();

        let (a_j, a_j_t) = unsafe {
            let mut a_j = _mm256_setr_epi64x(2825332433, 3362019074, 2509869327, 4246358974);
            let mut a_j_t = _mm256_setr_epi64x(4119828181, 1439360284, 2754719283, 3826442627);
            let cap_s = _mm256_setr_epi64x(3770581993, 2870057844, 3898594835, 2313218121);

            table.ntt_kernel_4(&cap_s, &mut a_j, &mut a_j_t);

            (a_j, a_j_t)
        };

        assert_eq!(
            [4023029885, 1682970774, 2185318912, 2519542288],
            extract_m256i_to_u64(&a_j)
        );
        assert_eq!(
            [1627634981, 747148653, 2834419742, 1679256939],
            extract_m256i_to_u64(&a_j_t)
        );
    }
}
