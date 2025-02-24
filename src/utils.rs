//! Utilities to compute values related to NTT.

use num_primes::{BigUint, Verification};
use rand::{self, Rng};

/// Computes the bit-reversal of the given number `n` depending on the `bits` of the maximum expected number of `n`.
///
/// Examples
/// ```rust
/// use ntt::utils::bit_reverse;
///
/// assert_eq!(bit_reverse(2, 2), 1);
/// ```
pub fn bit_reverse(n: u64, bits: u32) -> u64 {
    let shift = (64 - bits) / 2;
    (n << shift).reverse_bits() >> shift
}

/// Finds a prime `p < 2^bits` that has 2^17th roots of unity
pub fn find_prime_n_primitive_root(bits: u32) -> u64 {
    let mut tmp: u64 = 2u64.pow(bits) - 1;
    loop {
        if !Verification::is_prime(&BigUint::from_bytes_be(&tmp.to_be_bytes())) {
            tmp -= 2;
            continue;
        }

        if (tmp - 1) % 2u64.pow(17) == 0 {
            println!("found {tmp}");
            break tmp;
        }

        tmp -= 2;
    }
}

#[macro_export]
macro_rules! make_egcd {
    ($t:ty, $name:ident) => {
        /// Extended euclidean algorithm
        ///
        /// ```rust
        /// use ntt::utils::egcd_i128;
        ///
        /// const A: i128 = 0x1fffffffffe00001;
        /// const B: i128 = 2i128.pow(64);
        /// let (gcd, x, y) = egcd_i128(0x1fffffffffe00001, 2i128.pow(64));
        ///
        /// assert_eq!(gcd, A*x + B*y);
        /// ```
        pub fn $name(a: $t, m: $t) -> ($t, $t, $t) {
            let mut r = (a, m);
            let mut s = (1, 0);
            let mut t = (0, 1);

            while r.1 != 0 {
                let q = r.0 / r.1;
                r = (r.1, (r.0 - q * r.1));
                s = (s.1, (s.0 - q * s.1));
                t = (t.1, (t.0 - q * t.1));
            }

            (r.0, s.0, t.0)
        }
    };
}

make_egcd!(i128, egcd_i128);

#[macro_export]
macro_rules! make_mod_exp {
    ($t:ty, $t_larger:ty, $name:ident) => {
        /// Computes `base ^ exponent mod modulus` using fast exponentiation by squaring.
        ///
        /// The method is expected to be used for precomputable values only.
        pub fn $name(base: $t, mut exp: $t, modulus: $t) -> $t {
            let mut out = 1;

            let mut acc = base;

            while exp > 0 {
                if exp % 2 == 1 {
                    out = ((out as $t_larger * acc as $t_larger) % modulus as $t_larger) as $t;
                }

                acc = ((acc as $t_larger * acc as $t_larger) % modulus as $t_larger) as $t;

                exp >>= 1;
            }

            out
        }
    };
}

make_mod_exp!(u64, u128, mod_exp_u64);
make_mod_exp!(i64, i128, mod_exp_i64);

#[macro_export]
macro_rules! make_find_nth_root {
    ($t: ty, $name:ident, $mod_exp:ident) => {
        /// Finds a nth root of unity.
        ///
        /// This method is expected to be used for precomputable values only.
        pub fn $name(n: $t, modulus: $t) -> $t {
            let mut rand = rand::rng();

            let mut tmp;
            loop {
                tmp = rand.random_range(2..modulus);

                let g = $mod_exp(tmp, (modulus - 1) / n, modulus);

                match $mod_exp(g, n / 2, modulus) {
                    1 => continue,
                    _ => break g,
                }
            }
        }
    };
}

make_find_nth_root!(u64, find_nth_unity_root_u64, mod_exp_u64);
make_find_nth_root!(i64, find_nth_unity_root_i64, mod_exp_i64);

#[derive(Debug, Eq, PartialEq)]
pub struct MontgomeryTransformer<O> {
    /// Value such that `q * k + 2^bit_len(O) * r' = 1` for some integer r.
    /// Needed for Montgomery multiplication
    pub k: O,
    /// Precomputed value `( 2^bit_len(O) ^ 2 ) mod q`.
    /// Needed for quick Montgomery transformation
    pub r_square: O,
    modulus: O,
}

#[macro_export]
macro_rules! impl_montgomery_transformer {
    ($t: ty) => {
        impl MontgomeryTransformer<$t> {
            pub fn from_modulus(modulus: $t) -> Self {
                let (gcd, x, _) = egcd_i128(modulus as i128, 2i128.pow(64));

                if gcd != 1 {
                    panic!("modulus is not coprime to helper modulus R=2^64")
                }

                let k = if x < 0 {
                    (x + modulus as i128) as $t
                } else {
                    x as $t
                };

                let r_mod_n = 2u128.pow(64) % modulus as u128;
                let r_square = ((r_mod_n * r_mod_n) % modulus as u128) as $t;

                Self {
                    k,
                    r_square,
                    modulus,
                }
            }

            /// Montgomery reduction using R=2^64-1
            pub fn reduce(&self, a: u128) -> $t {
                // m = ((a mod 2^64) * k) mod 2^64
                let m = ((a & 0xffffffffffffffff) * self.k as u128) & 0xffffffffffffffff;

                let m_n = m * self.modulus as u128;

                let y = (a.overflowing_sub(m_n).0 >> 64) as $t;

                if a < m_n {
                    y.overflowing_add(self.modulus).0
                } else {
                    y
                }
            }

            /// Transforms a into Montgomery form using R=2^64-1
            pub fn transform(&self, a: $t) -> $t {
                self.reduce(a as u128 * self.r_square as u128)
            }

            pub fn mul(&self, a: $t, b: $t) -> $t {
                self.reduce(a as u128 * b as u128)
            }
        }
    };
}

impl_montgomery_transformer!(u64);
impl_montgomery_transformer!(i64);

#[cfg(test)]
mod tests {
    use super::MontgomeryTransformer;

    #[test]
    fn montgomery_params_generation() {
        let params = MontgomeryTransformer::<u64>::from_modulus(0x1fffffffffe00001u64);

        assert_eq!(
            MontgomeryTransformer {
                k: 6917533425689690113,
                r_square: 281474708275264,
                modulus: 2305843009211596801
            },
            params
        );
    }

    #[test]
    fn montgomery_transformation() {
        let params = MontgomeryTransformer::<u64>::from_modulus(0x1fffffffffe00001u64);

        let values = [1u64, 16777208u64];

        for val in values {
            assert_eq!(val, params.reduce(params.transform(val) as u128));
        }
    }
}

// #[test]
// fn montgomery_transformation() {
//     let table = Table::<u64>::new();

//     let values = [1u64, 16777208u64];

//     for val in values {
//         assert_eq!(
//             val,
//             table.montgomery_reduce(table.to_montgomery(val) as u128)
//         );
//     }
// }

// #[test]
// fn montgomery_transformation_goldilock() {
//     let table = Table::<u64>::new_goldilock();

//     let values = [1u64, 16777208u64];

//     for val in values {
//         assert_eq!(
//             val,
//             table.montgomery_reduce(table.to_montgomery(val) as u128)
//         );
//     }
// }

// #[test]
// fn montgomery_transformation_u32_compatible() {
//     let table = Table::<u64>::new_u32_compatible();

//     let values = [1u64, 16777208u64];

//     for val in values {
//         assert_eq!(
//             val,
//             table.montgomery_reduce(table.to_montgomery(val) as u128)
//         );
//     }
// }

// #[test]
// fn montgomery_transformation_simple() {
//     let table = Table::<u64>::new_simple();

//     let values = [1u64, 1904u64];

//     for val in values {
//         assert_eq!(
//             val,
//             table.montgomery_reduce(table.to_montgomery(val) as u128)
//         );
//     }
// }
