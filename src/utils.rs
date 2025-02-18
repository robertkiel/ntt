use num_primes::{BigUint, Verification};
use rand::{self, Rng};

/// Computes the bit-reversal of the given number `n` depending on the `bits` of the maximum expected number of `n`.
///
/// Examples
/// ```rust
/// use ntt::utils::bit_reverse;
///
/// assert_eq!(bit_reverse(2, 2), 4);
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

/// Computes `a^-1 mod m` (the multiplicative inverse on the modulus m).
///
/// Performs a slightly simplified version of the extended euclidean algorithm.
///
/// Prevents the use of bigger types, such as u128, by using the
/// fast [exponentiation algorithm](https://en.wikipedia.org/wiki/Exponentiation_by_squaring) using `log(m)` additions.
pub fn egcd(a: i128, m: i128) -> (i128, i128) {
    let mut r = (a, m);
    let mut s = (1, 0);
    let mut t = (0, 1);

    while r.1 != 0 {
        let q = r.0 / r.1;
        r = (r.1, (r.0 - q * r.1));
        s = (s.1, (s.0 - q * s.1));
        t = (t.1, (t.0 - q * t.1));
    }

    (s.0, t.0)
}

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

#[test]
// #[ignore = "no need to invert r twice"]
fn test_egcd() {
    // println!("{}", 4293918721 * 1048577 - 2i128.pow(32) * 1048321);
    // println!("{:?}", egcd(4293918721, 2i128.pow(32)))
    println!("{:?}", egcd(0xffffffff00000001, 2i128.pow(64)));
    // println!("{:?}", 2i128.pow(64) - 5367591850636746239);
}

#[ignore = "no need to find the prime twice"]
#[test]
fn find_prime() {
    find_prime_n_primitive_root(32);
}
