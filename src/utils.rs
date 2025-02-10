use num_primes::{BigUint, Verification};

#[macro_export]
macro_rules! make_mod_mul {
    ($t:ty, $name:ident) => {
        /// Computes `a * b mod m`.
        ///
        /// Multiplies `a` and `b` by adding `b`-times `a` using the fast [exponentiation algorithm](https://en.wikipedia.org/wiki/Exponentiation_by_squaring)
        /// using `log(b)` additions and applying the modulus after each addition.
        ///
        /// This prevents the use of bigger types, such as u128, which need to be emulated on off-the-shelf 64-bit machines
        #[inline]
        pub fn $name(a: $t, mut b: $t, m: $t) -> $t {
            let mut out = 0;

            let mut acc = a;

            while b > 0 {
                if b % 2 == 1 {
                    out = (out + acc) % m;
                    b -= 1;
                }

                acc = (acc + acc) % m;

                b >>= 1;
            }

            out
        }
    };
}

make_mod_mul!(u128, mod_mul_u128);
make_mod_mul!(u64, mod_mul_u64);
make_mod_mul!(u32, mod_mul_u32);
make_mod_mul!(u16, mod_mul_u16);

#[macro_export]
macro_rules! make_mod_inv {
    ($t:ty, $name:ident, $mod_mul:ident) => {
        /// Computes `a^-1 mod m` (the multiplicative inverse on the modulus m).
        ///
        /// Performs a slightly simplified version of the extended euclidean algorithm.
        ///
        /// Prevents the use of bigger types, such as u128, by using the
        /// fast [exponentiation algorithm](https://en.wikipedia.org/wiki/Exponentiation_by_squaring) using `log(m)` additions.
        pub fn $name(a: $t, m: $t) -> ($t, $t) {
            let mut r = (a, m);
            //     (old_s, s) := (1, 0)
            let mut s = (1, 0);
            let mut t = (0, 1);

            while r.1 != 0 {
                let q = r.0 / r.1;
                r = (r.1, (r.0 + (m - $mod_mul(q, r.1, m))) % m);
                s = (s.1, (s.0 + (m - $mod_mul(q, s.1, m))) % m);
                t = (t.1, (t.0 + (m - $mod_mul(q, t.1, m))) % m);
            }

            (s.0, t.0)
        }
    };
}

make_mod_inv!(u128, mod_inv_u128, mod_mul_u128);
make_mod_inv!(u64, mod_inv_u64, mod_mul_u64);
make_mod_inv!(u32, mod_inv_u32, mod_mul_u32);
make_mod_inv!(u16, mod_inv_u16, mod_mul_u16);

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

pub fn find_prime_n_primitive_root(bits: usize) -> u64 {
    let mut tmp: u64 = 2u64.pow(32) - 1;
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

#[test]
fn find_prime() {
    find_prime_n_primitive_root(32);
}

#[test]
fn foo() {
    println!(
        "{:?}",
        (0x1fffffffffe00001u128 * 6917533425689690113)
            .overflowing_add(17582052395499126784 * 2u128.pow(64))
    );
    println!("{:?}", mod_inv_u128(0x1fffffffffe00001u128, 2u128.pow(64)));
}

#[test]
fn bar() {
    println!(
        "{:?}",
        (4503599626321921u128 * 1099512676353)
            .overflowing_add(4502500114694399 * 2u128.pow(52))
            .0
            % 2u128.pow(52)
    );
    println!("{:?}", mod_inv_u128(4503599626321921u128, 2u128.pow(52)));
}

pub fn egcd(a: i128, m: i128) -> (i128, i128) {
    let mut r = (a, m);
    //     (old_s, s) := (1, 0)
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

#[test]
fn test_egcd() {
    println!("{:?}", egcd(4503599626321921i128, 2i128.pow(52)));
}
