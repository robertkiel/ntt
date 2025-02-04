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
        pub fn $name(a: $t, m: $t) -> $t {
            let mut r = (a, m);
            let mut t = (0, 1);

            while r.1 != 0 {
                let q = r.0 / r.1;
                r = (r.1, (r.0 + (m - $mod_mul(q, r.1, m))) % m);
                t = (t.1, (t.0 + (m - $mod_mul(q, t.1, m))) % m);
            }

            t.1
        }
    };
}

make_mod_inv!(u64, mod_inv_u64, mod_mul_u64);
make_mod_inv!(u32, mod_inv_u32, mod_mul_u32);
make_mod_inv!(u16, mod_inv_u16, mod_mul_u16);
