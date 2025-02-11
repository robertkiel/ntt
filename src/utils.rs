use num_primes::{BigUint, Verification};

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

#[ignore = "no need to find the prime twice"]
#[test]
fn find_prime() {
    find_prime_n_primitive_root(32);
}
